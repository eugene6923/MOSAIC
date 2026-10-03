# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MOSAIC DiT pipeline: SD3 transformer conditioned on one-hot tokens through a small condition encoder."""
import inspect
from typing import List, Optional, Union

import torch
import torch.nn as nn

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL
from ...models.transformers import SD3Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import StableDiffusion3PipelineOutput


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Calls `scheduler.set_timesteps` (with custom `timesteps` or `sigmas` if given) and returns (timesteps, num_inference_steps)."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        if "timesteps" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError(f"{scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if "sigmas" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError(f"{scheduler.__class__}'s `set_timesteps` does not support custom sigmas schedules.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class MyCustomPipeline_dit(DiffusionPipeline, StableDiffusionMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        condition_encoder: nn.Module = None,
    ):
        super().__init__()
        self.register_modules(vae=vae, transformer=transformer, scheduler=scheduler, condition_encoder=condition_encoder)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        if isinstance(transformer, torch.nn.parallel.DistributedDataParallel):
            transformer = transformer.module
        self.default_sample_size = self.transformer.config.sample_size if getattr(self, "transformer", None) is not None else 128
        self.patch_size = self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 2

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # SD3 doesn't scale by init_noise_sigma
        else:
            latents = latents.to(device)
        return latents

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def encode_condition(self, condition, device):
        """One-hot tokens (B, T, 10) -> (B, T, C); the unconditional branch for CFG uses all-zero tokens."""
        if self.do_classifier_free_guidance:
            condition_embeds = torch.cat([self.condition_encoder(torch.zeros_like(condition).to(device)), self.condition_encoder(condition.to(device))])
        else:
            condition_embeds = self.condition_encoder(condition.to(device))
        if condition_embeds.dim() == 2:
            condition_embeds = condition_embeds.unsqueeze(1)
        elif condition_embeds.dim() == 4:
            condition_embeds = condition_embeds.squeeze(2)
        assert condition_embeds.dim() == 3, f"Condition embeddings should be (batch, seq_len, dim), got {condition_embeds.shape}"
        return condition_embeds

    @torch.no_grad()
    def __call__(
        self,
        condition: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        batch_size = condition.shape[0]
        device = self._execution_device

        condition_embeds = self.encode_condition(condition, device)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            height,
            width,
            condition_embeds.dtype,
            device,
            generator,
            latents,
        )

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=condition_embeds,
                    pooled_projections=None,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)  # some platforms (eg. apple mps) misbehave due to a pytorch bug
                progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image.detach(), output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return StableDiffusion3PipelineOutput(images=image)
