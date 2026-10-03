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
"""MOSAIC UNet pipeline: UNet2DConditionModel conditioned on one-hot tokens through a small condition encoder."""
import inspect
from typing import List, Optional, Union

import torch
import torch.nn as nn

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import StableDiffusionPipelineOutput


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


class MyCustomPipeline(DiffusionPipeline, StableDiffusionMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        condition_encoder: nn.Module = None,
    ):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler, condition_encoder=condition_encoder)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        if isinstance(unet, torch.nn.parallel.DistributedDataParallel):
            unet = unet.module
        self._is_unet_config_sample_size_int = unet is not None and isinstance(unet.config.sample_size, int)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = latents * self.scheduler.init_noise_sigma  # scale the initial noise by the scheduler's std
        else:
            latents = latents.to(device)
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) is only used with the DDIMScheduler; generator only by schedulers that accept it
        step_params = set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if "eta" in step_params:
            extra_step_kwargs["eta"] = eta
        if "generator" in step_params:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

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
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        if not height or not width:
            sample_size = self.unet.config.sample_size
            height = sample_size if self._is_unet_config_sample_size_int else sample_size[0]
            width = sample_size if self._is_unet_config_sample_size_int else sample_size[1]
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        batch_size = condition.shape[0]
        device = self.unet.device

        condition_embeds = self.encode_condition(condition, device)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            condition_embeds.dtype,
            device,
            generator,
            latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=condition_embeds, return_dict=False)[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image = self.image_processor.postprocess(image.detach(), output_type=output_type, do_denormalize=[True] * image.shape[0])

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, None)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
