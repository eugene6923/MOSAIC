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
import os
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import json
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...configuration_utils import FrozenDict
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, ImageProjection
from ...models.transformers import SD3Transformer2DModel
from ...models.lora import adjust_lora_scale_text_encoder
from ...pipelines import StableDiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import StableDiffusion3PipelineOutput

from model import condition_encoder

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class MyCustomPipeline_dit(DiffusionPipeline, StableDiffusionMixin):

    def __init__(self, 
        vae: AutoencoderKL,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        condition_encoder: condition_encoder=None,
        condition_encoder2: condition_encoder=None,
        feature_extractor: CLIPImageProcessor=None,
        image_encoder: CLIPVisionModelWithProjection=None,
        requires_safety_checker: bool = False,
        id_condition: bool = False,
        ):
        
        super().__init__()

        

        modules_to_register = {
            'vae': vae,
            'transformer': transformer,
            'scheduler': scheduler,
            'feature_extractor': feature_extractor,
            'image_encoder': image_encoder,
            'condition_encoder': condition_encoder
        }

        if condition_encoder2 is not None:
            modules_to_register['condition_encoder2'] = condition_encoder2
        
        self.register_modules(**modules_to_register)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker,
            id_condition=id_condition)
        if isinstance(transformer, torch.nn.parallel.DistributedDataParallel):
            transformer = transformer.module
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )
        self.id_condition = id_condition
        
        # CLIP text-embedding lookup tables (only needed when text_embedding mode is used);
        # loaded lazily on first use, see _get_clip_embeddings.
        self.clip_embeddings_cache = {}

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # SD3 doesn't scale by init_noise_sigma
        else:
            latents = latents.to(device)

        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def guidance_scale(self):
        return self._guidance_scale
    
    @property
    def guidance_rescale(self):
        return self._guidance_rescale
    
    @property
    def clip_skip(self):
        return self._clip_skip

    def run_safety_checker(self, image, device, dtype):
        if torch.is_tensor(image):
            feature_extractor_input = self.image_processor.postprocess(image.detach(), output_type="pil")
        else:
            feature_extractor_input = self.image_processor.numpy_to_pil(image)
        has_nsfw_concept = None

        return feature_extractor_input, has_nsfw_concept

    _CLIP_EMBEDDING_FILES = {
        "counting": "clip_embedding_counting.json",
        "position": "clip_embedding_position.json",
        "attribute": "clip_embedding_attribute.json",
        "counting_ood": "clip_embedding_counting_ood.json",
        "position_ood": "clip_embedding_position_ood.json",
        "object": "clip_embedding_counting_cars.json",
    }

    def _get_clip_embeddings(self, name):
        """Load (once) and return the pre-computed CLIP embedding table for `name`."""
        if name not in self.clip_embeddings_cache:
            path = self._CLIP_EMBEDDING_FILES[name]
            try:
                with open(path, "r") as f:
                    self.clip_embeddings_cache[name] = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{path} is required for text_embedding mode '{name}' but was not found in {os.getcwd()}"
                )
        return self.clip_embeddings_cache[name]

    def __call__(self,
        condition: Union[str, List[str]] = None,
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
        start_timestep: Optional[int] = None,  # starting timestep for partial denoising
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        interpolation = False,
        text_embedding = None,
        **kwargs,):

            # 0. Default height and width to transformer
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks
        # 1. Check inputs. Raise error if not correct

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        
        if condition is not None and isinstance(condition, float):
            batch_size = 1
        elif condition is not None and isinstance(condition, list):
            batch_size = len(condition)
        elif interpolation and isinstance(condition, torch.Tensor):
            batch_size = condition.shape[0] // 2
        elif isinstance(condition, tuple):
            batch_size = condition[1].shape[0]
        else:
            batch_size = condition.shape[0]

        device = self._execution_device


        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes

        has_condition_encoder2 = hasattr(self, 'condition_encoder2') and self.condition_encoder2 is not None

        if self.condition_encoder is not None and not has_condition_encoder2:
            if self.do_classifier_free_guidance:
                condition_embeds = torch.cat([self.condition_encoder(torch.zeros(condition.size()).to(device)), self.condition_encoder(condition.to(device))])
            else:
                if isinstance(condition, tuple):
                    condition_embeds = self.condition_encoder(condition)
                else: condition_embeds = self.condition_encoder(condition.to(device))

            if not self.id_condition:
                if condition_embeds.dim() == 1:
                    # if the condition encoder returns a single vector, we need to unsqueeze it
                    # to match the expected shape of (batch_size, sequence_length, hidden_size)
                    condition_embeds = condition_embeds.unsqueeze(0).unsqueeze(0)
                elif condition_embeds.dim() == 2:
                    # if the condition encoder returns a 2D tensor, we need to unsqueeze it
                    # to match the expected shape of (batch_size, sequence_length, hidden_size)
                    condition_embeds = condition_embeds.unsqueeze(1)
                elif condition_embeds.dim() == 4:
                    # if the condition encoder returns a 4D tensor, we need to unsqueeze it
                    # to match the expected shape of (batch_size, sequence_length, height, width)
                    condition_embeds = condition_embeds.squeeze(2)
                assert condition_embeds.dim() == 3, f'Condition embeddings should be 3D, but got {condition_embeds.dim()} dimensions. Shape: {condition_embeds.shape}'
            else:
                if condition_embeds.dim() == 3:
                    # if the condition encoder returns a 3D tensor, we need to unsqueeze it
                    # to match the expected shape of (batch_size, sequence_length, hidden_size)
                    condition_embeds = condition_embeds.squeeze(1)
                elif condition_embeds.dim() == 4:
                    condition_embeds = condition_embeds.squeeze(1).squeeze(1)
                assert condition_embeds.dim() == 2, f'Condition embeddings should be 2D, but got {condition_embeds.dim()} dimensions. Shape: {condition_embeds.shape}'
        
        elif self.condition_encoder is not None and self.condition_encoder2 is not None:
            assert condition[0].shape[1] == 2
            if condition[1].dim() == 3:
                position = condition[1].squeeze(1)

            encoder_hidden_states1 = self.condition_encoder(condition[0][:, 0].to(device))
            encoder_hidden_states2 = self.condition_encoder(condition[0][:, 1].to(device))
            encoder_hidden_states3 = self.condition_encoder2(position.to(device))
            condition_embeds = torch.cat([encoder_hidden_states1.unsqueeze(1), encoder_hidden_states3.unsqueeze(1), encoder_hidden_states2.unsqueeze(1)], dim=1)
        elif self.condition_encoder is None and text_embedding[0]:
            if "ood" not in text_embedding[1]:
                if "attribute" in text_embedding[1]:
                    clip_embedding_counting = self._get_clip_embeddings('attribute')
                    encoder_hidden_states = []
                    for i in range(condition.shape[0]):
                        # Flatten condition to handle nested dimensions (e.g., (1, 1, 10) -> (10,))
                        key = str((tuple(condition[i][0].cpu().tolist()), tuple(condition[i][1].cpu().tolist())))  # Convert to string for JSON key lookup
                        text_embedding = torch.tensor(clip_embedding_counting[key]["embedding"], device=device)
                        encoder_hidden_states.append(text_embedding)
                    encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
                    condition_embeds = encoder_hidden_states.squeeze(1) 
                elif "position" in text_embedding[1]:
                    clip_embedding_counting = self._get_clip_embeddings('position')
                    encoder_hidden_states = []
                    for i in range(condition.shape[0]):
                        # Flatten condition to handle nested dimensions (e.g., (1, 1, 10) -> (10,))
                        key = str(tuple(condition[i].flatten().tolist()))  # Convert to string for JSON key lookup
                        text_embedding = torch.tensor(clip_embedding_counting[key]["embedding"], device=device)
                        encoder_hidden_states.append(text_embedding)
                    encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
                    condition_embeds = encoder_hidden_states.squeeze(1)
                elif "object" in text_embedding[1]:
                     clip_embedding_counting = self._get_clip_embeddings('object')
                     encoder_hidden_states = []
                     for i in range(condition.shape[0]):
                        # Flatten condition to handle nested dimensions (e.g., (1, 1, 10) -> (10,))
                        key = str(tuple(condition[i].flatten().tolist()))  # Convert to string for JSON key lookup
                        text_embedding = torch.tensor(clip_embedding_counting[key]["embedding"], device=device)
                        encoder_hidden_states.append(text_embedding)
                     encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
                     condition_embeds = encoder_hidden_states.squeeze(1)
                else:
                    clip_embedding_counting = self._get_clip_embeddings('counting')
                    encoder_hidden_states = []
                    for i in range(condition.shape[0]):
                        # Flatten condition to handle nested dimensions (e.g., (1, 1, 10) -> (10,))
                        key = str(tuple(condition[i].flatten().tolist()))  # Convert to string for JSON key lookup
                        text_embedding = torch.tensor(clip_embedding_counting[key]["embedding"], device=device)
                        encoder_hidden_states.append(text_embedding)
                    encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
                    condition_embeds = encoder_hidden_states.squeeze(1)
            else:
                if "position" in text_embedding[1] and "count" not in text_embedding[1]:
                    clip_embedding_counting = self._get_clip_embeddings('position_ood')
                    encoder_hidden_states = []
                    for i in range(condition.shape[0]):
                        # Flatten condition to handle nested dimensions (e.g., (1, 1, 10) -> (10,))
                        key = str((tuple(condition[i][0].cpu().tolist()), tuple(condition[i][1].cpu().tolist())))  # Convert to string for JSON key lookup
                        text_embedding = torch.tensor(clip_embedding_counting[key]["embedding"], device=device)
                        encoder_hidden_states.append(text_embedding)
                    encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
                    condition_embeds = encoder_hidden_states.squeeze(1) 
                elif "attribute" in text_embedding[1]:
                    clip_embedding_counting = self._get_clip_embeddings('attribute')
                    encoder_hidden_states = []
                    for i in range(condition.shape[0]):
                        # Flatten condition to handle nested dimensions (e.g., (1, 1, 10) -> (10,))
                        key = str((tuple(condition[i][0].cpu().tolist()), tuple(condition[i][1].cpu().tolist())))  # Convert to string for JSON key lookup
                        text_embedding = torch.tensor(clip_embedding_counting[key]["embedding"], device=device)
                        encoder_hidden_states.append(text_embedding)
                    encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
                    condition_embeds = encoder_hidden_states.squeeze(1) 
                else:
                    clip_embedding_counting = self._get_clip_embeddings('counting_ood')
                    encoder_hidden_states = []
                    for i in range(condition.shape[0]):
                        # Flatten condition to handle nested dimensions (e.g., (1, 1, 10) -> (10,))

                        key = str((tuple(condition[i][0].cpu().tolist()), tuple(condition[i][1].cpu().tolist())))  # Convert to string for JSON key lookup

                        text_embedding = torch.tensor(clip_embedding_counting[key]["embedding"], device=device)
                        encoder_hidden_states.append(text_embedding)
                    encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
                    condition_embeds = encoder_hidden_states.squeeze(1)    
        else:
            condition_embeds = condition.to(device)
        
        
        if interpolation:
            condition_embeds = 0.5*condition_embeds[:batch_size] + 0.5*condition_embeds[batch_size:]


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        
        # If start_timestep is provided, find the corresponding index and slice timesteps
        
        if start_timestep is not None:
            print(f"Original timesteps: {timesteps}")
            # Find the index of the start_timestep in the timesteps array
            # Since timesteps are in descending order, we want timesteps <= start_timestep
            start_idx = (timesteps <= start_timestep).nonzero(as_tuple=True)[0]
            print(f"Start index: {start_idx}")
            if len(start_idx) > 0:
                start_idx = start_idx[0].item()
                timesteps = timesteps[start_idx:]
                num_inference_steps = len(timesteps)
            else:
                # If start_timestep is larger than all timesteps, start from the beginning
                pass
            print(f"Modified timesteps: {timesteps}")
            print(f"Number of inference steps: {num_inference_steps}")

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels

        
        latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        condition_embeds.dtype,
        device,
        generator,
        latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        # 6.2 Optionally get Guidance Scale Embedding (not used in SD3)
        # SD3 transformer doesn't use time_cond_proj_dim

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict the noise residual using SD3 transformer
                # SD3 uses hidden_states, encoder_hidden_states, and pooled_projections
                if self.condition_encoder is not None and not self.id_condition:
                    # For sequence conditioning (like text)
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=condition_embeds,
                        pooled_projections=None,  # Can be added if you have pooled embeddings
                        joint_attention_kwargs=self.cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    # For class/id conditioning
                    # Note: SD3 transformer expects encoder_hidden_states and pooled_projections
                    # You may need to adapt this based on your conditioning setup
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=condition_embeds.unsqueeze(1) if condition_embeds.dim() == 2 else condition_embeds,
                        pooled_projections=None,
                        joint_attention_kwargs=self.cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug
                        latents = latents.to(latents_dtype)


        if output_type == "latent":
            image = latents
        else:
            # SD3 uses shift_factor in addition to scaling_factor
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image.detach(), output_type=output_type)
        
        has_nsfw_concept = None

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusion3PipelineOutput(images=image)