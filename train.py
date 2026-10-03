#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Train a conditional UNet or DiT on one MOSAIC task (count / position / attribute)."""

import copy
import logging
import math
import os
import random
import shutil
import signal
import sys
from contextlib import nullcontext

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, broadcast_object_list, set_seed
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

sys.path.append("./diffusers/src")
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    MyCustomPipeline,
    MyCustomPipeline_dit,
    SD3Transformer2DModel,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version
from diffusers.utils.torch_utils import is_compiled_module
import wandb

from data import get_dataset
from model import condition_encoder, two_condition_encoder
from train_args import parse_args as parse_train_args
from train_data import make_collate_fn
from utils import (
    best_model_metric,
    load_classifiers,
    log_accuracy_results,
    task_of,
    test_accuracy,
    uses_two_token_condition,
)
from validation_prompts import build_validation_prompts

check_min_version("0.34.0.dev0")

logger = get_logger(__name__, log_level="INFO")

VALIDATION_SCHEDULERS = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "DDPMScheduler": DDPMScheduler,
}


def format_param_count(param_count):
    """Format parameter count in a readable way (K, M, B)"""
    if param_count >= 1e9:
        return f"{param_count / 1e9:.1f}B"
    elif param_count >= 1e6:
        return f"{param_count / 1e6:.1f}M"
    elif param_count >= 1e3:
        return f"{param_count / 1e3:.1f}K"
    else:
        return str(param_count)


def build_pipeline(args, save_dir, vae, unet, Encoder, scheduler=None):
    """Pipeline around the current (unwrapped) models; loads the rest (scheduler etc.) from save_dir."""
    if args.model == "dit":
        return MyCustomPipeline_dit.from_pretrained(save_dir, vae=vae, transformer=unet, condition_encoder=Encoder)
    kwargs = {} if scheduler is None else {"scheduler": scheduler}
    return MyCustomPipeline.from_pretrained(save_dir, vae=vae, unet=unet, condition_encoder=Encoder, **kwargs)


def log_validation(vae, Encoder, unet, args, accelerator, epoch, save_dir, validation_key=None, no_image=False):
    logger.info("Running validation... ")

    validation_scheduler = None
    if args.model != "dit":
        scheduler_cls = VALIDATION_SCHEDULERS.get(args.validation_scheduler, EulerDiscreteScheduler)
        validation_scheduler = scheduler_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    unet.eval()
    Encoder.eval()
    pipeline = build_pipeline(
        args, save_dir, accelerator.unwrap_model(vae), accelerator.unwrap_model(unet), accelerator.unwrap_model(Encoder), validation_scheduler
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    autocast_ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(accelerator.device.type)

    with autocast_ctx:
        batched_prompts = torch.stack(args.validation_prompts)
        if args.validation_batch_size is None:
            images = pipeline(batched_prompts, num_inference_steps=20, generator=generator).images
        else:
            images = []
            for i in range(0, len(batched_prompts), args.validation_batch_size):
                batch_prompts = batched_prompts[i : i + args.validation_batch_size]
                images.extend(pipeline(batch_prompts, num_inference_steps=20, generator=generator).images)

    if not no_image:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            elif tracker.name == "wandb":
                # one image per unique validation key
                logged_keys = set()
                images_to_log, captions_to_log = [], []
                for i, image in enumerate(images):
                    key = validation_key[i] if validation_key and i < len(validation_key) else "unknown"
                    if isinstance(key, int):
                        caption = f"{i}: {key}_objects" if key else f"{i}: no_objects"
                    else:
                        caption = f"{i}: {key}"
                    if key not in logged_keys:
                        logged_keys.add(key)
                        images_to_log.append(image)
                        captions_to_log.append(caption)
                tracker.log({"validation": [wandb.Image(image, caption=caption) for image, caption in zip(images_to_log, captions_to_log)]})
            else:
                logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()
    unet.train()
    Encoder.train()
    return images


def load_testers(args, device):
    """Pretrained classifiers that score validation images (shared with evaluate.py). Tester2 is None when unused."""
    return load_classifiers(args.test_mode, device, args.resolution)


def main():
    args = parse_train_args()

    if args.report_to == "wandb":

        def handle_termination(signum, frame):
            print("Received SIGTERM. Finishing W&B run...")
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_termination)
        signal.signal(signal.SIGINT, handle_termination)

    if args.validation_epochs is not None:
        args.validation_steps = None

    task_of(args.test_mode)  # validate the test_mode early

    # NOTE: this string names the run folder of every existing checkpoint; keep it stable.
    param_string = (
        f"model{args.model}_resolution{args.resolution}_lr{args.learning_rate}"
        f"{f'_warmup{args.lr_warmup_steps}' if args.lr_scheduler != 'constant' else ''}"
        f"_bs{args.train_batch_size}_ga{args.gradient_accumulation_steps}_sch{args.lr_scheduler}"
        f"_valshc{args.validation_scheduler}_wd{args.adam_weight_decay}"
        f"{'_mixed_precision_' + args.mixed_precision if args.mixed_precision is not None else ''}"
        f"_drop_out{args.drop_rate}_condition_drop_out{args.condition_encoder_drop_rate}"
    )

    save_root = args.save_dir if args.save_dir is not None else ("dit_weights" if args.model == "dit" else "unet_weights")
    save_dir = os.path.join(save_root, "seed_" + str(args.seed), args.test_mode, param_string)
    logging_dir = os.path.join(save_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=save_dir, logging_dir=logging_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("Training/evaluation parameters %s", args)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if accelerator.is_main_process:
        print(args)

    # ------------------------------------------------------------------ models
    if args.model == "dit":
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="scheduler")
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        noise_scheduler_copy = None

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    vae.requires_grad_(False)

    Tester, Tester2 = load_testers(args, accelerator.device) if (accelerator.is_main_process and args.test_accuracy) else (None, None)

    if args.model == "dit":
        condition_encoder_config = {"input_dim": 10, "output_dim": 512, "hidden_dims": [64, 128, 256], "dropout_rate": args.condition_encoder_drop_rate}
    else:
        condition_encoder_config = {"input_dim": 10, "output_dim": 64, "hidden_dims": [], "dropout_rate": args.condition_encoder_drop_rate}
    encoder_cls = two_condition_encoder if uses_two_token_condition(args.test_mode) else condition_encoder
    Encoder = encoder_cls(**condition_encoder_config)
    Encoder.train()

    if args.model == "dit":
        unet = SD3Transformer2DModel(
            sample_size=args.resolution // 8,
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_layers=10,
            attention_head_dim=32,
            num_attention_heads=16,
            joint_attention_dim=condition_encoder_config["output_dim"],
            caption_projection_dim=condition_encoder_config["output_dim"],
            pooled_projection_dim=condition_encoder_config["output_dim"],
        )
        backbone_cls = SD3Transformer2DModel
    else:
        unet = UNet2DConditionModel(
            sample_size=args.resolution // 8,
            in_channels=4,
            out_channels=4,
            block_out_channels=(80, 160, 320, 320),
            layers_per_block=4,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=condition_encoder_config["output_dim"],
            use_linear_projection=False,
            norm_num_groups=16,
        )
        backbone_cls = UNet2DConditionModel

    # accelerate save/load hooks: backbone goes to <ckpt>/unet, encoder to <ckpt>/condition_encoder
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process and not args.no_save:
                for model in models:
                    unwrapped = accelerator.unwrap_model(model)
                    if isinstance(unwrapped, backbone_cls):
                        unwrapped.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(unwrapped, (condition_encoder, two_condition_encoder)):
                        unwrapped.save_pretrained(os.path.join(output_dir, "condition_encoder"))
                    else:
                        raise TypeError(f"Unexpected model type in save hook: {type(unwrapped)}")
                    weights.pop()

        def load_model_hook(models, input_dir):
            for model in list(models):
                if isinstance(model, backbone_cls):
                    load_model = backbone_cls.from_pretrained(os.path.join(input_dir, "unet"))
                elif isinstance(model, (condition_encoder, two_condition_encoder)):
                    load_model = type(model).from_pretrained(os.path.join(input_dir, "condition_encoder"))
                else:
                    raise TypeError(f"Unexpected model type in load hook: {type(model)}")
                model.load_state_dict(load_model.state_dict())
                del load_model
                models.remove(model)

        if not args.no_save:
            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(Encoder.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ------------------------------------------------------------------ data
    interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode {args.image_interpolation_mode}.")

    with accelerator.main_process_first():
        presaved_path = os.path.join(args.presaved_path, "resolution_" + str(args.resolution))
        train_dataset = get_dataset(args.test_mode, resolution=args.resolution, interpolation=interpolation, presaved_path=presaved_path)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=make_collate_fn(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"training #images: {len(train_dataset)}\n")

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    unet, Encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, Encoder, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    args.validation_prompts, validation_key = build_validation_prompts(args, train_dataset, weight_dtype, accelerator.device)

    Encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    # ------------------------------------------------------------------ train
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    backbone_name = "UNet" if args.model == "unet" else "Transformer"
    unet_params = sum(p.numel() for p in unwrap_model(unet).parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in unwrap_model(Encoder).parameters() if p.requires_grad)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Num trainable {backbone_name} params = {format_param_count(unet_params)}")
    logger.info(f"  Num trainable Condition params = {format_param_count(encoder_params)}")
    logger.info(f"  Num VAE params = {format_param_count(sum(p.numel() for p in unwrap_model(vae).parameters()))}")
    print(f"Trainable {backbone_name} parameters: {format_param_count(unet_params)}")
    print(f"Trainable Condition parameters: {format_param_count(encoder_params)}")

    global_step = 0
    first_epoch = 0
    best_val_accuracy = 0.0
    patience_counter = 0
    should_stop = False

    def early_stopping_triggered(patience):
        """Main process only: True once best accuracy passed the threshold and patience ran out."""
        if not args.early_stopping or patience_counter < patience:
            return False
        if args.early_stopping_threshold is not None and best_val_accuracy < args.early_stopping_threshold:
            return False
        with open(os.path.join(save_dir, "early_stopping.txt"), "w") as f:
            f.write(f"early stopping triggered at step {global_step}: best_val_accuracy={best_val_accuracy:.4f}, "
                    f"patience={patience}, threshold={args.early_stopping_threshold}\n")
        logger.info("Early stopping triggered. Stopping training.")
        return True

    def init_trackers(resume=False):
        if not accelerator.is_main_process:
            return
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        if args.tracker_run_name is None:
            args.tracker_run_name = args.test_mode + "_" + param_string
        wandb_kwargs = {"name": args.tracker_run_name}
        if resume:
            wandb_kwargs["resume"] = "allow"
        accelerator.init_trackers(args.tracker_project_name, tracker_config, init_kwargs={"wandb": wandb_kwargs})

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(save_dir)] if os.path.exists(save_dir) else []
            dirs = sorted((d for d in dirs if d.startswith("checkpoint")), key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
            init_trackers()
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(save_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            init_trackers(resume=True)
            if os.path.exists(os.path.join(save_dir, "best_val_accuracy.txt")):
                with open(os.path.join(save_dir, "best_val_accuracy.txt"), "r") as f:
                    best_val_accuracy = float(f.readline().strip())
    else:
        initial_global_step = 0
        init_trackers()

    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Save the pipeline skeleton (model_index.json, scheduler, vae) once so validation can load it.
    if accelerator.is_main_process and not os.path.exists(f"{save_dir}/model_index.json"):
        if args.model == "dit":
            pipeline = MyCustomPipeline_dit(vae=vae, transformer=accelerator.unwrap_model(unet), condition_encoder=accelerator.unwrap_model(Encoder), scheduler=noise_scheduler)
        else:
            pipeline = MyCustomPipeline(vae=vae, unet=accelerator.unwrap_model(unet), condition_encoder=accelerator.unwrap_model(Encoder), scheduler=noise_scheduler)
        pipeline.save_pretrained(save_dir)
        del pipeline
        torch.cuda.empty_cache()

    def run_validation(epoch, patience):
        """Main process only: generate, score, keep best_model. Returns True when early stopping triggers."""
        nonlocal best_val_accuracy, patience_counter
        images = log_validation(
            vae=vae, Encoder=Encoder, unet=unet, args=args, accelerator=accelerator, epoch=global_step,
            save_dir=save_dir, validation_key=validation_key, no_image=args.no_image,
        )

        if args.image_save and not args.no_save:
            save_dir2 = os.path.join(save_dir, "validation_images", f"checkpoint-{global_step}")
            os.makedirs(save_dir2, exist_ok=True)
            for idx, image in enumerate(images):
                label = validation_key[idx] if idx < len(validation_key) else "unknown"
                image.save(os.path.join(save_dir2, f"validation_{idx}_label_{label}.png"))

        if not args.test_accuracy:
            return False

        results = test_accuracy(images=images, validation_key=validation_key, args=args, train_dataset=train_dataset,
                                accelerator=accelerator, Tester=Tester, Tester2=Tester2)
        log_accuracy_results(results, args, accelerator, global_step)

        if results["confidence"] > 0.95:
            with open(os.path.join(save_dir, "confidence.txt"), "a") as f:
                f.write(f"Step {global_step}: {results['confidence'].item():.2f}%\n")

        best_save_dir = os.path.join(save_dir, "best_model")
        current_accuracy = results[best_model_metric(args.test_mode)]
        if current_accuracy > best_val_accuracy and current_accuracy > 0.0:
            best_val_accuracy = current_accuracy
            patience_counter = 0
            accelerator.save_state(best_save_dir)  # creates best_model/ on first save
            logger.info(f"Saved state to {best_save_dir}")
            with open(os.path.join(save_dir, "best_val_accuracy.txt"), "w") as f:
                f.write(f"{best_val_accuracy:.4f}\n")
            with open(os.path.join(best_save_dir, "best_val_accuracy_steps.txt"), "a") as f:
                f.write(f"Step {global_step} Epoch {epoch}\n")
        else:
            patience_counter += 1

        logger.info(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
        if args.early_stopping:
            logger.info(f"Patience Counter: {patience_counter}/{patience}")
            return early_stopping_triggered(patience)
        return False

    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_train_loss = 0.0
        epoch_steps = 0
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, Encoder):
                if "pixel_values" in batch:
                    new_latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    new_latents = new_latents * vae.config.scaling_factor

                    # Cache the latents so later epochs skip the VAE (every rank writes its own files).
                    for latent, latent_path in zip(new_latents, batch["latent_path"]):
                        os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                        torch.save(latent.detach().cpu(), latent_path)

                    if "latents_values" in batch:
                        # Mixed batch: put cached and freshly encoded latents back in input_ids order.
                        latent_mask = batch["latent_mask"].to(new_latents.device)
                        latents = new_latents.new_empty((latent_mask.shape[0], *new_latents.shape[1:]))
                        latents[latent_mask] = batch["latents_values"].to(new_latents.device, new_latents.dtype)
                        latents[~latent_mask] = new_latents
                    else:
                        latents = new_latents
                else:
                    latents = batch["latents_values"].to(weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                if args.model == "dit":
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)
                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                else:
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # condition tokens: (B, 1, C) or (B, 2, C)
                encoder_hidden_states = Encoder(batch["input_ids"].to(weight_dtype))
                if encoder_hidden_states.dim() == 4:
                    encoder_hidden_states = encoder_hidden_states.squeeze(1)
                if encoder_hidden_states.dim() == 2:
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                assert encoder_hidden_states.dim() == 3, f"Encoder hidden states should be (batch, seq_len, dim), got {encoder_hidden_states.shape}"

                if args.model == "dit":
                    target = latents if args.precondition_outputs else noise - latents
                    model_pred = unet(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_projections=None,
                        return_dict=False,
                    )[0]
                    if args.precondition_outputs:
                        model_pred = model_pred * (-sigmas) + noisy_latents
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                    loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
                    loss = loss.mean()
                else:
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(unet.parameters()) + list(Encoder.parameters()), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                epoch_train_loss += train_loss
                epoch_steps += 1
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 and not args.no_save:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted((d for d in os.listdir(save_dir) if d.startswith("checkpoint")), key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                removing_checkpoints = checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]
                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    shutil.rmtree(os.path.join(save_dir, removing_checkpoint))

                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Step-based validation
                step_true = args.validation_steps is not None and global_step % args.validation_steps == 0 and global_step > 0
                if step_true and accelerator.is_main_process:
                    should_stop = run_validation(epoch, args.early_stopping_patience_steps)
                # The stop decision is made on the main process; share it so every rank leaves the loop together.
                if step_true and args.early_stopping:
                    should_stop = broadcast_object_list([should_stop])[0]

            if should_stop:
                break

            progress_bar.set_postfix(step_loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            accelerator.log(
                {"epoch": epoch, "avg_epoch_loss": epoch_train_loss / max(epoch_steps, 1), "learning_rate": lr_scheduler.get_last_lr()[0]},
                step=global_step,
            )

        # Epoch-based validation
        epoch_true = args.validation_epochs is not None and epoch % args.validation_epochs == 0 and epoch > 0
        if epoch_true and accelerator.is_main_process:
            should_stop = run_validation(epoch, args.early_stopping_patience_epochs)
        if args.early_stopping and args.validation_epochs is not None:
            should_stop = broadcast_object_list([should_stop])[0]
        if should_stop:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not args.no_save:
        pipeline = build_pipeline(args, save_dir, vae, unwrap_model(unet), accelerator.unwrap_model(Encoder))
        pipeline.save_pretrained(save_dir)

    accelerator.end_training()

    if accelerator.is_main_process:
        logger.info("Training completed.")
        with open(os.path.join(save_dir, "completed.txt"), "w") as f:
            f.write(f"Training completed successfully with training steps {global_step}.\n")


if __name__ == "__main__":
    main()
