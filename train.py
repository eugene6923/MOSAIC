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

import copy
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import sys
import signal

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, broadcast_object_list, set_seed

from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
from utils import test_accuracy, log_accuracy_results, sample_t_given_k_batch, sample_timesteps_oversampling, sample_timesteps_oversampling2
sys.path.append('./diffusers/src')
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler, UNet2DConditionModel, SD3Transformer2DModel, MyCustomPipeline, MyCustomPipeline_dit, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import random
from data import get_dataset
from train_args import parse_args as parse_train_args
from train_data import make_collate_fn
from validation_prompts import build_validation_prompts
import wandb
import json

check_min_version("0.34.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))

def bring_text_embedding(input_ids, args):
    if "attribute" in args.text_mode:
        with open('clip_embedding_attributes.json', 'r') as f:
            embedding_dict = json.load(f)
    elif "position" in args.text_mode:
        with open('clip_embedding_positions.json', 'r') as f:
            embedding_dict = json.load(f)
    elif "object" in args.text_mode:
        with open('clip_embedding_objects.json', 'r') as f:
            embedding_dict = json.load(f)
    else:
        with open('clip_embedding_counting.json', 'r') as f:
            embedding_dict = json.load(f)

    new_embedding = []
    for i in input_ids:
        new_embedding.append(embedding_dict[str(i.item())])
    return new_embedding

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

def log_validation(vae, Encoder, unet, args, accelerator, weight_dtype, epoch,save_dir, validation_key = None, no_image = False):
    logger.info("Running validation... ")

    use_dit = args.model == "dit"

    if not use_dit:
        scheduler_map = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "DDPMScheduler": DDPMScheduler,
        }
        validation_scheduler_cls = scheduler_map.get(args.validation_scheduler, EulerDiscreteScheduler)
        validation_scheduler = validation_scheduler_cls.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
    else:
        validation_scheduler = None

    unet.eval()
    if Encoder is not None:
        if isinstance(Encoder, tuple):
            Encoder[0].eval()
            Encoder[1].eval()

            if use_dit:
                pipeline = MyCustomPipeline_dit.from_pretrained(
                    save_dir,
                    vae=accelerator.unwrap_model(vae),
                    transformer=accelerator.unwrap_model(unet),
                    condition_encoder=accelerator.unwrap_model(Encoder[0]),
                    condition_encoder2=accelerator.unwrap_model(Encoder[1]),
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                    id_condition=args.id_condition_encoder,
                )
            else:
                pipeline = MyCustomPipeline.from_pretrained(
                    save_dir,
                    vae=accelerator.unwrap_model(vae),
                    unet=accelerator.unwrap_model(unet),
                    condition_encoder=accelerator.unwrap_model(Encoder[0]),
                    condition_encoder2=accelerator.unwrap_model(Encoder[1]),
                    scheduler=validation_scheduler,
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                    id_condition=args.id_condition_encoder,
                )

        else:
            Encoder.eval()

            if use_dit:
                pipeline = MyCustomPipeline_dit.from_pretrained(
                    save_dir,
                    vae=accelerator.unwrap_model(vae),
                    transformer=accelerator.unwrap_model(unet),
                    condition_encoder=accelerator.unwrap_model(Encoder),
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                    id_condition=args.id_condition_encoder,
                )
            else:
                pipeline = MyCustomPipeline.from_pretrained(
                    save_dir,
                    vae=accelerator.unwrap_model(vae),
                    unet=accelerator.unwrap_model(unet),
                    condition_encoder=accelerator.unwrap_model(Encoder),
                    scheduler=validation_scheduler,
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                    id_condition=args.id_condition_encoder,
                )
    else:
        if use_dit:
            pipeline = MyCustomPipeline_dit.from_pretrained(
                save_dir,
                vae=accelerator.unwrap_model(vae),
                transformer=accelerator.unwrap_model(unet),
                condition_encoder=None,
                image_encoder=None,
                feature_extractor=None,
                safety_checker=None,
            )
        else:
            pipeline = MyCustomPipeline.from_pretrained(
                save_dir,
                vae=accelerator.unwrap_model(vae),
                unet=accelerator.unwrap_model(unet),
                scheduler=validation_scheduler,
                condition_encoder=None,
                image_encoder=None,
                feature_extractor=None,
                safety_checker=None,
            )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        if not isinstance(Encoder, tuple):
            batched_prompts = torch.stack(args.validation_prompts)
            if args.validation_batch_size is not None:
                all_images = []
                for i in range(0, len(batched_prompts), args.validation_batch_size):
                    batch_prompts = batched_prompts[i:i+args.validation_batch_size]
                    images = pipeline(batch_prompts, num_inference_steps=20, generator=generator).images
                    all_images.extend(images)
                images = all_images
            else: images = pipeline(batched_prompts, num_inference_steps=20, generator=generator).images
        else:
            batched_prompts = torch.stack([i[0] for i in args.validation_prompts])
            batched_prompts2 = torch.stack([i[1] for i in args.validation_prompts])
            batched_prompts = (batched_prompts, batched_prompts2)
            if args.validation_batch_size is not None:
                all_images = []
                for i in range(0, len(batched_prompts[0]), args.validation_batch_size):
                    batch_prompts = (batched_prompts[0][i:i+args.validation_batch_size], batched_prompts[1][i:i+args.validation_batch_size])
                    images = pipeline(batch_prompts, num_inference_steps=20, generator=generator).images
                    all_images.extend(images)
                images = all_images
            else: images = pipeline(batched_prompts, num_inference_steps=20, generator=generator).images

    if not no_image:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            elif tracker.name == "wandb":
                logged_keys = set()
                images_to_log = []
                captions_to_log = []

                for i, image in enumerate(images):
                    if validation_key and i < len(validation_key):
                        key = validation_key[i]
                        if isinstance(key, str):
                            caption = f"{i}: {key}"
                        elif isinstance(key, int):
                            if key == 0:
                                caption = f"{i}: no_objects"
                            else:
                                caption = f"{i}: {key}_objects"
                        else:
                            caption = f"{i}: unknown"

                        # Only log if we haven't seen this key before
                        if key not in logged_keys:
                            logged_keys.add(key)
                            images_to_log.append(image)
                            captions_to_log.append(caption)

                    else:
                        # Fallback: try to parse from prompt tensor
                        prompt = args.validation_prompts[i]
                        if "onehot" in args.test_mode:
                            if prompt.sum() > 0:
                                count = torch.argmax(prompt.squeeze()) + 1
                                caption = f"{i}: {count}_objects"
                            else:
                                caption = f"{i}: no_objects"
                        elif "composition" in args.test_mode:
                            # Handle composition case - prompt is a concatenated tensor
                            caption = f"{i}: composition_unknown"
                        elif args.id_condition:
                            count = int(prompt.squeeze().item())
                            caption = f"{i}: {count}_objects" if count > 0 else f"{i}: no_objects"
                        else:
                            if prompt.sum() > 0:
                                count = int(round(prompt.squeeze()[0].item() * 9 + 1))
                                caption = f"{i}: {count}_objects" if count > 0 else f"{i}: no_objects"
                            else:
                                caption = f"{i}: no_objects"


                    # Extract key from caption for deduplication
                    key_from_caption = caption.split(": ")[1] if ": " in caption else caption
                    if key_from_caption not in logged_keys:
                        logged_keys.add(key_from_caption)
                        images_to_log.append(image)
                        captions_to_log.append(caption)

                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=caption)
                            for image, caption in zip(images_to_log, captions_to_log) # only one image per unique key
                        ]
                    }
                )
            else:
                logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()
    # Set models back to training mode
    unet.train()
    if Encoder is not None:
        if isinstance(Encoder, tuple):
            Encoder[0].train()
            Encoder[1].train()
        else:
            Encoder.train()

    return images


def main():
    args = parse_train_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    if args.report_to == "wandb":

        def handle_termination(signum, frame):
            print("Received SIGTERM. Finishing W&B run...")
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_termination)
        signal.signal(signal.SIGINT, handle_termination)  # Optional: for Ctrl+C


    if args.id_condition and args.id_condition_encoder:
        raise ValueError("Cannot use both id_condition and id_condition_encoder at the same time. Please choose one.")
    if (args.id_condition or args.id_condition_encoder) and args.off_encoder:
        raise ValueError("Cannot use both id_condition (or id_condition_encoder) and off_encoder at the same time. Please choose one.")
    if "onehot" in args.test_mode and args.id_condition:
        raise ValueError("Cannot use onehot test mode with id_condition. Please choose a different test mode.")

    if args.validation_epochs is not None:
        args.validation_steps = None

    if args.validation_steps is not None and args.validation_epochs is not None:
        raise ValueError("Cannot use both validation_steps and validation_epochs. Please choose one.")

    if args.classification_loss and args.classification_loss_ce:
        raise ValueError("Cannot use both classification_loss and classification_loss_ce at the same time. Please choose one.")


    param_string = f"model{args.model}_resolution{args.resolution}_lr{args.learning_rate}{'_no_encoder' if args.off_encoder else ''}{args.encoder_learning_rate if args.encoder_learning_rate is not None and not args.off_encoder else ''}{'switch'+str(args.switch_training) if args.switch_training is not None else ''}{f'_warmup{args.lr_warmup_steps}' if args.lr_scheduler != 'constant' else ''}_bs{args.train_batch_size}_ga{args.gradient_accumulation_steps}_sch{args.lr_scheduler}_valshc{args.validation_scheduler}_wd{args.adam_weight_decay}{'_memory_efficient' if args.enable_xformers_memory_efficient_attention else ''}{'_id_condition' if args.id_condition else ''}{'_train_id_condition' if args.id_condition_encoder else ''}{'_class_concat' if args.id_condition and args.class_embeddings_concat else ''}{'_num_tokens' + str(args.num_tokens) if args.num_tokens > 1 else ''}{'_text_enable' if args.text_enable else ''}{'_dream_training' if args.dream_training else ''}{'_use_ema' if args.use_ema else ''}{'_offload_ema' if args.offload_ema else ''}{'_foreach_ema' if args.foreach_ema else ''}{'_mixed_precision_' + args.mixed_precision if args.mixed_precision is not None else ''}{'_drop_out' + str(args.drop_rate)}{'_condition_drop_out' + str(args.condition_encoder_drop_rate)}{f'_classification_loss{args.classification_loss_weight}' if args.classification_loss else ''}{f'_classification_loss_ce{args.classification_loss_weight}' if args.classification_loss_ce else ''}{'_noise_oversampling' if args.noise_oversampling else ''}{'_noise_oversampling2' if args.noise_oversampling2 else ''}{'_model_only' if args.pretrained_encoder_path is not None or args.text_enable else ''}"

    save_root = args.save_dir if args.save_dir is not None else ("dit_weights" if args.model == "dit" else "unet_weights")
    save_dir = os.path.join(save_root, 'seed_'+str(args.seed), args.test_mode, param_string)

    logging_dir = os.path.join(save_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=save_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True)
    ]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("Training/evaluation parameters %s", args)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        print(args)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(save_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    if args.model == "dit":
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="scheduler"
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        noise_scheduler_copy = None

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    vae.requires_grad_(False)

    if accelerator.is_main_process and args.test_accuracy:
        from model import Classifier, PretrainedClassifier

        if args.resolution == 128:

            if "attribute" in args.test_mode:
                Tester = PretrainedClassifier(num_classes=100)
                Tester.load_state_dict(torch.load("./classifier_weights/best_classifier_attribute_pretrained.pth", map_location='cpu', weights_only=False))
                Tester2 = PretrainedClassifier(num_classes=6)
                Tester2.load_state_dict(torch.load("./classifier_weights/best_classifier_attribute_shape_pretrained.pth", map_location='cpu', weights_only=False))
                Tester2.to(accelerator.device)
                Tester2.requires_grad_(False)

            elif "position" in args.test_mode:
                Tester = PretrainedClassifier(num_classes=10)
                Tester.load_state_dict(torch.load("./classifier_weights/best_classifier_position_pretrained.pth", map_location=accelerator.device))
                if "composition" in args.test_mode:
                    Tester2 = PretrainedClassifier(num_classes=10)
                    Tester2.load_state_dict(torch.load("./classifier_weights/best_classifier_position_color_pretrained.pth", map_location=accelerator.device))
                    Tester2.to(accelerator.device)
                    Tester2.requires_grad_(False)

            elif "count" in args.test_mode:
                Tester = Classifier(num_classes=20)
                Tester.load_state_dict(torch.load("./classifier_weights/best_classifier_count.pth", map_location='cpu'))
                if "composition" in args.test_mode:
                    Tester2 = Classifier(num_classes=10)
                    Tester2.load_state_dict(torch.load("./classifier_weights/best_classifier_count_color.pth", map_location=accelerator.device))
                    Tester2.to(accelerator.device)
                    Tester2.requires_grad_(False)

        Tester.to(accelerator.device)
        Tester.requires_grad_(False)

    else:
        Tester = None
        Tester2 = None

    if not args.text_enable:
        try:
            if "composition" not in args.test_mode and "attribute" not in args.test_mode:
                from model import condition_encoder

                if args.model == "dit":
                    condition_encoder_config = {
                        "input_dim": 10,
                        "output_dim": 512,
                        "hidden_dims": [64, 128, 256],
                        "dropout_rate": args.condition_encoder_drop_rate,
                        "num_tokens": args.num_tokens,
                    }
                else:
                    condition_encoder_config = {
                        "input_dim": 10,
                        "output_dim": 64,
                        "dropout_rate": args.condition_encoder_drop_rate,
                        "num_tokens": args.num_tokens,
                    }

                Encoder = condition_encoder(
                    input_dim=condition_encoder_config["input_dim"],
                    output_dim=condition_encoder_config["output_dim"],
                    hidden_dims=condition_encoder_config.get("hidden_dims", []),  # Optional hidden layers
                    num_tokens=args.num_tokens,
                    dropout_rate=condition_encoder_config["dropout_rate"],
                )
                Encoder.train()

                if args.classification_loss or args.classification_loss_ce:
                    from model import ClassificationHead
                    classification_head = ClassificationHead(in_features=condition_encoder_config["output_dim"], num_classes=10)
                    classification_head.train()

            else:
                from model import two_condition_encoder
                if args.model == "dit":
                    condition_encoder_config = {
                        "input_dim": 1 if "onecolor" in args.test_mode and not "onehot" in args.test_mode else 10,
                        "output_dim": 512,
                        "hidden_dims": [64, 128, 256],
                        "dropout_rate": args.condition_encoder_drop_rate,
                        "num_tokens": args.num_tokens,
                    }
                else:
                    condition_encoder_config = {
                        "input_dim": 10,
                        "output_dim": 64,
                        "dropout_rate": args.condition_encoder_drop_rate,
                    }

                Encoder = two_condition_encoder(
                    input_dim=condition_encoder_config["input_dim"],
                    output_dim=condition_encoder_config["output_dim"],
                    hidden_dims=condition_encoder_config.get("hidden_dims", []),  # Optional hidden layers
                    dropout_rate=condition_encoder_config["dropout_rate"],
                )

                Encoder.train()
        except:
            print("Error loading condition encoder. We will not use a condition encoder.")
            args.test_accuracy = False


    if args.pretrained_encoder_path is not None:
        if args.text_enable:
            raise ValueError("Cannot use pretrained_encoder_path with text_enable.")

        if args.id_condition or args.off_encoder:
            raise ValueError("Cannot use pretrained_encoder_path with id_condition or off_encoder.")

        logger.info(f"Loading pretrained encoder from {args.pretrained_encoder_path}")
        if isinstance(Encoder, tuple):
            # Load both encoders if it's a tuple
            encoder1_path = os.path.join(args.pretrained_encoder_path, "encoder1")
            encoder2_path = os.path.join(args.pretrained_encoder_path, "encoder2")
            if os.path.exists(encoder1_path) and os.path.exists(encoder2_path):
                Encoder[0].load_state_dict(torch.load(os.path.join(encoder1_path, "pytorch_model.bin")))
                Encoder[1].load_state_dict(torch.load(os.path.join(encoder2_path, "pytorch_model.bin")))
                logger.info("Successfully loaded pretrained encoders (tuple)")
            else:
                raise ValueError(f"Could not find encoder checkpoints at {encoder1_path} and {encoder2_path}")
        else:
            # Load single encoder
            if not args.pretrained_encoder_path.endswith("condition_encoder"):
                args.pretrained_encoder_path = os.path.join(args.pretrained_encoder_path, "condition_encoder")
            encoder_checkpoint_path = os.path.join(args.pretrained_encoder_path, "pytorch_model.bin")
            if os.path.exists(encoder_checkpoint_path):
                Encoder.load_state_dict(torch.load(encoder_checkpoint_path))
                logger.info("Successfully loaded pretrained encoder")
            else:
                raise ValueError(f"Could not find encoder checkpoint at {encoder_checkpoint_path}")

        # Freeze the pretrained encoder
        if isinstance(Encoder, tuple):
            Encoder[0].requires_grad_(False)
            Encoder[1].requires_grad_(False)
            logger.info("Froze pretrained encoders (tuple)")
        else:
            Encoder.requires_grad_(False)
            logger.info("Froze pretrained encoder")

    if args.text_enable:
        #TODO: CLIP embeddings as condition - need to load CLIP model and tokenizer, encode text, and use as condition
        condition_encoder_config = {"output_dim": 768}


    if args.model == "dit":
        transformer_config = {
            "sample_size": args.resolution // 8,
            "patch_size": 2,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 10,
            "attention_head_dim": 32,
            "num_attention_heads": 16,
            "joint_attention_dim": condition_encoder_config["output_dim"],
            "caption_projection_dim": condition_encoder_config["output_dim"],
            "pooled_projection_dim": condition_encoder_config["output_dim"],
        }
        unet = SD3Transformer2DModel(**transformer_config)
        model_cls_for_ema = SD3Transformer2DModel
    else:
        unet_config = {
            "sample_size": args.resolution // 8,
            "in_channels": 4,
            "out_channels": 4,
            "block_out_channels": (80, 160, 320, 320),
            "layers_per_block": 4,
            "down_block_types": ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",  "CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": condition_encoder_config["output_dim"],
            "use_linear_projection": False,
            "norm_num_groups": 16,
        }
        unet = UNet2DConditionModel(**unet_config)
        model_cls_for_ema = UNet2DConditionModel

    # Create EMA for the unet.
    if args.use_ema:
        if args.model == "dit":
            ema_backbone = SD3Transformer2DModel(**transformer_config)
        else:
            ema_backbone = UNet2DConditionModel(**unet_config)
        ema_unet = EMAModel(
            ema_backbone.parameters(),
            model_cls=model_cls_for_ema,
            model_config=ema_backbone.config,
            foreach=args.foreach_ema,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):

            if accelerator.is_main_process:

                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                if args.classification_loss or args.classification_loss_ce:
                    # UNet + Encoder + ClassificationHead
                    if len(models) == 3:
                        unwrapped_unet = accelerator.unwrap_model(models[0])
                        unwrapped_encoder = accelerator.unwrap_model(models[1])
                        unwrapped_classification_head = accelerator.unwrap_model(models[2])

                        if not args.no_save:
                            unwrapped_unet.save_pretrained(os.path.join(output_dir, "unet"))
                            unwrapped_encoder.save_pretrained(os.path.join(output_dir, "condition_encoder"))
                            torch.save(unwrapped_classification_head.state_dict(),
                                        os.path.join(output_dir, "classification_head.pth"))
                            for _ in range(len(models)):
                                weights.pop()
                else:
                    if len(models) == 3:  # UNet + 2 Encoders (tuple case)
                        unwrapped_unet = accelerator.unwrap_model(models[0])
                        unwrapped_encoder1 = accelerator.unwrap_model(models[1])
                        unwrapped_encoder2 = accelerator.unwrap_model(models[2])

                        if not args.no_save:
                            unwrapped_unet.save_pretrained(os.path.join(output_dir, "unet"))
                            unwrapped_encoder1.save_pretrained(os.path.join(output_dir, "condition_encoder"))
                            unwrapped_encoder2.save_pretrained(os.path.join(output_dir, "condition_encoder2"))
                            # Clear weights list
                            for _ in range(len(models)):
                                weights.pop()
                    elif len(models) == 2:
                        unwrapped_unet = accelerator.unwrap_model(models[0])
                        unwrapped_encoder = accelerator.unwrap_model(models[1])
                        if not args.no_save:
                            unwrapped_unet.save_pretrained(os.path.join(output_dir, "unet"))
                            unwrapped_encoder.save_pretrained(os.path.join(output_dir, "condition_encoder"))
                            weights.pop()
                            weights.pop()
                    elif len(models) == 1:
                        unwrapped_model = accelerator.unwrap_model(models[0])
                        print(f"Model type: {type(unwrapped_model)}")
                        if not args.no_save:
                            unwrapped_model.save_pretrained(os.path.join(output_dir, "unet"))
                            weights.pop()
                    else:
                        for i, model in enumerate(models):
                            unwrapped_model = accelerator.unwrap_model(model)
                            print(f"Model {i} type: {type(unwrapped_model)}")
                            if not args.no_save:
                                unwrapped_model.save_pretrained(os.path.join(output_dir, "unet"))
                                weights.pop()


        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), model_cls_for_ema, foreach=args.foreach_ema
                )
                ema_unet.load_state_dict(load_model.state_dict())
                if args.offload_ema:
                    ema_unet.pin_memory()
                else:
                    ema_unet.to(accelerator.device)
                del load_model

            if "composition" not in args.test_mode:
                from model import condition_encoder, two_condition_encoder

                models_copy = list(models)
                for i, model in  enumerate(models_copy):
                    if (args.model == "unet" and isinstance(model, UNet2DConditionModel)) or (
                        args.model == "dit" and isinstance(model, SD3Transformer2DModel)
                    ):
                        if args.model == "dit":
                            load_model = SD3Transformer2DModel.from_pretrained(
                                os.path.join(input_dir, "unet"), use_safetensors=True
                            )
                        else:
                            load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, "unet"))
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                        models.remove(model)
                    elif isinstance(model, condition_encoder):
                        if os.path.exists(os.path.join(input_dir, "condition_encoder2")):
                            # Load based on model order (after UNet)
                            encoder_index = i - 1  # Subtract 1 for UNet
                            if encoder_index == 0:
                                load_model = condition_encoder.from_pretrained(os.path.join(input_dir, "condition_encoder"))
                            else:
                                load_model = condition_encoder.from_pretrained(os.path.join(input_dir, "condition_encoder2"))
                        else:
                            load_model = condition_encoder.from_pretrained(os.path.join(input_dir, "condition_encoder"))
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                        models.remove(model)
                    elif isinstance(model,two_condition_encoder):
                        load_model = two_condition_encoder.from_pretrained(os.path.join(input_dir, "condition_encoder"))
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                        models.remove(model)

                    elif args.classification_loss and hasattr(model, 'in_features'):  # Check if it's classification head
                        # Load classification head
                        if os.path.exists(os.path.join(input_dir, "classification_head.pth")):
                            model.load_state_dict(torch.load(os.path.join(input_dir, "classification_head.pth")))
                        models.remove(model)
            else:
                from model import two_condition_encoder
                models_copy = list(models)
                for model in models_copy:
                    if (args.model == "unet" and isinstance(model, UNet2DConditionModel)) or (
                        args.model == "dit" and isinstance(model, SD3Transformer2DModel)
                    ):
                        if args.model == "dit":
                            load_model = SD3Transformer2DModel.from_pretrained(
                                os.path.join(input_dir, "unet"), use_safetensors=True
                            )
                        else:
                            load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, "unet"))
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                        models.remove(model)
                    elif isinstance(model, two_condition_encoder):
                        load_model = two_condition_encoder.from_pretrained(os.path.join(input_dir, "condition_encoder"))
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                        models.remove(model)

        if not args.no_save:
            # Register the save and load hooks
            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)


    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.text_enable or args.id_condition or args.off_encoder or args.pretrained_encoder_path is not None:
        trainable_params = list(unet.parameters())
    elif isinstance(Encoder, tuple):
        trainable_params = list(unet.parameters()) + list(Encoder[0].parameters()) + list(Encoder[1].parameters())
    else:
        trainable_params = list(unet.parameters()) + list(Encoder.parameters())

    if args.classification_loss or args.classification_loss_ce:
        trainable_params.extend(list(classification_head.parameters()))

    if args.encoder_learning_rate is None or not args.text_enable or args.id_condition or args.off_encoder or args.pretrained_encoder_path is not None:

        optimizer = optimizer_cls(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif not args.id_condition and not args.off_encoder and args.encoder_learning_rate is not None and args.pretrained_encoder_path is None:
        # If using id_condition, we only train the unet
        optimizer = optimizer_cls(
            [
                {"params": unet.parameters(), "lr": args.learning_rate},
                {"params": Encoder.parameters(), "lr": args.encoder_learning_rate},
            ],
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
    )
    else:
        raise ValueError(
            "If you are not using text_encoder or id_condition, you must specify --encoder_learning_rate."
            " Otherwise, please set --text_enable or --id_condition to True."
        )

    interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)

    # Raise an error if the interpolation method is invalid
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode {args.image_interpolation_mode}.")


    with accelerator.main_process_first():
        presaved_path = os.path.join(args.presaved_path, "resolution_" + str(args.resolution))
        train_dataset = get_dataset(args.test_mode, resolution = args.resolution, interpolation = interpolation, center_crop = args.center_crop, random_flip = args.random_flip, use_text_encoder = args.use_text_encoder, presaved_path = presaved_path, id_condition = args.id_condition)

    collate_fn = make_collate_fn(args)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
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
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    models_and_optim = [unet, optimizer, train_dataloader, lr_scheduler]
    if not args.text_enable and not args.id_condition and not args.off_encoder and args.pretrained_encoder_path is None:
        # We *are* training Encoder → include it
        if isinstance(Encoder, tuple):
            # Add both encoders separately
            models_and_optim.insert(1, Encoder[0])
            models_and_optim.insert(2, Encoder[1])
        else:
            # Add single encoder
            models_and_optim.insert(1, Encoder)

        if args.classification_loss or args.classification_loss_ce:
            models_and_optim.insert(-3, classification_head)  # Insert before optimizer, dataloader, scheduler

    prepared = accelerator.prepare(*models_and_optim)

    # Prepare everything with our `accelerator`.
    if args.text_enable or args.id_condition or args.off_encoder  or args.pretrained_encoder_path is not None:
        unet, optimizer, train_dataloader, lr_scheduler = prepared
    else:
        if isinstance(Encoder, tuple):
            if not args.classification_loss and not args.classification_loss_ce:
                unet, Encoder0, Encoder1, optimizer, train_dataloader, lr_scheduler = prepared
                Encoder = (Encoder0, Encoder1)  # Reconstruct tuple
            else:
                raise NotImplementedError("Not implemented for classification loss with two encoders.")
        else:
            if args.classification_loss or args.classification_loss_ce:
                unet, Encoder, classification_head, optimizer, train_dataloader, lr_scheduler = prepared
            else:
                unet, Encoder, optimizer, train_dataloader, lr_scheduler = prepared

    if args.use_ema:
        if args.offload_ema:
            ema_unet.pin_memory()
        else:
            ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    args.validation_prompts, validation_key = build_validation_prompts(
        args, train_dataset, weight_dtype, accelerator.device
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    if not args.text_enable and not args.id_condition and not args.off_encoder:
        if isinstance(Encoder, tuple):
            Encoder[0].to(accelerator.device, dtype=weight_dtype)
            Encoder[1].to(accelerator.device, dtype=weight_dtype)
        else:
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
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    unet_params = sum(p.numel() for p in unwrap_model(unet).parameters() if p.requires_grad)
    logger.info(f"  Num trainable {'UNet' if args.model == 'unet' else 'Transformer'} params = {format_param_count(unet_params)}")
    print(f"Trainable {'UNet' if args.model == 'unet' else 'Transformer'} parameters: {format_param_count(unet_params)}")
    if not args.id_condition:
        if isinstance(Encoder, tuple):
            encoder_params = sum(p.numel() for p in unwrap_model(Encoder[0]).parameters() if p.requires_grad) + sum(p.numel() for p in unwrap_model(Encoder[1]).parameters() if p.requires_grad)
        else:
            encoder_params = sum(p.numel() for p in unwrap_model(Encoder).parameters() if p.requires_grad)
        logger.info(f"  Num trainable Condition params = {format_param_count(encoder_params)}")
    print(f"Trainable Condition parameters: {format_param_count(encoder_params) if not args.id_condition else 0}")
    logger.info(f"  Num VAE params = {format_param_count(sum(p.numel() for p in unwrap_model(vae).parameters()))}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if "position" in args.test_mode:
        args.tracker_project_name = args.tracker_project_name + "_position"

    if "attribute" in args.test_mode:
        args.tracker_project_name = args.tracker_project_name + "_attribute"

    if "object" in args.test_mode:
        args.tracker_project_name = args.tracker_project_name + "_object"

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

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            if os.path.exists(save_dir):
                dirs = os.listdir(save_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
            else:
                path = None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
            if accelerator.is_main_process:
                tracker_config = dict(vars(args))
                tracker_config.pop("validation_prompts")
                if args.tracker_run_name is None:
                    args.tracker_run_name = args.test_mode + "_" + param_string
                run_name = args.tracker_run_name
                wandb_kwargs = {"name": run_name} if run_name else {}
                accelerator.init_trackers(
                    args.tracker_project_name,
                    tracker_config,
                    init_kwargs={"wandb": wandb_kwargs}
                )
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(save_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            if accelerator.is_main_process:
                tracker_config = dict(vars(args))
                tracker_config.pop("validation_prompts")
                if args.tracker_run_name is None:
                    args.tracker_run_name = args.test_mode + "_" + param_string
                run_name = args.tracker_run_name
                wandb_kwargs = {"name": run_name} if run_name else {}
                wandb_kwargs["resume"] = "allow"
                accelerator.init_trackers(
                    args.tracker_project_name,
                    tracker_config,
                    init_kwargs={"wandb": wandb_kwargs}
                )
            if os.path.exists(os.path.join(save_dir, "best_val_accuracy.txt")):
                with open(os.path.join(save_dir, "best_val_accuracy.txt"), "r") as f:
                    best_val_accuracy = float(f.readline().strip())

    else:
        initial_global_step = 0
        if accelerator.is_main_process:
            tracker_config = dict(vars(args))
            tracker_config.pop("validation_prompts")
            if args.tracker_run_name is None:
                args.tracker_run_name = args.test_mode + "_" + param_string
            run_name = args.tracker_run_name
            wandb_kwargs = {"name": run_name} if run_name else {}
            accelerator.init_trackers(
                args.tracker_project_name,
                tracker_config,
                init_kwargs={"wandb": wandb_kwargs}
            )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        if args.model != "dit":
            return None
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    if accelerator.is_main_process:

        if not os.path.exists(f'{save_dir}/model_index.json') and not args.id_condition and not args.off_encoder:
            if isinstance(Encoder, tuple):
                if args.model == "dit":
                    pipeline = MyCustomPipeline_dit(
                        vae=vae,
                        transformer=accelerator.unwrap_model(unet),
                        condition_encoder=accelerator.unwrap_model(Encoder[0]),
                        condition_encoder2=accelerator.unwrap_model(Encoder[1]),
                        scheduler=noise_scheduler,
                        image_encoder=None,
                        feature_extractor=None,
                        id_condition=args.id_condition_encoder,
                    )
                else:
                    pipeline = MyCustomPipeline(
                        vae=vae,
                        unet=accelerator.unwrap_model(unet),
                        condition_encoder=accelerator.unwrap_model(Encoder[0]),
                        condition_encoder2=accelerator.unwrap_model(Encoder[1]),
                        scheduler=noise_scheduler,
                        image_encoder=None,
                        feature_extractor=None,
                        safety_checker=None,
                        id_condition=args.id_condition_encoder,
                    )
            else:
                if args.model == "dit":
                    pipeline = MyCustomPipeline_dit(
                        vae=vae,
                        transformer=accelerator.unwrap_model(unet),
                        condition_encoder=accelerator.unwrap_model(Encoder),
                        scheduler=noise_scheduler,
                        image_encoder=None,
                        feature_extractor=None,
                        id_condition=args.id_condition_encoder,
                    )
                else:
                    pipeline = MyCustomPipeline(
                        vae=vae,
                        unet=accelerator.unwrap_model(unet),
                        condition_encoder=accelerator.unwrap_model(Encoder),
                        scheduler=noise_scheduler,
                        image_encoder=None,
                        feature_extractor=None,
                        safety_checker=None,
                        id_condition=args.id_condition_encoder,
                    )

            pipeline.save_pretrained(f"{save_dir}")
            del pipeline
            torch.cuda.empty_cache()

        elif not os.path.exists(f'{save_dir}/model_index.json') and (args.id_condition or args.off_encoder):
            if args.model == "dit":
                pipeline = MyCustomPipeline_dit(
                    vae=vae,
                    transformer=accelerator.unwrap_model(unet),
                    scheduler=noise_scheduler,
                    image_encoder=None,
                    feature_extractor=None,
                    id_condition=args.id_condition_encoder,
                )
            else:
                pipeline = MyCustomPipeline(
                    vae=vae,
                    unet=accelerator.unwrap_model(unet),
                    scheduler=noise_scheduler,
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                    id_condition=args.id_condition_encoder,
                )

            pipeline.save_pretrained(f"{save_dir}")
            del pipeline
            torch.cuda.empty_cache()

    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_train_loss = 0.0
        epoch_steps = 0
        train_loss = 0.0
        if args.classification_loss or args.classification_loss_ce:
            epoch_classification_loss = 0.0
            train_classification_loss = 0.0
        if args.switch_training is not None:
            if epoch % args.switch_training == 0 and args.switch_training is not None:
                # Freeze UNet
                for param in unet.parameters():
                    param.requires_grad = False
                for param in Encoder.parameters():
                    param.requires_grad = True
            else:
                # Unfreeze both
                for param in unet.parameters():
                    param.requires_grad = True
                for param in Encoder.parameters():
                    param.requires_grad = True


        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [unet]
            if not args.text_enable and not args.id_condition and not args.off_encoder:
                models_to_accumulate.append(Encoder)

            with accelerator.accumulate(*models_to_accumulate):

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


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
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
                elif args.timestep_sampling is None:
                    if not args.noise_oversampling and not args.noise_oversampling2:
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    else:
                        if args.noise_oversampling:
                            timesteps, _, _ = sample_timesteps_oversampling(
                                num_steps=noise_scheduler.config.num_train_timesteps,
                                bsz=bsz,
                                mu=0.7, sigma=0.08, lam=0.5,
                                importance_correction=True,
                                correction_power=0.5,
                                return_continuous=True,
                            )
                        else:
                            timesteps, _, _ = sample_timesteps_oversampling2(
                                num_steps=noise_scheduler.config.num_train_timesteps,
                                bsz=bsz,
                                mu=0.7, sigma=0.08, lam=0.5,
                                importance_correction=True,
                                correction_power=0.5,
                                return_continuous=True,
                            )

                        timesteps = torch.from_numpy(timesteps).to(latents.device)
                else:
                    if args.noise_oversampling:
                        raise NotImplementedError("Noise oversampling with timestep sampling not implemented.")
                    timesteps = sample_t_given_k_batch(
                        noise_scheduler.config.num_train_timesteps,
                        batch["input_ids"].cpu().numpy(),
                        k_max=10,
                        alpha_range=args.timestep_sampling,
                    )
                    timesteps = torch.from_numpy(timesteps).to(latents.device)

                if args.model != "dit":
                    timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.model == "dit":
                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                else:
                    if args.input_perturbation:
                        noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if not args.text_enable and not args.id_condition and not args.off_encoder:

                    if not isinstance(Encoder, tuple):
                        if not isinstance(batch["input_ids"], tuple):
                            encoder_hidden_states = Encoder(batch["input_ids"].to(weight_dtype))
                        else:
                            encoder_hidden_states = Encoder(batch["input_ids"])
                    else:

                        encoder_hidden_states1 = Encoder[0](batch["input_ids"][0].to(weight_dtype))
                        encoder_hidden_states2 = Encoder[1](batch["input_ids"][1].to(weight_dtype))
                        encoder_hidden_states = []

                        for i in range(encoder_hidden_states2.shape[0]):
                            encoder_hidden_state = []
                            encoder_hidden_state.append(encoder_hidden_states1[i*2].unsqueeze(0).unsqueeze(1))
                            encoder_hidden_state.append(encoder_hidden_states2[i].unsqueeze(0).unsqueeze(1))
                            encoder_hidden_state.append(encoder_hidden_states1[i*2+1].unsqueeze(0).unsqueeze(1))
                            encoder_hidden_states.append(torch.cat(encoder_hidden_state, dim=1))

                        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)


                    if not args.id_condition_encoder:
                        if encoder_hidden_states.dim() == 4:
                            encoder_hidden_states = encoder_hidden_states.squeeze(1)  # Ensure condition is of shape (batch_size, seq_len, feature_dim)
                        if encoder_hidden_states.dim() == 2:
                            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                        assert encoder_hidden_states.dim() == 3, f"Encoder hidden states should be of shape (batch_size, seq_len, feature_dim), but got {encoder_hidden_states.shape}"
                    else:
                        if encoder_hidden_states.dim() == 3:
                            encoder_hidden_states = encoder_hidden_states.squeeze(1)
                        assert encoder_hidden_states.dim() == 2, f"Encoder hidden states should be of shape (batch_size, feature_dim), but got {encoder_hidden_states.shape}"
                elif args.text_enable:
                    encoder_hidden_states = bring_text_embedding(batch["input_ids"].to(weight_dtype), args)

                elif args.id_condition or args.off_encoder:
                    encoder_hidden_states = batch["input_ids"].to(weight_dtype)
                    if args.off_encoder:
                        if encoder_hidden_states.dim() == 2:
                            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                        elif encoder_hidden_states.dim() == 4:
                            encoder_hidden_states = encoder_hidden_states.squeeze(1)
                        assert encoder_hidden_states.dim() == 3, f"Encoder hidden states should be of shape (batch_size, seq_len, feature_dim), but got {encoder_hidden_states.shape}"
                    elif args.id_condition:
                        if encoder_hidden_states.dim() == 1:
                            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                        elif encoder_hidden_states.dim() == 3:
                            encoder_hidden_states = encoder_hidden_states.squeeze(1)
                        assert encoder_hidden_states.dim() == 2, f"Encoder hidden states should be of shape (batch_size, feature_dim), but got {encoder_hidden_states.shape}"

                if accelerator.is_main_process and args.debug_condition and not args.id_condition and not args.off_encoder:
                    if "onehot" not in args.test_mode:
                        debug_validation_prompts = [torch.tensor([i,i], dtype=weight_dtype, device=accelerator.device).unsqueeze(0).unsqueeze(0) for i in range(1,11)]
                        debug_validation_prompts = [(p - 1) / 9 for p in debug_validation_prompts] # Normalize to [0, 1]
                    else :
                        debug_validation_prompts = [torch.eye(10, dtype=weight_dtype, device=accelerator.device)[i].unsqueeze(0) for i in range(10)]
                        debug_validation_prompts = torch.cat(debug_validation_prompts, dim=0).to(accelerator.device)
                        for i, prompt in enumerate(debug_validation_prompts):
                            result = Encoder(prompt)
                            print(f"Debug validation prompt {i}: {result.mean()}")


                # Get the target for loss depending on the model and prediction type.
                if args.model == "dit":
                    if args.precondition_outputs:
                        target = latents
                    else:
                        target = noise - latents
                else:
                    if args.prediction_type is not None:
                        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.dream_training:
                        noisy_latents, target = compute_dream_and_update_latents(
                            unet,
                            noise_scheduler,
                            timesteps,
                            noise,
                            noisy_latents,
                            target,
                            encoder_hidden_states,
                            args.dream_detail_preservation,
                        )

                # Predict the noise residual and compute loss
                if args.model == "dit":
                    model_pred = unet(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_projections=None,
                        return_dict=False,
                    )[0]
                    if args.precondition_outputs:
                        model_pred = model_pred * (-sigmas) + noisy_latents
                elif args.id_condition or args.id_condition_encoder:
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states= None, class_labels=encoder_hidden_states,  return_dict=False)[0]
                else:
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.model == "dit":
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()
                elif args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                if args.classification_loss or args.classification_loss_ce:

                    # Get embeddings from encoder_hidden_states
                    embeddings = encoder_hidden_states.squeeze()
                    if embeddings.dim() == 1:
                        embeddings = embeddings.unsqueeze(0)

                    # Convert input_ids to class labels (assuming they contain the ground truth)
                    if "onehot" in args.test_mode:
                        # For one-hot encoded labels, get the class index
                        class_labels = torch.argmax(batch['input_ids'], dim=-1).squeeze()
                    else:
                        # For other formats, you may need to adjust this based on your data format
                        class_labels = batch['input_ids'].long().squeeze()

                    # Compute contrastive loss (InfoNCE)
                    batch_size = embeddings.shape[0]

                    # Normalize embeddings
                    embeddings_normalized = F.normalize(embeddings, p=2, dim=1)

                    # Compute similarity matrix (batch_size x batch_size)
                    similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())

                    # Create mask for positive pairs (same class)
                    labels_expanded = class_labels.unsqueeze(1)
                    positive_mask = (labels_expanded == labels_expanded.t()).float()

                    # Remove self-similarity from positive mask
                    positive_mask = positive_mask - torch.eye(batch_size, device=positive_mask.device)

                    # Temperature scaling
                    temperature = getattr(args, 'contrastive_temperature', 0.07)
                    similarity_matrix = similarity_matrix / temperature

                    # Compute InfoNCE loss
                    # For each sample, compute loss against all other samples
                    log_prob = F.log_softmax(similarity_matrix, dim=1)

                    # Mask out self-similarity (diagonal)
                    mask = torch.eye(batch_size, device=log_prob.device, dtype=torch.bool)
                    log_prob = log_prob.masked_fill(mask, 0)

                    # For each positive pair, compute loss
                    contrastive_loss = 0
                    num_positive_pairs = 0

                    for i in range(batch_size):
                        positive_indices = torch.where(positive_mask[i] > 0)[0]
                        if len(positive_indices) > 0:
                            # Sum log probabilities for all positive pairs for sample i
                            contrastive_loss -= log_prob[i][positive_indices].sum() / len(positive_indices)
                            num_positive_pairs += len(positive_indices)

                    if num_positive_pairs > 0:
                        classification_loss = (contrastive_loss / batch_size) * args.classification_loss_weight
                    else:
                        # Fallback to cross-entropy if no positive pairs in batch
                        class_predictions = classification_head(encoder_hidden_states.squeeze())
                        classification_loss = F.cross_entropy(class_predictions, class_labels, reduction="mean") * args.classification_loss_weight
                if args.classification_loss_ce:
                    if "onehot" in args.test_mode:
                        # For one-hot encoded labels, get the class index
                        class_labels = torch.argmax(batch['input_ids'], dim=-1).squeeze()
                    else:
                        # For other formats, you may need to adjust this based on your data format
                        class_labels = batch['input_ids'].long().squeeze()
                    class_predictions = classification_head(encoder_hidden_states.squeeze())
                    classification_loss = F.cross_entropy(class_predictions, class_labels, reduction="mean") * args.classification_loss_weight

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                if args.classification_loss or args.classification_loss_ce:
                    avg_classification_loss = accelerator.gather(classification_loss.repeat(args.train_batch_size)).mean()
                    train_classification_loss += avg_classification_loss.item() / args.gradient_accumulation_steps

                # Backpropagate

                if args.classification_loss or args.classification_loss_ce:

                    accelerator.backward(loss, retain_graph=True)

                    accelerator.backward(classification_loss, retain_graph=False)
                else:
                    accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.text_enable or args.id_condition or args.off_encoder:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    else:
                        # Clip gradients for both models
                        if isinstance(Encoder, tuple):
                            params_to_clip = list(unet.parameters()) + list(Encoder[0].parameters()) + list(Encoder[1].parameters())
                        else:
                            params_to_clip = list(unet.parameters()) + list(Encoder.parameters())

                        if args.classification_loss or args.classification_loss_ce:
                            params_to_clip += list(classification_head.parameters())

                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if not args.text_enable and not args.id_condition and "onesample" in args.test_mode and not args.report_to == "wandb" and args.debug_condition:
                    for name, param in Encoder.named_parameters():
                        if "weight" in name:
                            print(f"{name}: {param.norm().item()}")

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    if args.offload_ema:
                        ema_unet.to(device="cuda", non_blocking=True)
                    ema_unet.step(unet.parameters())
                    if args.offload_ema:
                        ema_unet.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)

                if args.classification_loss or args.classification_loss_ce:
                    accelerator.log({"train_classification_loss": train_classification_loss}, step=global_step)
                    epoch_classification_loss += train_classification_loss
                    train_classification_loss = 0.0

                epoch_train_loss += train_loss
                epoch_steps += 1

                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 and not args.no_save:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(save_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Step-based validation
                if args.validation_steps is not None and accelerator.is_main_process:
                    step_true = global_step % args.validation_steps == 0 and global_step > 0

                    if args.validation_prompts is not None and step_true:
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                        # Run validation with appropriate encoder
                        encoder = None if args.text_enable else (None if (args.id_condition or args.off_encoder) else Encoder)
                        validation_kwargs = {
                            "vae": vae,
                            "Encoder": encoder,
                            "unet": unet,
                            "args": args,
                            "accelerator": accelerator,
                            "weight_dtype": weight_dtype,
                            "epoch": global_step,
                            "save_dir": save_dir,
                            "no_image": args.no_image
                        }

                        # Add validation_key only if needed
                        if not args.text_enable and validation_key is not None:
                            validation_kwargs["validation_key"] = validation_key

                        images = log_validation(**validation_kwargs)

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

                        save_dir2 = os.path.join(save_dir, "validation_images", f"checkpoint-{global_step}")
                        if not args.no_save and args.image_save:
                            os.makedirs(save_dir2, exist_ok=True)

                        for idx, image in enumerate(images):
                            label = validation_key[idx] if validation_key is not None and idx < len(validation_key) else "unknown"
                            image_path = os.path.join(save_dir2, f"validation_{idx}_label_{label}.png")
                            if args.image_save:
                                image.save(image_path)

                        if args.test_accuracy and validation_key is not None:
                            # Use the utility function for test accuracy
                            results = test_accuracy(
                                images=images,
                                validation_key=validation_key,
                                args=args,
                                train_dataset=train_dataset,
                                accelerator=accelerator,
                                Tester=Tester,
                                Tester2=Tester2 if ("composition" in args.test_mode and "object" not in args.test_mode) or ("attribute" in args.test_mode and "multi" not in args.test_mode and "disturber" not in args.test_mode) else None
                            )

                            # Use the utility function for logging results
                            log_accuracy_results(results, args, accelerator, global_step)

                            # Handle confidence logging
                            if results['confidence'] > 0.95:
                                with open(os.path.join(save_dir, "confidence.txt"), "a") as f:
                                    f.write(f"Step {global_step}: {results['confidence'].item():.2f}%\n")

                            # Track the best validation accuracy and keep a checkpoint of that model
                            best_save_dir = os.path.join(save_dir, "best_model")
                            os.makedirs(best_save_dir, exist_ok=True)

                            # Use joint accuracy for composition tasks, regular accuracy otherwise
                            current_accuracy = results['joint_accuracy'] if "composition" in args.test_mode and ("attribute" in args.test_mode and "multi" not in args.test_mode and "disturber" not in args.test_mode) else results['accuracy']

                            if current_accuracy > best_val_accuracy and current_accuracy > 0.0:
                                best_val_accuracy = current_accuracy
                                patience_counter = 0
                                accelerator.save_state(best_save_dir)
                                logger.info(f"Saved state to {best_save_dir}")
                                with open(os.path.join(save_dir, "best_val_accuracy.txt"), "w") as f:
                                    f.write(f"{best_val_accuracy:.4f}\n")
                                with open(os.path.join(best_save_dir, "best_val_accuracy_steps.txt"), "a") as f:
                                    f.write(f"Step {global_step} Epoch {epoch}\n")
                            else:
                                patience_counter += 1

                            logger.info(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
                            if args.early_stopping:
                                logger.info(f"Patience Counter: {patience_counter}/{args.early_stopping_patience_steps}")
                                should_stop = early_stopping_triggered(args.early_stopping_patience_steps)

                # The stop decision is made on the main process; share it so every rank leaves the loop together.
                if args.early_stopping and args.validation_steps is not None and global_step % args.validation_steps == 0 and global_step > 0:
                    should_stop = broadcast_object_list([should_stop])[0]

            if should_stop:
                break

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.classification_loss or args.classification_loss_ce:
                logs["classification_loss"] = classification_loss.detach().item()
                logs["total_encoder_loss"] = (loss + classification_loss).detach().item()
            progress_bar.set_postfix(**logs)


            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            avg_epoch_loss = epoch_train_loss / max(epoch_steps, 1)

            epoch_log_data = {
                "epoch": epoch,
                "avg_epoch_loss": avg_epoch_loss,
                "learning_rate": lr_scheduler.get_last_lr()[0],
            }

            if args.classification_loss or args.classification_loss_ce:
                avg_epoch_classification_loss = epoch_classification_loss / max(epoch_steps, 1)
                epoch_log_data["avg_epoch_classification_loss"] = avg_epoch_classification_loss

            accelerator.log(epoch_log_data, step=global_step)


        if accelerator.is_main_process:
            if args.validation_epochs is not None:
                epoch_true = epoch % args.validation_epochs == 0 and epoch > 0
            else:
                epoch_true = False

            if args.validation_prompts is not None and epoch_true:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                # Run validation with appropriate encoder
                encoder = None if args.text_enable else (None if (args.id_condition or args.off_encoder) else Encoder)
                validation_kwargs = {
                    "vae": vae,
                    "Encoder": encoder,
                    "unet": unet,
                    "args": args,
                    "accelerator": accelerator,
                    "weight_dtype": weight_dtype,
                    "epoch": global_step,
                    "save_dir": save_dir,
                    "no_image": args.no_image
                }

                # Add validation_key only if needed
                if not args.text_enable and validation_key is not None:
                    validation_kwargs["validation_key"] = validation_key

                images = log_validation(**validation_kwargs)

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

                save_dir2 = os.path.join(save_dir, "validation_images", f"checkpoint-{global_step}")
                if not args.no_save and args.image_save:
                    os.makedirs(save_dir2, exist_ok=True)

                for idx, image in enumerate(images):
                    label = validation_key[idx] if validation_key is not None and idx < len(validation_key) else "unknown"
                    image_path = os.path.join(save_dir2, f"validation_{idx}_label_{label}.png")
                    if args.image_save:
                        image.save(image_path)

                if args.test_accuracy and validation_key is not None:
                    # Use the utility function for test accuracy
                    results = test_accuracy(
                        images=images,
                        validation_key=validation_key,
                        args=args,
                        train_dataset=train_dataset,
                        accelerator=accelerator,
                        Tester=Tester,
                        Tester2=Tester2 if ("composition" in args.test_mode and "object" not in args.test_mode) else None
                    )

                if args.test_accuracy and validation_key is not None:
                    # Use the utility function for logging results
                    log_accuracy_results(results, args, accelerator, global_step)

                    # Handle confidence logging
                    if results['confidence'] > 0.95:
                        with open(os.path.join(save_dir, "confidence.txt"), "a") as f:
                            f.write(f"Step {global_step}: {results['confidence'].item():.2f}%\n")

                    # Track the best validation accuracy and keep a checkpoint of that model
                    best_save_dir = os.path.join(save_dir, "best_model")
                    os.makedirs(best_save_dir, exist_ok=True)

                    # Use joint accuracy for composition tasks, regular accuracy otherwise
                    current_accuracy = results['joint_accuracy'] if "composition" in args.test_mode else results['accuracy']

                    if current_accuracy >= best_val_accuracy and current_accuracy > 0.0:
                        best_val_accuracy = current_accuracy
                        patience_counter = 0
                        accelerator.save_state(best_save_dir)
                        logger.info(f"Saved state to {best_save_dir}")
                        with open(os.path.join(save_dir, "best_val_accuracy.txt"), "w") as f:
                            f.write(f"{best_val_accuracy:.4f}\n")
                        with open(os.path.join(best_save_dir, "best_val_accuracy_steps.txt"), "a") as f:
                            f.write(f"Step {global_step} Epoch {epoch}\n")
                    else:
                        patience_counter += 1

                    logger.info(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
                    if args.early_stopping:
                        logger.info(f"Patience Counter: {patience_counter}/{args.early_stopping_patience_epochs}")
                        should_stop = early_stopping_triggered(args.early_stopping_patience_epochs)

        if args.early_stopping and args.validation_epochs is not None:
            should_stop = broadcast_object_list([should_stop])[0]
        if should_stop:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not args.no_save:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        if args.id_condition or args.off_encoder:
            if args.model == "dit":
                pipeline = MyCustomPipeline_dit.from_pretrained(
                    save_dir,
                    vae=vae,
                    transformer=accelerator.unwrap_model(unet),
                    condition_encoder=None,
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                )
            else:
                pipeline = MyCustomPipeline.from_pretrained(
                    save_dir,
                    vae=vae,
                    unet=accelerator.unwrap_model(unet),
                    condition_encoder=None,
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                )
        elif isinstance(Encoder, tuple):
            if args.model == "dit":
                pipeline = MyCustomPipeline_dit.from_pretrained(
                    save_dir,
                    vae=vae,
                    transformer=accelerator.unwrap_model(unet),
                    condition_encoder=accelerator.unwrap_model(Encoder[0]),
                    condition_encoder2=accelerator.unwrap_model(Encoder[1]),
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                    id_condition=args.id_condition_encoder,
                )
            else:
                pipeline = MyCustomPipeline.from_pretrained(
                    save_dir,
                    vae=vae,
                    unet=accelerator.unwrap_model(unet),
                    condition_encoder=accelerator.unwrap_model(Encoder[0]),
                    condition_encoder2=accelerator.unwrap_model(Encoder[1]),
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                    id_condition=args.id_condition_encoder,
                )
        else:
            if args.model == "dit":
                pipeline = MyCustomPipeline_dit.from_pretrained(
                    save_dir,
                    vae=vae,
                    transformer=accelerator.unwrap_model(unet),
                    condition_encoder=accelerator.unwrap_model(Encoder),
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                    id_condition=args.id_condition_encoder,
                )
            else:
                pipeline = MyCustomPipeline.from_pretrained(
                    save_dir,
                    vae=vae,
                    unet=accelerator.unwrap_model(unet),
                    condition_encoder=accelerator.unwrap_model(Encoder),
                    image_encoder=None,
                    feature_extractor=None,
                    safety_checker=None,
                    id_condition=args.id_condition_encoder,
                )


        pipeline.save_pretrained(save_dir)

        images = []
        if args.validation_prompts is not None and not args.no_save:
            logger.info("Running inference for collecting generated images...")
            if args.model == "unet":
                scheduler_map = {
                    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
                    "EulerDiscreteScheduler": EulerDiscreteScheduler,
                    "DDPMScheduler": DDPMScheduler,
                }
                final_scheduler_cls = scheduler_map.get(args.validation_scheduler, EulerDiscreteScheduler)
                pipeline.scheduler = final_scheduler_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            if args.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

            with torch.autocast("cuda"):
                if args.id_condition or args.off_encoder:
                    batched_prompts = torch.cat(args.validation_prompts, dim=0)
                else:
                    if not isinstance(Encoder, tuple) and "attribute" not in args.test_mode:
                        batched_prompts = torch.stack(args.validation_prompts)

                    else:
                        batched_prompts = torch.stack([i[0] for i in args.validation_prompts])
                        batched_prompts2 = torch.stack([i[1] for i in args.validation_prompts])
                        batched_prompts = (batched_prompts, batched_prompts2)

                if args.validation_batch_size is None:
                    output = pipeline(batched_prompts, num_inference_steps=20, generator=generator)
                    images = output.images
                else:
                    for i in range(0, batched_prompts.shape[0], args.validation_batch_size):
                        if not isinstance(Encoder, tuple):
                            batch = batched_prompts[i:i + args.validation_batch_size]
                        else:
                            batch = (batched_prompts[0][i:i + args.validation_batch_size], batched_prompts[1][i:i + args.validation_batch_size])
                        output = pipeline(batch, num_inference_steps=20, generator=generator)
                        images += output.images

        if args.push_to_hub:
            save_model_card(args, repo_id, images, repo_folder=save_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=save_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

    if accelerator.is_main_process:
        logger.info("Training completed.")
        with open(os.path.join(save_dir, "completed.txt"), "w") as f:
            f.write(f"Training completed successfully with training steps {global_step}.\n")


if __name__ == "__main__":
    main()
