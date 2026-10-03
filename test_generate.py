#!/usr/bin/env python
# coding=utf-8
"""Generate images from the checkpoints of a trained run (UNet or DiT) with the custom pipelines."""
import argparse
import json
import os
import sys

import torch
import tqdm

sys.path.append("./diffusers/src")
from diffusers import AutoencoderKL, MyCustomPipeline, MyCustomPipeline_dit, UNet2DConditionModel
from diffusers.models.transformers import SD3Transformer2DModel
from model import condition_encoder, two_condition_encoder
from utils import is_composition, task_of, uses_two_token_condition

VAE_MODEL = "stabilityai/stable-diffusion-2-base"
COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE", "ORANGE", "CYAN", "GRAY", "WHITE", "BLACK"]
# kept in the output path so existing generation folders stay valid
OUTPUT_TAG = "guidance_scale{guidance_scale}_condition_scale1.0"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with a trained MOSAIC pipeline.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Training run directory (or a parent of it) that contains the checkpoint-* / best_model folders.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save generated images.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of images to generate per prompt.")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of images generated per pipeline call.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the torch generator.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. 'cuda', 'cpu'). Auto-detected if not set.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale (1.0 = off).")
    parser.add_argument("--counting", type=int, default=10, help="Number of count / position classes to generate.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of checkpoints to sample (evenly spaced, always including first and last).")
    parser.add_argument("--max_steps", type=int, default=None, help="Ignore checkpoints beyond this step.")
    parser.add_argument("--checkpoint_filter", type=str, default="best_model",
                        help="Use only this checkpoint step, '-1' for the latest, 'best_model' (default), or 'all' for evenly spaced checkpoints.")
    return parser.parse_args()


def checkpoint_step(name):
    return int(name.split("-")[1])


def has_checkpoints(path):
    return os.path.isdir(path) and any(f.startswith("checkpoint") or "best_model" in f for f in os.listdir(path))


def resolve_model_dir(model_dir):
    """Descend into sub-folders until a directory containing checkpoint-*/best_model is found."""
    current = model_dir.rstrip("/")
    if not os.path.isdir(current):
        raise FileNotFoundError(f"--model_dir {model_dir} does not exist.")
    while not has_checkpoints(current):
        candidates = [d for d in sorted(os.listdir(current)) if os.path.isdir(os.path.join(current, d))]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"No checkpoints in {current} and cannot pick a unique sub-folder (found {candidates}). "
                "Pass the run directory explicitly with --model_dir."
            )
        current = os.path.join(current, candidates[0])
    if current != model_dir.rstrip("/"):
        print(f"No checkpoints in {model_dir}; using {current}")
    return current


def test_mode_of(model_dir):
    """<root>/seed_<s>/<test_mode>/<param_string> -> <test_mode>"""
    return model_dir.rstrip("/").split("/")[-2]


def select_checkpoints(all_checkpoints, num_checkpoints, max_steps=None, checkpoint_filter=None):
    """Pick evenly spaced checkpoint-* folders (first and last always included)."""
    if checkpoint_filter == "best_model":
        return []  # best_model is appended separately by the caller

    names = sorted((f for f in all_checkpoints if f.startswith("checkpoint")), key=checkpoint_step)
    if max_steps is not None:
        names = [f for f in names if checkpoint_step(f) <= max_steps]
    if checkpoint_filter is not None:
        wanted = checkpoint_step(names[-1]) if checkpoint_filter == "-1" else int(checkpoint_filter)
        names = [f for f in names if checkpoint_step(f) == wanted]

    if len(names) <= num_checkpoints:
        return names
    last = len(names) - 1
    indices = {0, last} | {int(i * last / (num_checkpoints - 1)) for i in range(1, num_checkpoints - 1)}
    return [names[i] for i in sorted(indices)]


def build_prompts(test_mode, counting):
    """Return {prompt_key: condition tensor}. Shapes match build_validation_prompts in training:
    (1, 10) for count / position, (2, 10) for attribute (color1, color2) and composition (color, position)."""
    eye = torch.eye(10)
    prompts = {}
    task = task_of(test_mode)
    if task == "attribute":
        for i, color1 in enumerate(COLOR_NAMES):
            for j, color2 in enumerate(COLOR_NAMES):
                prompts[f"{color1}_{color2}"] = torch.stack((eye[i], eye[j]))
    elif is_composition(test_mode):
        for i in range(counting):
            for color in COLOR_NAMES:
                prompts[f"{color}_{i + 1}"] = torch.stack((eye[COLOR_NAMES.index(color)], eye[i]))
    else:
        for i in range(counting):
            prompts[i + 1] = eye[i].unsqueeze(0)
    return prompts


def load_encoder(checkpoint_dir, test_mode):
    if not os.path.exists(os.path.join(checkpoint_dir, "condition_encoder", "config.json")):
        raise FileNotFoundError(f"{checkpoint_dir} has no condition_encoder/config.json; the checkpoint must contain its own condition encoder.")
    encoder_cls = two_condition_encoder if uses_two_token_condition(test_mode) else condition_encoder
    return encoder_cls.from_pretrained(checkpoint_dir, subfolder="condition_encoder")


def is_dit_run(model_dir):
    """Detect the backbone from the saved config instead of the path string ("condition" contains "dit")."""
    for sub in ("", "best_model", *sorted(os.listdir(model_dir))):
        cfg = os.path.join(model_dir, sub, "unet", "config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                return "Transformer" in json.load(f).get("_class_name", "")
    if os.path.exists(os.path.join(model_dir, "transformer")):
        return True
    if os.path.exists(os.path.join(model_dir, "unet")):
        return False
    raise FileNotFoundError(f"Cannot determine backbone type (unet/dit) from {model_dir}")


def load_pipeline(args, checkpoint_dir, vae, device):
    encoder = load_encoder(checkpoint_dir, args.test_mode)
    if args.use_dit:
        transformer = SD3Transformer2DModel.from_pretrained(checkpoint_dir, subfolder="unet")
        pipeline = MyCustomPipeline_dit.from_pretrained(args.model_dir, vae=vae, transformer=transformer, condition_encoder=encoder)
        backbone = pipeline.transformer
    else:
        unet = UNet2DConditionModel.from_pretrained(checkpoint_dir, subfolder="unet")
        pipeline = MyCustomPipeline.from_pretrained(args.model_dir, vae=vae, unet=unet, condition_encoder=encoder)
        backbone = pipeline.unet

    for module in (backbone, pipeline.vae, pipeline.condition_encoder):
        module.eval().requires_grad_(False)
    pipeline = pipeline.to(device)
    pipeline.condition_encoder.to(device)
    return pipeline


def main():
    args = parse_args()

    if args.num_samples % args.batch_size != 0:
        args.batch_size = args.num_samples
    if args.checkpoint_filter == "all":
        args.checkpoint_filter = None
    if args.checkpoint_filter is not None:
        args.max_steps = None

    args.model_dir = resolve_model_dir(args.model_dir)
    args.test_mode = test_mode_of(args.model_dir)
    args.use_dit = is_dit_run(args.model_dir)
    args.output_dir = os.path.join(args.output_dir, "dit" if args.use_dit else "unet")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, subfolder="vae")

    all_checkpoints = [f for f in os.listdir(args.model_dir) if f.startswith("checkpoint") or "best_model" in f]
    # a run whose accuracy never left 0 can have an empty best_model/ folder: skip anything without weights
    incomplete = [f for f in all_checkpoints if not os.path.exists(os.path.join(args.model_dir, f, "unet", "config.json"))]
    if incomplete:
        print(f"Skipping checkpoints without saved weights: {incomplete}")
        all_checkpoints = [f for f in all_checkpoints if f not in incomplete]
    selected = select_checkpoints(all_checkpoints, args.num_epochs, args.max_steps, args.checkpoint_filter)
    if selected:
        steps = [checkpoint_step(f) for f in all_checkpoints if f.startswith("checkpoint")]
        print(f"Available steps: {min(steps)} to {max(steps)}")
        print(f"Selected steps: {sorted(checkpoint_step(f) for f in selected)}")
    selected.extend(f for f in all_checkpoints if "best_model" in f)
    print(f"Checkpoints to process: {selected}")

    run_parts = args.model_dir.rstrip("/").split("/")[-4:]
    output_tag = OUTPUT_TAG.format(guidance_scale=args.guidance_scale)
    prompts = build_prompts(args.test_mode, args.counting)
    expected_images = len(prompts) * args.num_samples

    pbar = tqdm.tqdm(selected, desc="Processing checkpoints")
    for checkpoint in pbar:
        pbar.set_description(f"Processing {checkpoint}")
        checkpoint_dir = os.path.join(args.model_dir, checkpoint)
        save_dir = os.path.join(args.output_dir, f"gen_seed_{args.seed}", *run_parts, output_tag, "images", checkpoint)

        if os.path.isdir(save_dir) and len(os.listdir(save_dir)) == expected_images:
            print(f"Skipping {checkpoint_dir}: already has {expected_images} images.")
            continue

        pipeline = load_pipeline(args, checkpoint_dir, vae, device)
        os.makedirs(save_dir, exist_ok=True)  # only after the checkpoint loaded, so a crash leaves no empty folder
        generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

        with torch.no_grad():
            for prompt_key, prompt in prompts.items():
                for start in range(0, args.num_samples, args.batch_size):
                    batch = min(args.batch_size, args.num_samples - start)
                    batch_prompts = prompt.repeat(batch, 1, 1).to(device)
                    images = pipeline(
                        batch_prompts,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        guidance_scale=args.guidance_scale,
                    ).images
                    for i, image in enumerate(images):
                        image.save(os.path.join(save_dir, f"prompt_key_{prompt_key}_sample_{start + i:02d}.png"))

        print(f"Completed generation for checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
