import random

import torch


def make_collate_fn(args):
    """Stack the condition tokens and apply classifier-free-guidance dropout.

    A sample's condition is either one tensor (count/position: (10,)) or a tuple of two
    tensors (attribute: (color1, color2), composition: (color, position)). Tuples are
    stacked into a (B, 2, 10) tensor. With probability `args.drop_rate` the condition is
    zeroed (per sample for single tensors, per batch for tuples, as in the original runs).
    """

    def collate_fn(examples):
        first = examples[0]["input_ids"]
        if isinstance(first, torch.Tensor):
            input_ids = torch.stack(
                [
                    example["input_ids"] if random.random() > args.drop_rate else torch.zeros_like(example["input_ids"])
                    for example in examples
                ]
            )
        else:
            keep = random.random() > args.drop_rate
            input_ids1 = torch.stack([example["input_ids"][0] if keep else torch.zeros_like(example["input_ids"][0]) for example in examples])
            input_ids2 = torch.stack([example["input_ids"][1] if keep else torch.zeros_like(example["input_ids"][1]) for example in examples])
            input_ids = torch.cat((input_ids1.unsqueeze(1), input_ids2.unsqueeze(1)), dim=1)

        # VAE latent cache: examples carry either `presaved_latents` (cache hit) or
        # `image` + `latent_path` (cache miss, train.py encodes and writes the file).
        batch = {"input_ids": input_ids}
        has_latent = torch.tensor(["presaved_latents" in example for example in examples])
        if has_latent.any():
            latents_values = torch.stack([example["presaved_latents"] for example in examples if "presaved_latents" in example])
            batch["latents_values"] = latents_values.to(memory_format=torch.contiguous_format).float()
        if not has_latent.all():
            pixel_values = torch.stack([example["image"] for example in examples if "image" in example])
            batch["pixel_values"] = pixel_values.to(memory_format=torch.contiguous_format).float()
            batch["latent_path"] = [example["latent_path"] for example in examples if "latent_path" in example]
        if has_latent.any() and not has_latent.all():
            # mixed batch: remember which positions (in input_ids order) came from the cache
            batch["latent_mask"] = has_latent
        return batch

    return collate_fn
