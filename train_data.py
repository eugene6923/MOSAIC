import random

import torch


def make_collate_fn(args):
    def collate_fn(examples):
        if not args.id_condition:
            if isinstance(examples[0]["input_ids"], torch.Tensor):
                input_ids = torch.stack(
                    [
                        example["input_ids"]
                        if random.random() > args.drop_rate
                        else torch.zeros_like(example["input_ids"])
                        for example in examples
                    ]
                )
            elif isinstance(examples[0]["input_ids"], tuple):
                random_chance = random.random() > args.drop_rate
                if not isinstance(examples[0]["input_ids"][0], tuple):
                    input_ids1 = torch.stack(
                        [
                            example["input_ids"][0]
                            if random_chance
                            else torch.zeros_like(example["input_ids"][0])
                            for example in examples
                        ]
                    )
                    input_ids2 = torch.stack(
                        [
                            example["input_ids"][1]
                            if random_chance
                            else torch.zeros_like(example["input_ids"][1])
                            for example in examples
                        ]
                    )
                    input_ids = torch.cat((input_ids1.unsqueeze(1), input_ids2.unsqueeze(1)), dim=1)
                else:
                    input_ids_color1 = torch.stack(
                        [
                            example["input_ids"][0][0]
                            if random_chance
                            else torch.zeros_like(example["input_ids"][0][0])
                            for example in examples
                        ]
                    )
                    input_ids_color2 = torch.stack(
                        [
                            example["input_ids"][0][1]
                            if random_chance
                            else torch.zeros_like(example["input_ids"][0][1])
                            for example in examples
                        ]
                    )
                    input_ids_position = torch.stack(
                        [
                            example["input_ids"][1]
                            if random_chance
                            else torch.zeros_like(example["input_ids"][1])
                            for example in examples
                        ]
                    )
                    input_ids = torch.cat((input_ids_color1, input_ids_color2), dim=0), input_ids_position
        else:
            input_ids = torch.stack([example["input_ids"] for example in examples])

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
