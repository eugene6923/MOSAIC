import torch

from utils import is_composition, task_of


def build_validation_prompts(args, train_dataset, weight_dtype, device):
    """Return (prompts, validation_key) for the run's task.

    Each prompt is a tensor of one-hot tokens: shape (1, 10) for count/position and
    (2, 10) for attribute (color1, color2) and composition (color, position).
    Every prompt is repeated `args.val_num_samples` times.
    """
    task = task_of(args.test_mode)
    one_hot = torch.eye(10)
    prompts, validation_key = [], []

    if task == "count" and is_composition(args.test_mode):
        color_keys = list(train_dataset.color_dict.keys())
        for _ in range(args.val_num_samples):
            for k in range(10):
                for i in range(10):
                    prompts.append(torch.stack((one_hot[k], one_hot[i])))
                    validation_key.append(f"{color_keys[k]}_{i + 1}")

    elif task == "count":
        for i in range(10):
            for _ in range(args.val_num_samples):
                prompts.append(one_hot[i].unsqueeze(0).to(dtype=weight_dtype, device=device))
                validation_key.append(i + 1)

    elif task == "attribute":
        color_keys = list(train_dataset.color_dict.keys())
        for _ in range(args.val_num_samples):
            for i in range(10):
                for k in range(10):
                    prompts.append(torch.stack((one_hot[i], one_hot[k])))
                    validation_key.append(f"{color_keys[i]}_{color_keys[k]}")

    elif is_composition(args.test_mode):
        color_keys = list(train_dataset.color_dict.keys())
        position_keys = list(train_dataset.position_cond_dict.keys())
        for _ in range(args.val_num_samples):
            for k in range(10):
                for i in range(10):
                    prompts.append(torch.stack((one_hot[k], one_hot[i])))
                    validation_key.append(f"{color_keys[k]}_{position_keys[i]}")

    else:
        position_keys = list(train_dataset.position_cond_dict.keys())
        for _ in range(args.val_num_samples):
            for i in range(10):
                prompts.append(one_hot[i].unsqueeze(0))
                validation_key.append(position_keys[i])

    return prompts, validation_key
