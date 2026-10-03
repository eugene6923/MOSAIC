import re
import torch


def build_validation_prompts(args, train_dataset, weight_dtype, device):
    validation_key = None

    if "count" in args.test_mode:
        if "composition" in args.test_mode:
            counting = 10
            onehot_condition = torch.eye(counting)
            prompts = [torch.cat((onehot_condition[0].unsqueeze(0), onehot_condition[i].unsqueeze(0)), dim=0) for i in range(counting)] * args.val_num_samples
            validation_key = [list(train_dataset.color_dict.keys())[0] + "_" + str(i + 1) for i in range(counting)] * args.val_num_samples

        else:
            condition_set = torch.eye(10, dtype=weight_dtype, device=device)
            prompts = []
            for i in range(10):
                for _ in range(args.val_num_samples):
                    prompts.append(condition_set[i].unsqueeze(0).unsqueeze(0))
            validation_key = []
            for i in range(1, 11):
                for _ in range(args.val_num_samples):
                    validation_key.append(i)

    elif "attribute" in args.test_mode or "attribution" in args.test_mode:
        color_condition = torch.eye(10)
        color_keys = list(train_dataset.color_dict.keys())
        prompts = []
        validation_key = []
        for _ in range(args.val_num_samples):
            for i in range(10):
                for k in range(10):
                    color = color_condition[i].unsqueeze(0)
                    color2 = color_condition[k].unsqueeze(0)
                    prompts.append(torch.cat((color, color2), dim=0))
                    validation_key.append(f"{color_keys[i]}_{color_keys[k]}")
    

    elif "position" in args.test_mode:
        if "composition" in args.test_mode:
            color_condition = torch.eye(10)
            position_condition = torch.eye(10)
            color_keys = list(train_dataset.color_dict.keys())
            position_keys = list(train_dataset.position_cond_dict.keys())
            prompts = []
            validation_key = []
            for _ in range(args.val_num_samples):
                for k in range(10):
                    for i in range(10):
                        color = color_condition[k].unsqueeze(0)
                        position = position_condition[i].unsqueeze(0)
                        prompts.append(torch.cat((color, position), dim=0))
                        validation_key.append(f"{color_keys[k]}_{position_keys[i]}")
        else:
            position_condition = torch.eye(10)
            position_keys = list(train_dataset.position_cond_dict.keys())
            prompts = []
            validation_key = []
            for _ in range(args.val_num_samples):
                for i in range(10):
                    position = position_condition[i].unsqueeze(0)
                    prompts.append(position)
                    validation_key.append(f"{position_keys[i]}")
    
    
    else:
        raise ValueError(f"Unknown test mode {args.test_mode}")

    return prompts, validation_key
