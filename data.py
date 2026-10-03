import os
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import random
import glob
from torch.utils.data import Dataset
import fnmatch
import json
from collections import Counter
import re
import pandas as pd
import io
from pathlib import Path


def _dataset_folder_name(path):
    """'./mosaic/data/comfort_ball_count' -> 'comfort_ball_count',
    './mosaic/data/comfort_ball_position5/behind/default' -> 'comfort_ball_position5'."""
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3] if len(parts) >= 3 and parts[-2] == "behind" else parts[-1]


def _sample_with_latent_cache(dataset, idx, condition):
    """Build one training sample. Images are always on disk; the VAE latent is used
    from `<presaved_path>/<image name>.pt` when it exists, otherwise the image is returned
    together with `latent_path` so train.py can create the cache file."""
    image_path = dataset.data[idx]
    sample = {'input_ids': condition}
    if dataset.ood_filter or dataset.id_filter:
        sample['path'] = image_path

    latent_path = os.path.join(dataset.presaved_path, os.path.basename(image_path).replace(".png", ".pt"))
    if os.path.exists(latent_path):
        try:
            sample['presaved_latents'] = torch.load(latent_path)
            return sample
        except Exception:
            pass  # corrupted cache file: fall back to the image and let train.py rewrite it

    img = Image.open(image_path).convert("RGB")
    sample['image'] = dataset.transform(img)
    sample['latent_path'] = latent_path
    return sample


def get_dataset(type, resolution=512, interpolation=Image.BICUBIC, center_crop=False, random_flip=False, path = None, use_text_encoder=False, presaved_path="./presaved_latents", data_name=False,id_condition=False, checkpoint_filter=None, color_filter=False, all_filter= False, ood_filter=False, id_filter=False, original_type= None, joint_test=False, vae_subset=False):

    if type == "original_dataset" or type == "generated_dataset":
        return TestDataset(path, original_type = original_type, resolution=resolution, type = type, data_name=data_name, checkpoint_filter= checkpoint_filter, color_filter= color_filter, all_filter = all_filter, ood_filter = ood_filter, id_filter = id_filter, joint_test = joint_test, subset= vae_subset)
    
    elif "attribute" in type:
        if "complex"  in type:
            path = "./mosaic/data/comfort_ball_attribute_multi/behind/distractor"
        else: path = "./mosaic/data/comfort_ball_attribute/behind/distractor"

        return ComfortBallDataset_attribute(path, subset = type, resolution=resolution, interpolation=interpolation, center_crop=center_crop, random_flip=random_flip, use_text_encoder=use_text_encoder, presaved_path = presaved_path,id_condition=id_condition, ood_filter = ood_filter, id_filter = id_filter)

    elif "position" in type:
        if "composition" not in type:
            if "complex" in type:
                path = "./mosaic/data/comfort_ball_position5_disturber/behind/default"
            else:
                path = "./mosaic/data/comfort_ball_position5/behind/default"
            return ComfortBallDataset_position(path, subset = type, resolution=resolution, interpolation=interpolation, center_crop=center_crop, random_flip=random_flip, use_text_encoder=use_text_encoder, presaved_path = presaved_path,id_condition=id_condition, ood_filter = ood_filter, id_filter = id_filter)
        else:
            path = "./mosaic/data/comfort_ball_position5_colors/behind/default"
            return ComfortBallDataset_position(path, subset = type, resolution=resolution, interpolation=interpolation, center_crop=center_crop, random_flip=random_flip, use_text_encoder=use_text_encoder, presaved_path = presaved_path,id_condition=id_condition, ood_filter = ood_filter, id_filter = id_filter)

    elif "count" in type:
        if "composition" in type:
            path = './mosaic/data/comfort_ball_change25_colors/behind/distractor'
            return ComfortBallDataset_counting(path, subset = type, resolution=resolution, interpolation=interpolation, center_crop=center_crop, random_flip=random_flip, use_text_encoder=use_text_encoder, presaved_path = presaved_path,id_condition=id_condition, ood_filter = ood_filter, id_filter = id_filter)

        else:
            if path is None:
                path = "./mosaic/data/comfort_ball_count"
            return ComfortBallDataset_counting(path, subset = type, resolution=resolution, interpolation=interpolation, center_crop=center_crop, random_flip=random_flip, use_text_encoder=use_text_encoder, presaved_path = presaved_path,id_condition=id_condition, id_filter = id_filter)

    else:
        raise ValueError(f"Unknown dataset type: {type}")


class ComfortBallDataset_counting(Dataset):
    def __init__(self, path, subset, resolution=512, interpolation=Image.BICUBIC, center_crop=False, random_flip=False, use_text_encoder=False, presaved_path = None,id_condition=False, ood_filter=False, id_filter=False):
        # Assign filter flags early so they can be used throughout __init__
        self.ood_filter = ood_filter
        self.id_filter = id_filter
        
        self.onehot = True if "onehot" in subset else False
        self.transform = T.Compose(            
        [
            T.Resize(resolution, interpolation=interpolation),  # Use dynamic interpolation method
            T.RandomHorizontalFlip() if random_flip else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

        self.color_dict = {"RED": (1, 0, 0, 1),
                      "GREEN": (0, 1, 0, 1),
                      "BLUE": (0, 0, 1, 1),
                      "YELLOW": (1, 1, 0, 1),
                      "PURPLE": (1, 0, 1, 1),
                      "ORANGE": (1, 0.5, 0, 1),
                      "CYAN": (0, 1, 1, 1),
                      "GRAY": (0.5, 0.5, 0.5, 1),
                      "WHITE": (1, 1, 1, 1),
                      "BLACK": (0, 0, 0, 1)}
        
        self.color_keys = list(self.color_dict.keys())
        self.num_counts = 10
        self.num_colors = len(self.color_dict.keys())

        color_condition = torch.eye(self.num_colors)
        counting_condition = torch.eye(self.num_counts)

        color_cond_dict = {str(self.color_dict[i]): color_condition[j] for j, i in enumerate(self.color_dict.keys())}
        color_dict_reverse = {str(v): k for k, v in self.color_dict.items()}
        if "resolution" not in presaved_path:
            presaved_path = os.path.join(presaved_path, "resolution_128")
        self.presaved_path = os.path.join(presaved_path, _dataset_folder_name(path))
        

        data = [i for i in glob.glob(os.path.join(path,'*.png')) if int(i.split("/")[-1].split("_")[-1][:-4]) <= self.num_counts]

        skewed = False
        offset = 0
        size = None

        if "_" in subset:
            size = int(subset.split("_")[1])
            subset = subset.split("_")[0]

            if "composition" not in subset: 
    
                if "skewed" in subset:
                    if size == 2000:
                        skewed = True
                        sample_size = [45, 36, 29, 23, 18, 14, 12, 9, 7, 7]
                    elif size == 100000:
                        skewed = True
                        sample_size = [2255, 1795, 1435, 1145, 915, 730, 585, 465, 375, 300]
                    else:
                        raise ValueError(f"Unknown skewed size: {size}")

                else:
                    sample_size = [size // 10] * 10  # Default uniform distribution if not skewed
                
        else:
            raise ValueError(f"Provide the dataset size: {subset}_<size>")
            

        if "composition" in subset:
            filter_list = self.make_ood_filter_list(subset, offset=offset)
            
        else:
            filter_list = []

        if "composition" not in subset:
            self.data = [f for f in data if not any(fnmatch.fnmatch(os.path.basename(f)[:-4], pat) for pat in filter_list) and int(f.split('_')[-2]) < sample_size[int(f.split('_')[-1][:-4])-1]]
        else:
            match = re.match(r"composition(\d+)", subset)
            if not match:
                raise ValueError(f"Unknown composition subset: {subset}")
            
            N = int(match.group(1))
            if not ood_filter and not id_filter:
                if N ==0:
                    self.data = [f for f in data if not any(fnmatch.fnmatch(os.path.basename(f)[:-4], pat) for pat in filter_list) and int(f.split('_')[-2]) < size//100]
                elif N ==1:
                    self.data = [f for f in data if not any(fnmatch.fnmatch(os.path.basename(f)[:-4], pat) for pat in filter_list) and int(f.split('_')[-2]) < size//90]
                elif N == 3:
                    self.data = [f for f in data if not any(fnmatch.fnmatch(os.path.basename(f)[:-4], pat) for pat in filter_list) and int(f.split('_')[-2]) < size//70]

                elif N ==5:
                    self.data = [f for f in data if not any(fnmatch.fnmatch(os.path.basename(f)[:-4], pat) for pat in filter_list) and int(f.split('_')[-2]) < size//50]

                elif N ==8:
                    self.data = [f for f in data if not any(fnmatch.fnmatch(os.path.basename(f)[:-4], pat) for pat in filter_list) and int(f.split('_')[-2]) < size//20]

            else:
                if self.ood_filter:
                    self.data = [f for f in data if any(fnmatch.fnmatch(os.path.basename(f)[:-4], pat) for pat in filter_list) and int(f.split('_')[-2]) < 50]
                elif self.id_filter:
                    self.data = [f for f in data if not any(fnmatch.fnmatch(os.path.basename(f)[:-4], pat) for pat in filter_list) and int(f.split('_')[-2]) < 50]
        
        data_count = {}
        data_color = {}
        
        color_dict_reverse = {str(v): k for k, v in self.color_dict.items()}
        for f in self.data:
            count = int(f.split('_')[-1][:-4])
            color = f.split('/')[-1].split('_')[0]
            data_count[count] = data_count.get(count, 0) + 1
            data_color[color_dict_reverse[color]] = data_color.get(color_dict_reverse[color], 0) + 1

        print(f"len(self.data): {len(self.data)}")

        self.colors = [color_cond_dict[i.split('/')[-1].split('_')[0]] for i in self.data]
        self.countings = [counting_condition[int(i.split('/')[-1].split('_')[-1][:-4])-1] for i in self.data]

        if len(self.data) != len(self.colors) or len(self.data) != len(self.countings):
            raise ValueError("Data and condition lists must have the same length.")

        self.subset = subset
        self.resolution = resolution


    def make_ood_filter_list(self, subset, offset=0):
        # Extract the number from 'oodN'
        
        match = re.match(r"composition(\d+)", subset)
        if not match:
            raise ValueError(f"Unknown composition subset: {subset}")
        
        N = int(match.group(1))
        filter_list = []
        
        # Use color tuple strings instead of color names
        color_tuples = [str(self.color_dict[key]) for key in self.color_keys]
        
        # Generate N complete diagonals (each with 10 items)
        for diagonal in range(N):
            for count in range(1, self.num_counts + 1):  # 1 to 10
                color_idx = (diagonal + count - 1 + offset) % self.num_colors  # Wrap around colors
                filter_list.append(f"{color_tuples[color_idx]}_*_{count}")
        
        expected_length = 10 * N
        if len(filter_list) != expected_length:
            raise ValueError(f"Filter list length mismatch for {subset}: expected {expected_length}, got {len(filter_list)}")
    
        return filter_list


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return _sample_with_latent_cache(self, idx, (self.colors[idx], self.countings[idx]))


class ComfortBallDataset_position(Dataset):
    def __init__(self, path, subset, resolution=512, interpolation=Image.BICUBIC, center_crop=False, random_flip=False, use_text_encoder=False, presaved_path = None,id_condition=False, ood_filter = False, id_filter=False):
        self.onehot = True if "onehot" in subset else False
        self.transform = T.Compose(            
        [
            T.Resize(resolution, interpolation=interpolation),  # Use dynamic interpolation method
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        

        skewed = False
        offset = 0
        training = False
        validation = False

        self.num_positions = 10
        position_condition = torch.eye(self.num_positions)
        
        self.position_cond_dict = {"position1": position_condition[0],
                            "position2": position_condition[1],
                            "position3": position_condition[2],
                            "position4": position_condition[3],
                            "position5": position_condition[4],
                            "position6": position_condition[5],
                            "position7": position_condition[6],
                            "position8": position_condition[7],
                            "position9": position_condition[8],
                            "position10": position_condition[9]}

        
        if "composition" in subset:
            self.color_dict = {"RED": (1, 0, 0, 1),
                        "GREEN": (0, 1, 0, 1),
                        "BLUE": (0, 0, 1, 1),
                        "YELLOW": (1, 1, 0, 1),
                        "PURPLE": (1, 0, 1, 1),
                        "ORANGE": (1, 0.5, 0, 1),
                        "CYAN": (0, 1, 1, 1),
                        "GRAY": (0.5, 0.5, 0.5, 1),
                        "WHITE": (1, 1, 1, 1),
                        "BLACK": (0, 0, 0, 1)}
    
            self.color_keys = list(self.color_dict.keys())
            self.num_colors = len(self.color_dict.keys())
            color_condition = torch.eye(self.num_colors)
            color_cond_dict = {str(self.color_dict[i]): color_condition[j] for j, i in enumerate(self.color_dict.keys())}

            filter_list = self.make_ood_filter_list(subset)
            print(f"Filter list for {subset}: {filter_list}")

            data = glob.glob(os.path.join(path,'*.png'))

            
            if not ood_filter:
                self.data = [j for j in data 
                            if ("composite" not in j and
                        # Check that this file doesn't match any of the excluded diagonal patterns
                        not any(
                            any(
                                j.split('/')[-1].split('_')[1] == color_val and 
                                j.split('/')[-1].split('_')[-2] == angle_range
                                for color_val, angle_range in diagonal
                            )
                            for diagonal in filter_list
                        )
                    )
                    ]
            else:
                self.data = [j for j in data 
                            if ("composite" not in j and
                        # Check that this file matches any of the excluded diagonal patterns
                        any(
                            any(
                                j.split('/')[-1].split('_')[1] == color_val and 
                                j.split('/')[-1].split('_')[-2] == angle_range
                                for color_val, angle_range in diagonal
                            )
                            for diagonal in filter_list
                        )
                    )
                    ]
                size = 50
            
        else:
            self.data = [i for i in glob.glob(os.path.join(path,'*.png')) if "composite" not in i]

        if "training" in subset:
            training = True
            subset = subset.replace("_training", "")
        if "validation" in subset:   
            validation = True
            subset = subset.replace("_validation", "")


        size = int(subset.split("_")[-1]) if "_" in subset else None
        

        if training and validation:
            raise ValueError("Subset cannot be both train and validation.")
    
        if "composition" not in subset:
            
            if size == 2000:
                if "train" in subset:
                    distribution = [451,359,287,229,183,146,117,93,75,60]
                elif "reverse" in subset:
                    distribution = [451,359,287,229,183,146,117,93,75,60]
                    distribution.reverse()
                else:
                    distribution = [200] * 10
            elif size == 100000:
                if "train" in subset:
                    distribution = [22550, 17950, 14350, 11450, 9150, 7300, 5850, 4650, 3750, 3000]
                elif "reverse" in subset:
                    distribution = [22550, 17950, 14350, 11450, 9150, 7300, 5850, 4650, 3750, 3000]
                    distribution.reverse()
                else:
                    distribution = [10000] * 10
            elif size == 20000:
                if "train" in subset:
                    distribution = [4400, 3600, 2900, 2300, 1800, 1500, 1200, 900, 800, 600]
                elif "reverse" in subset:
                    distribution = [4400, 3600, 2900, 2300, 1800, 1500, 1200, 900, 800, 600]
                    distribution.reverse()
                else:
                    distribution = [2000] * 10
            elif size == 10000:
                if "train" in subset:
                    distribution = [2255, 1795, 1435, 1145, 915, 730, 585, 465, 375, 300]
                elif "reverse" in subset:
                    distribution = [2255, 1795, 1435, 1145, 915, 730, 585, 465, 375, 300]
                    distribution.reverse()
                else:
                    distribution = [1000] * 10
            elif size == 50000:
                if "train" in subset:
                    distribution = [11275, 8975, 7175, 5725, 4575, 3650, 2925, 2325, 1875, 1500]
                elif "reverse" in subset:
                    distribution = [11275, 8975, 7175, 5725, 4575, 3650, 2925, 2325, 1875, 1500]
                    distribution.reverse()
                else:
                    distribution = [5000] * 10
            
        else:
            if size == 50 or id_filter or ood_filter:
                distribution = [50] * 10
            elif size == 100000:
                if "train" in subset:
                    distribution = [2255, 1795, 1435, 1145, 915, 730, 585, 465, 375, 300]
                elif "reverse" in subset:
                    distribution = [2255, 1795, 1435, 1145, 915, 730, 585, 465, 375, 300]
                    distribution.reverse()
                else:
                    distribution = [1000] * 10

            elif size == 20000:
                if "train" in subset:
                    distribution = [451,359,287,229,183,146,117,93,75,60]
                elif "reverse" in subset:
                    distribution = [451,359,287,229,183,146,117,93,75,60]
                    distribution.reverse()
                else:
                    distribution = [200] * 10
            elif size == 10000:
                if "train" in subset:
                    distribution = [225, 179, 143, 114, 91, 73, 58, 46, 37, 34]
                elif "reverse" in subset:
                    distribution = [225, 179, 143, 114, 91, 73, 58, 46, 37, 34]
                    distribution.reverse()
                else:
                    distribution = [100] * 10
            elif size == 50000:
                if "train" in subset:
                    distribution = [1128, 898, 718, 572, 457, 365, 292, 232, 187, 151]
                elif "reverse" in subset:
                    distribution = [1128, 898, 718, 572, 457, 365, 292, 232, 187, 151]
                    distribution.reverse()
                else:
                    distribution = [500] * 10
            elif size == 2000:
                if "train" in subset:
                    distribution = [45, 36, 29, 23, 18, 14, 12, 9, 7, 7]
                elif "reverse" in subset:
                    distribution = [45, 36, 29, 23, 18, 14, 12, 9, 7, 7]
                    distribution.reverse()
                else:
                    distribution = [20] * 10
            
            N = int(re.search(r"composition(\d+)", subset).group(1))

            if not ood_filter and not id_filter:
                if N == 1:
                    distribution = [d * 10//9 for d in distribution]
                elif N == 3:
                    distribution = [d * 10//7 for d in distribution]
                elif N == 5:
                    distribution = [d * 10//5 for d in distribution]
                elif N == 8:
                    distribution = [d * 5 for d in distribution] 


        if not training and not validation:
            self.data = [f for f in self.data if int(f.split('_')[-1][:-4]) < distribution[int(f.split('_')[-2].replace("position", ""))-1]]
        else:
            if training:
                self.data = [f for f in self.data if int(f.split('_')[-1][:-4]) < 100]
            elif validation:
                self.data = [f for f in self.data if int(f.split('_')[-1][:-4]) >= 10000 and int(f.split('_')[-1][:-4]) < 10100]

        # group by position for debugging
        if "composition" in subset:
            data_position = {}
            for f in self.data:
                position = f.split('_')[-2]
                data_position[position] = data_position.get(position, 0) + 1
            print(data_position)
            data_color = {}
            color_dict_reverse = {str(v): k for k, v in self.color_dict.items()}
            for f in self.data:
                color = f.split('/')[-1].split('_')[1]
                data_color[color_dict_reverse[color]] = data_color.get(color_dict_reverse[color], 0) + 1
            print(data_color)

        if "composition" in subset: self.colors = [color_cond_dict[i.split('/')[-1].split('_')[1]] for i in self.data]
        self.positions  = [self.position_cond_dict[i.split('_')[-2]] for i in self.data]

        if len(self.data) != len(self.positions):
            raise ValueError("Data and condition lists must have the same length.")
        if "composition" in subset and len(self.data) != len(self.colors):
            raise ValueError("Data and condition lists must have the same length.")
        if "resolution" not in presaved_path:
            presaved_path = os.path.join(presaved_path, "resolution_128")
        self.presaved_path = os.path.join(presaved_path, _dataset_folder_name(path))

        self.subset = subset
        self.resolution = resolution
        self.ood_filter = ood_filter
        self.id_filter = id_filter

    def get_angle_region(self, angle, angle_dict):
        """
        Determine which position region an angle falls into with boundary handling
        """
        for position, (start, end) in angle_dict.items():
            # Handle overlapping boundaries - use <= for start, < for end (except last region)
            if position == "position10":  # Last region includes end boundary
                if start <= angle <= end:
                    return position
            else:
                if start <= angle < end:
                    return position
        return None

    def make_ood_filter_list(self, subset, offset=0):
        # Extract the number from 'oodN'
        match = re.search(r"composition(\d+)", subset)
        if not match:
            raise ValueError(f"Unknown composition subset: {subset}")
        
        N = int(match.group(1))
        
        # This function now returns the diagonal patterns that should be excluded
        # Each diagonal represents a color-angle combination pattern
        excluded_diagonals = []
        
        angle_keys = list(self.position_cond_dict.keys())
        
        # Generate N diagonals to exclude
        for diagonal in range(N):
            diagonal_conditions = []
            for position_idx in range(self.num_positions):
                color_idx = (diagonal + position_idx + offset) % self.num_colors
                color_val = str(self.color_dict[self.color_keys[color_idx]])

                # Store the condition as a tuple: (color_value, angle_range)
                diagonal_conditions.append((color_val, angle_keys[position_idx]))
            
            excluded_diagonals.append(diagonal_conditions)
        
        return excluded_diagonals


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        position = self.positions[idx]
        if "composition" in self.subset:
            color = self.colors[idx]
            condition = (color, position)
        else:
            condition = (position)
        return _sample_with_latent_cache(self, idx, condition)


class ComfortBallDataset_attribute(Dataset):
    def __init__(self, path, subset, resolution=512, interpolation=Image.BICUBIC, center_crop=False, random_flip=False, use_text_encoder=False, presaved_path = None,id_condition=False, ood_filter = False, id_filter=False):
        self.onehot = True if "onehot" in subset else False
        self.transform = T.Compose(            
        [
            T.Resize(resolution, interpolation=interpolation),  # Use dynamic interpolation method
            T.RandomHorizontalFlip() if random_flip else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

        skewed = False
        offset = 0
        self.num_colors = 10

        self.color_dict = {"RED": (1, 0, 0, 1),
                    "GREEN": (0, 1, 0, 1),
                    "BLUE": (0, 0, 1, 1),
                    "YELLOW": (1, 1, 0, 1),
                    "PURPLE": (1, 0, 1, 1),
                    "ORANGE": (1, 0.5, 0, 1),
                    "CYAN": (0, 1, 1, 1),
                    "GRAY": (0.5, 0.5, 0.5, 1),
                    "WHITE": (1, 1, 1, 1),
                    "BLACK": (0, 0, 0, 1)}

        self.color_keys = list(self.color_dict.keys())
        self.num_colors = len(self.color_dict.keys())
        color_condition = torch.eye(self.num_colors)
        color_dict_reverse = {str(v): k for k, v in self.color_dict.items()}

        color_cond_dict = {str(self.color_dict[i]): color_condition[j] for j, i in enumerate(self.color_dict.keys())}
        
        if "attribute" in subset:
            self.classifier_color_cond_dict = {}
            for j, color1 in enumerate(self.color_dict.keys()):
                for k, color2 in enumerate(self.color_dict.keys()):
                    self.classifier_color_cond_dict[(color1, color2)] = j*self.num_colors + k

        if "composition" in subset:
            filter_list = self.make_ood_filter_list(subset)
            
            print(filter_list)
            
            data = glob.glob(os.path.join(path,'*.png'))

            if not ood_filter:
                self.data = [j for j in data 
                                if ("composite" not in j and

                            not any(
                            any(
                                j.split('/')[-1].split('_')[0] == color_val and 
                                j.split('/')[-1].split('_')[1] == color_val2
                                for color_val, color_val2 in diagonal
                            )
                            for diagonal in filter_list
                        )
                    )
                        ]
            else:
                self.data = [j for j in data 
                                if ("composite" not in j and

                            any(
                            any(
                                j.split('/')[-1].split('_')[0] == color_val and 
                                j.split('/')[-1].split('_')[1] == color_val2
                                for color_val, color_val2 in diagonal
                            )
                            for diagonal in filter_list
                        )
                    )
                        ]
                size = 50
            
        else:
            
            self.data = [i for i in glob.glob(os.path.join(path,'*.png')) if "composite" not in i]
        
        size = int(subset.split("_")[-1]) if "_" in subset else None

        if size == 2000:
            if "train" in subset:
                distribution = [45, 36, 29, 23, 18, 14, 12, 9, 7, 7]  # Sum = 200
            elif "reverse" in subset:
                distribution = [45, 36, 29, 23, 18, 14, 12, 9, 7, 7]  # Sum = 200
                distribution.reverse()
            else:
                distribution = [20] * 10
        elif size == 100000:
            if "train" in subset:
                distribution = [2255, 1795, 1435, 1145, 915, 730, 585, 465, 375, 300]
            elif "reverse" in subset:
                distribution = [2255, 1795, 1435, 1145, 915, 730, 585, 465, 375, 300]
                distribution.reverse()
            else:
                distribution = [1000] * 10
        elif size == 20000:
            if "train" in subset:
                distribution = [451,359,287,229,183,146,117,93,75,60]
            elif "reverse" in subset:
                distribution = [451,359,287,229,183,146,117,93,75,60]
                distribution.reverse()
            else:
                distribution = [200] * 10
        elif size == 10000:
                if "train" in subset:
                    distribution = [225, 179, 143, 114, 91, 73, 58, 46, 37, 34]
                elif "reverse" in subset:
                    distribution = [225, 179, 143, 114, 91, 73, 58, 46, 37, 34]
                    distribution.reverse()
                else:
                    distribution = [100] * 10
        elif size == 50000:
            if "train" in subset:
                distribution = [1128, 898, 718, 572, 457, 365, 292, 232, 187, 151]
            elif "reverse" in subset:
                distribution = [1128, 898, 718, 572, 457, 365, 292, 232, 187, 151]
                distribution.reverse()
            else:
                distribution = [500] * 10

        if "composition" in subset and not ood_filter and not id_filter:
            N = int(re.search(r"composition(\d+)", subset).group(1))
            if N == 1:
                distribution = [d * 10//9 for d in distribution]
            elif N == 3:
                distribution = [d * 10//7 for d in distribution]
            elif N == 5:
                distribution = [d * 10//5 for d in distribution]
            elif N == 8:
                distribution = [d * 10//2 for d in distribution]

        # ./mosaic/data/comfort_ball_attribute/behind/distractor/(0, 0, 0, 1)_(1, 1, 0, 1)_586.png
        # For attribute datasets, filter based on the first color's distribution
        # For ood_filter=True, we want to limit to 50 samples per color combination
        if ood_filter or id_filter:
            # Group by color combination and take first 50 of each
            self.data = [f for f in self.data if int(f.split('_')[-1][:-4]) < 50]
        else:
            # Normal filtering based on distribution and first color
            self.data = [f for f in self.data if int(f.split('_')[-1][:-4]) < distribution[list(color_dict_reverse.keys()).index((f.split('/')[-1].split('_')[0]))]]

        self.conditions  = [(color_cond_dict[(i.split('/')[-1].split('_')[0])], color_cond_dict[(i.split('/')[-1].split('_')[1])]) for i in self.data]

        if len(self.data) != len(self.conditions):
            raise ValueError("Data and condition lists must have the same length.")
        
        if "resolution" not in presaved_path:
            
            self.presaved_path = os.path.join(presaved_path, "resolution_128", "comfort_ball_attribute")
        else:
            self.presaved_path = os.path.join(presaved_path, "comfort_ball_attribute")

        if "multi" in subset:
            self.presaved_path = self.presaved_path + "_multi"
        elif "disturber" in subset:
            self.presaved_path  = self.presaved_path + "_disturber"

        self.subset = subset
        self.resolution = resolution
        self.ood_filter = ood_filter
        self.id_filter = id_filter

    def make_ood_filter_list(self, subset, offset=0):
        # Extract the number from 'oodN'
        match = re.search(r"composition(\d+)", subset)
        if not match:
            raise ValueError(f"Unknown composition subset: {subset}")
        
        N = int(match.group(1))
        
        # This function now returns the diagonal patterns that should be excluded
        # Each diagonal represents a color pair combination pattern
        excluded_diagonals = []
        
        # For attributes, we need to create diagonal patterns across the 100 color combinations
        # Each diagonal has 10 color pairs (one for each "step" in the diagonal)
        
        # Generate N diagonals to exclude
        for diagonal in range(N):
            diagonal_conditions = []
            for step in range(self.num_colors):  # 10 steps per diagonal
                # First color follows the step pattern
                color1_idx = (step + offset) % self.num_colors
                color1_val = str(self.color_dict[self.color_keys[color1_idx]])
                color2_idx = (step + diagonal + offset) % self.num_colors
                color2_val = str(self.color_dict[self.color_keys[color2_idx]])
                # Store the condition as a tuple: (color1_value, color2_value)
                diagonal_conditions.append((color1_val, color2_val))

            excluded_diagonals.append(diagonal_conditions)
        
        return excluded_diagonals


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return _sample_with_latent_cache(self, idx, self.conditions[idx])


class TestDataset(Dataset):
    def __init__(self, path=None, original_type=None, resolution=128, type="generated_dataset", data_name=False, checkpoint_filter=None, color_filter=False, all_filter=False, ood_filter=False, id_filter=False, joint_test=False, interpolation=Image.BICUBIC, center_crop=False, random_flip=False, use_text_encoder=False, subset=False):
        self.color_filter = color_filter
        self.all_filter = all_filter
        self.joint_test = joint_test

        BICYCLE_MOUNTAIN = "bicycle_mountain"
        CAR_SEDAN = "car_sedan"
        COUCH = "Sofa"
        BASKETBALL = "Basketball"
        CHAIR = "Chair"
        DOG = "Dog"
        BED = "Bed"
        DUCK = "Duck"
        LAPTOP = "Laptop"
        HORSE_R = "HorseR"
        BENCH = "Bench"
        SOPHIA = "Sophia" # addressee
        HORSE_L = "HorseL"
        self.object_lists = [BICYCLE_MOUNTAIN, COUCH, CHAIR, DOG, BED, LAPTOP, BENCH, SOPHIA, BASKETBALL,HORSE_R]

        if original_type is None or "attribute" in original_type or (path is not None and "attribute" in path):
            if  self.color_filter or (isinstance(path, str) and ("composition" in path or "ten" in path or "attribute" in path)) or (original_type is not None and "attribute" in original_type):
                self.color_dict = {"RED": (1, 0, 0, 1),
                            "GREEN": (0, 1, 0, 1),
                            "BLUE": (0, 0, 1, 1),
                            "YELLOW": (1, 1, 0, 1),
                            "PURPLE": (1, 0, 1, 1),
                            "ORANGE": (1, 0.5, 0, 1),
                            "CYAN": (0, 1, 1, 1),
                            "GRAY": (0.5, 0.5, 0.5, 1),
                            "WHITE": (1, 1, 1, 1),
                            "BLACK": (0, 0, 0, 1)}
                reverse_color_dict = {str(v): k for k, v in self.color_dict.items()}

                if (path is not None and not "attribute" in path) or (original_type is not None and "attribute" not in original_type):
                    self.color_condition = {k: torch.eye(len(self.color_dict.keys()))[i] for i, (k, v) in enumerate(self.color_dict.items())}
                else:
                    self.color_condition = {}
                    idx = 0
                    for key1 in self.color_dict.keys():
                        for key2 in self.color_dict.keys():
                            self.color_condition[(key1, key2)] = torch.eye(len(self.color_dict.keys())**2)[idx]
                            idx += 1

        if type == "generated_dataset":
            if "attribute" in path:
                self.attribute = True
            else:
                self.attribute = False

            if checkpoint_filter is None:
                if ood_filter:
                    # Check if path directly contains PNG files (timestep directory case)
                    png_files = [f for f in os.listdir(path) if f.endswith('.png')]
                    if png_files:
                        # Path already points to a timestep directory with images - no filtering
                        self.data = [os.path.join(path, f) for f in png_files]
                    else:
                        # Path contains checkpoint directories - apply filtering
                        filter_list = self.make_ood_filter_list(path)
                        self.data = []
                        for checkpoint_dir in glob.glob(os.path.join(path, '*')):
                            if not os.path.isdir(checkpoint_dir):
                                continue
                            # Check if this checkpoint has timestep subdirectories
                            subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                            if subdirs:
                                # New structure: checkpoint/timestep/image.png
                                for timestep_dir in subdirs:
                                    timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                                    self.data.extend([os.path.join(timestep_path, j) for j in os.listdir(timestep_path) 
                                                    if j.endswith('.png') and any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                            else:
                                # Old structure: checkpoint/image.png
                                self.data.extend([os.path.join(checkpoint_dir, j) for j in os.listdir(checkpoint_dir) 
                                                if j.endswith('.png') and any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])

                elif id_filter:
                    # Check if path directly contains PNG files (timestep directory case)
                    png_files = [f for f in os.listdir(path) if f.endswith('.png')]
                    if png_files:
                        # Path already points to a timestep directory with images - no filtering
                        self.data = [os.path.join(path, f) for f in png_files]
                    else:
                        # Path contains checkpoint directories - apply filtering
                        filter_list = self.make_ood_filter_list(path)
                        self.data = []
                        for checkpoint_dir in glob.glob(os.path.join(path, '*')):
                            if not os.path.isdir(checkpoint_dir):
                                continue
                            # Check if this checkpoint has timestep subdirectories
                            subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                            if subdirs:
                                # New structure: checkpoint/timestep/image.png
                                for timestep_dir in subdirs:
                                    timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                                    self.data.extend([os.path.join(timestep_path, j) for j in os.listdir(timestep_path) 
                                                    if j.endswith('.png') and not any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                            else:
                                # Old structure: checkpoint/image.png
                                self.data.extend([os.path.join(checkpoint_dir, j) for j in os.listdir(checkpoint_dir) 
                                                if j.endswith('.png') and not any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                else:
                    # Handle both old structure (checkpoint/image.png) and new structure (checkpoint/timestep/image.png)
                    self.data = []
                    
                    # Check if path directly contains PNG files (timestep directory case)
                    png_files = [f for f in os.listdir(path) if f.endswith('.png')]
                    if png_files:
                        # Path already points to a timestep directory with images
                        self.data = [os.path.join(path, f) for f in png_files]
                    else:
                        # Path contains checkpoint directories
                        for checkpoint_dir in glob.glob(os.path.join(path, '*')):
                            if not os.path.isdir(checkpoint_dir):
                                continue
                            # Check if this checkpoint has timestep subdirectories
                            subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                            if subdirs:
                                # New structure: checkpoint/timestep/image.png
                                for timestep_dir in subdirs:
                                    timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                                    self.data.extend([os.path.join(timestep_path, j) for j in os.listdir(timestep_path) if j.endswith('.png')])
                            else:
                                # Old structure: checkpoint/image.png
                                self.data.extend([os.path.join(checkpoint_dir, j) for j in os.listdir(checkpoint_dir) if j.endswith('.png')])

                # Extract checkpoint - handle both old and new structure
                self.checkpoints = []
                for i in self.data:
                    # Check if path has timestep folder (new structure)
                    # New: .../checkpoint/timestep/image.png -> split('/')[-3] is checkpoint
                    # Old: .../checkpoint/image.png -> split('/')[-2] is checkpoint
                    parent_dir = i.split('/')[-2]
                    grandparent_dir = i.split('/')[-3]
                    
                    # If parent is a number (timestep), use grandparent as checkpoint
                    # Otherwise use parent as checkpoint
                    try:
                        int(parent_dir)  # If this is a timestep (number)
                        checkpoint = grandparent_dir
                    except (ValueError, AttributeError):
                        checkpoint = parent_dir
                    
                    if "best_model" in checkpoint:
                        self.checkpoints.append("best_model")
                    else:
                        # Extract number from checkpoint-XXXXX format
                        try:
                            self.checkpoints.append(int(checkpoint.split('-')[-1]))
                        except:
                            self.checkpoints.append(checkpoint)
                
                if "attribute" in path:
                    self.condition = [self.extract_prompt_key(i) for i in self.data]
                if "object" in path:
                    condition = [self.extract_object_key(i) for i in self.data]
                    self.object_condition = {}
                    idx = 0
                    for key1 in self.object_lists:
                        for key2 in self.object_lists:
                            self.object_condition[(key1, key2)] = torch.eye(len(self.object_lists)**2)[idx]
                            idx += 1

                    self.condition = [self.object_condition[(i[0], i[1])] for i in condition]

            elif checkpoint_filter == -1:
                checkpoints = [int(i.split('-')[-1]) for i in os.listdir(path) if i.startswith('checkpoint')]
                max_checkpoint = max(checkpoints)
                
                if ood_filter:
                    filter_list = self.make_ood_filter_list(path)
                    # Handle both old and new structure
                    self.data = []
                    checkpoint_dir = os.path.join(path, f'checkpoint-{max_checkpoint}')
                    if os.path.exists(checkpoint_dir):
                        subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                        if subdirs:
                            # New structure with timesteps
                            for timestep_dir in subdirs:
                                timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                                self.data.extend([os.path.join(timestep_path, j) for j in os.listdir(timestep_path) 
                                                if j.endswith('.png') and any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                        else:
                            # Old structure without timesteps
                            self.data.extend([os.path.join(checkpoint_dir, j) for j in os.listdir(checkpoint_dir) 
                                            if j.endswith('.png') and any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                elif id_filter:
                    filter_list = self.make_ood_filter_list(path)
                    # Handle both old and new structure
                    self.data = []
                    checkpoint_dir = os.path.join(path, f'checkpoint-{max_checkpoint}')
                    if os.path.exists(checkpoint_dir):
                        subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                        if subdirs:
                            # New structure with timesteps
                            for timestep_dir in subdirs:
                                timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                                self.data.extend([os.path.join(timestep_path, j) for j in os.listdir(timestep_path) 
                                                if j.endswith('.png') and not any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                        else:
                            # Old structure without timesteps
                            self.data.extend([os.path.join(checkpoint_dir, j) for j in os.listdir(checkpoint_dir) 
                                            if j.endswith('.png') and not any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                else: 
                    self.data = glob.glob(os.path.join(path, f'checkpoint-{max_checkpoint}', '*.png'))
                    # If no direct .png files, check for timestep subdirectories
                    if len(self.data) == 0:
                        checkpoint_dir = os.path.join(path, f'checkpoint-{max_checkpoint}')
                        subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                        for timestep_dir in subdirs:
                            timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                            self.data.extend(glob.glob(os.path.join(timestep_path, '*.png')))
                
                self.checkpoints = [max_checkpoint] * len(self.data)
                self.condition = [self.extract_prompt_key(i) for i in self.data]

            else: # checkpoint filter
                if ood_filter:
                    filter_list = self.make_ood_filter_list(path)
                    # Handle both old and new structure
                    self.data = []
                    if checkpoint_filter != "best_model":
                        checkpoint_dir = os.path.join(path, f'checkpoint-{checkpoint_filter}')
                    else:
                        checkpoint_dir = os.path.join(path, 'best_model')
                    
                    if os.path.exists(checkpoint_dir):
                        subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                        if subdirs:
                            # New structure with timesteps
                            for timestep_dir in subdirs:
                                timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                                self.data.extend([os.path.join(timestep_path, j) for j in os.listdir(timestep_path) 
                                                if j.endswith('.png') and any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                        else:
                            # Old structure without timesteps
                            self.data.extend([os.path.join(checkpoint_dir, j) for j in os.listdir(checkpoint_dir) 
                                            if j.endswith('.png') and any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                    
                elif id_filter:
                    filter_list = self.make_ood_filter_list(path)
                    # Handle both old and new structure
                    self.data = []
                    if checkpoint_filter != "best_model":
                        checkpoint_dir = os.path.join(path, f'checkpoint-{checkpoint_filter}')
                    else:
                        checkpoint_dir = os.path.join(path, 'best_model')
                    
                    if os.path.exists(checkpoint_dir):
                        subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                        if subdirs:
                            # New structure with timesteps
                            for timestep_dir in subdirs:
                                timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                                self.data.extend([os.path.join(timestep_path, j) for j in os.listdir(timestep_path) 
                                                if j.endswith('.png') and not any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                        else:
                            # Old structure without timesteps
                            self.data.extend([os.path.join(checkpoint_dir, j) for j in os.listdir(checkpoint_dir) 
                                            if j.endswith('.png') and not any(fnmatch.fnmatch(os.path.basename(j)[:-4], pat) for pat in filter_list)])
                else: 
                    if checkpoint_filter != "best_model":
                        self.data = glob.glob(os.path.join(path, f'checkpoint-{checkpoint_filter}', '*.png'))
                        # If no direct .png files, check for timestep subdirectories
                        if len(self.data) == 0:
                            checkpoint_dir = os.path.join(path, f'checkpoint-{checkpoint_filter}')
                            subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                            for timestep_dir in subdirs:
                                timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                                self.data.extend(glob.glob(os.path.join(timestep_path, '*.png')))
                    else:
                        self.data = glob.glob(os.path.join(path, 'best_model', '*.png'))
                        # If no direct .png files, check for timestep subdirectories
                        if len(self.data) == 0:
                            checkpoint_dir = os.path.join(path, 'best_model')
                            subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                            for timestep_dir in subdirs:
                                timestep_path = os.path.join(checkpoint_dir, timestep_dir)
                                self.data.extend(glob.glob(os.path.join(timestep_path, '*.png')))
                
                self.checkpoints = [checkpoint_filter] * len(self.data)
                self.condition = [self.extract_prompt_key(i) for i in self.data]

        elif type == "original_dataset":
            
            if original_type is not None:
                self.data = get_dataset(original_type).data

            else:
                self.data = glob.glob(os.path.join(path, '*.png'))

            self.checkpoints = None


            if path is not None and "attribute" in path: # vae in attribute dataset
                if subset:
                    self.data = [i for i in self.data if int(i.split('_')[-1][:-4]) < 50]
                if self.color_filter:
                    self.condition = [1 for i in self.data]
                elif self.all_filter:
                    self.condition = [(reverse_color_dict[i.split("/")[-1].split('_')[0]], reverse_color_dict[i.split("/")[-1].split('_')[1]]) for i in self.data]
                else:
                    # Return class index (0-99) instead of one-hot for attribute
                    self.condition = []
                    for i in self.data:
                        color1 = reverse_color_dict[i.split("/")[-1].split('_')[0]]
                        color2 = reverse_color_dict[i.split("/")[-1].split('_')[1]]
                        # Find the index in color_condition dictionary
                        class_idx = list(self.color_condition.keys()).index((color1, color2))
                        self.condition.append(class_idx)
            
            elif path is not None and "position" in path and "count" not in path:

                if subset:
                    self.data = [i for i in self.data if int(i.split('_')[-1][:-4]) < 50]
                if self.color_filter:
                    self.condition = [self.color_condition[reverse_color_dict[i.split("/")[-1].split('_')[1]]] for i in self.data]
                elif self.all_filter:
                    raise NotImplementedError("all_filter is not implemented for position dataset")
                else:
                    if original_type is not None and "position" in original_type:
                        self.condition = [int(i.split('_')[-2].replace("position","")) for i in self.data]
                    elif original_type is not None and "attribute" in original_type:
                        self.condition = [self.color_condition[(reverse_color_dict[i.split('/')[-1].split('_')[0]], reverse_color_dict[i.split('/')[-1].split('_')[1]])] for i in self.data]
            
                    else:
                        if not self.color_filter:
                            self.condition = [int(i.split('_')[-2].replace("position","")) for i in self.data]
                        else:
                            self.condition = [self.color_condition[reverse_color_dict[i.split("/")[-1].split('_')[0]]] for i in self.data]

            elif path is not None and "object" in path:
                if subset:
                    self.data = [i for i in self.data if int(i.split('_')[-1][:-4]) < 50]
                else:
                    # Return class index (0-99) instead of one-hot for attribute
                    self.condition = []
                    object_tuple  = [(i,j) for i in self.object_lists for j in self.object_lists]
                    
                    self.object_classifier_dict = {(obj1,obj2): torch.eye(len(self.object_lists)**2)[idx] for idx, (obj1, obj2) in enumerate(object_tuple)}

                    for i in self.data:
                        obj1, obj2 = self.extract_object_key(i)
                        self.condition.append(self.object_classifier_dict[(obj1, obj2)])

            else:
                if subset:
                    try:
                        self.data = [i for i in self.data if int(i.split('_')[-2]) < 50]
                    except:
                        self.data = [i for i in self.data if int(i.split('/')[-1].split('_')[-2]) < 50]
                if self.color_filter:
                    self.condition = [self.color_condition[reverse_color_dict[i.split("/")[-1].split('_')[0]]] for i in self.data]
                elif self.all_filter:
                    self.condition = [(reverse_color_dict[i.split("/")[-1].split('_')[0]], int(i.split("_")[-1][:-4])) for i in self.data]
                else:
                    self.condition = [int(i.split('_')[-1][:-4]) for i in self.data]
        

        # Ensure self.condition is always set
        if not hasattr(self, 'condition'):
            if "object" not in path:
                self.condition = [self.extract_prompt_key(i) for i in self.data]
            else:
                
                
                self.condition = [self.extract_object_key(i) for i in self.data]
        
        # Ensure self.checkpoints is always set
        if not hasattr(self, 'checkpoints'):
            self.checkpoints = None
        
        self.resolution = resolution
        self.data_name = data_name
        self.type = type
    
    def make_ood_filter_list(self, subset):
        # Extract the number from 'oodN'
        import re
        match = re.search(r"composition(\d+)", subset)
        offset_match = re.search(r"_offset(\d+)", subset)
        offset = 0
        if offset_match: offset = int(offset_match.group(1))
        
        N = int(match.group(1))
        filter_list = []
        
        if "object" in subset:
            for diagonal in range(N):
                for step in range(10):  # 10 steps per diagonal
                    color1_idx = (step + offset) % len(list(self.object_lists))
                    color2_idx = (step + diagonal + offset) % len(list(self.object_lists))
                    filter_list.append(f"*_{list(self.object_lists)[color1_idx]}_{list(self.object_lists)[color2_idx]}_*")
            
            expected_length = 10 * N
            if len(filter_list) != expected_length:
                raise ValueError(f"Filter list length mismatch for {subset}: expected {expected_length}, got {len(filter_list)}")
            
        elif "attribute" not in subset: 
            for diagonal in range(N):
                for count in range(1, 11):  # 1 to 10
                    color_idx = (diagonal + count - 1 + offset) % len(list(self.color_dict.keys()))  # Wrap around colors
                    filter_list.append(f"*_{list(self.color_dict.keys())[color_idx]}_{count}_*")
            
            expected_length = 10 * N
            if len(filter_list) != expected_length:
                raise ValueError(f"Filter list length mismatch for {subset}: expected {expected_length}, got {len(filter_list)}")

            if "ten" in subset:
                # exclude "BLACK_*"
                filter_list = [f for f in filter_list if "BLACK" not in f]
        else:
            for diagonal in range(N):
                for step in range(10):  # 10 steps per diagonal
                    color1_idx = (step + offset) % len(list(self.color_dict.keys()))
                    color2_idx = (step + diagonal + offset) % len(list(self.color_dict.keys()))
                    filter_list.append(f"*_{list(self.color_dict.keys())[color1_idx]}_{list(self.color_dict.keys())[color2_idx]}_*")
            
            expected_length = 10 * N
            if len(filter_list) != expected_length:
                raise ValueError(f"Filter list length mismatch for {subset}: expected {expected_length}, got {len(filter_list)}")
        return filter_list

    def extract_prompt_key(self, filepath):
        filename = filepath.split('/')[-1]  # Get just the filename
        # Split by '_' and find the part after 'key'
        parts = filename.split('_')

        key_index = parts.index('key') + 1

        if self.attribute:
            return self.color_condition[(parts[key_index], parts[key_index+1])]

        if self.color_filter:
            return self.color_condition[parts[key_index]]
            
        if self.all_filter or self.joint_test:
            return (parts[key_index], int(parts[key_index+1]))

        try:
            return int(parts[key_index])
        except:
            return  int(parts[key_index + 1])

    def extract_object_key(self, filename):
        basename = filename.split('/')[-1].replace("prompt_key_","").replace("sample_","").replace('.png', '')
        import re
        basename = re.sub(r'_\d+$', '', basename)
        
        sorted_objects = sorted(self.object_lists, key=lambda x: len(x), reverse=True)
        
        found_objects = []
        remaining = basename
        
        for obj in sorted_objects:
            if remaining.startswith(obj):
                found_objects.append(obj)
                remaining = remaining[len(obj):].lstrip('_')
                break
        
        for obj in sorted_objects:
            if remaining.startswith(obj) or remaining == obj:
                found_objects.append(obj)
                break
        
        if len(found_objects) == 2:
            return tuple(found_objects)
        else:
            raise ValueError(f"Could not parse two objects from filename: {filename}, found: {found_objects}")
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.data[idx])
        except:
            print(f"Error loading image: {self.data[idx]}")
            raise ValueError(f"Error loading image: {self.data[idx]}")
        img = img.convert("RGB")
        img = T.Resize((self.resolution, self.resolution))(img)
        img = T.ToTensor()(img)
        condition = self.condition[idx]
        if isinstance(condition, tuple) and self.all_filter:
            color, counting = condition
            condition = {color_key: 0 for color_key in self.color_dict.keys()}
            condition[color] = counting
        elif isinstance(condition, tuple) and self.joint_test:
            color, counting = condition
            condition = {list(self.color_dict.keys()).index(color): counting}

        if self.checkpoints is not None and not self.data_name:
            checkpoint_tensor = self.checkpoints[idx]
            return {'image': img, 'input_ids': condition,'checkpoint': checkpoint_tensor}
        
        elif self.checkpoints is not None and self.data_name:

            checkpoint_tensor = self.checkpoints[idx]
            return {'image': img, 'input_ids': condition, 'data_name': self.data[idx], 'checkpoint': checkpoint_tensor}
        
        elif self.checkpoints is None and self.data_name:
            return {'image': img, 'input_ids': condition, 'data_name': self.data[idx]}
        
        else:
            return {'image': img, 'input_ids': condition}
