import glob
import os
import re
from collections import Counter

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from utils import is_complex, is_composition, task_of

COLOR_DICT = {
    "RED": (1, 0, 0, 1),
    "GREEN": (0, 1, 0, 1),
    "BLUE": (0, 0, 1, 1),
    "YELLOW": (1, 1, 0, 1),
    "PURPLE": (1, 0, 1, 1),
    "ORANGE": (1, 0.5, 0, 1),
    "CYAN": (0, 1, 1, 1),
    "GRAY": (0.5, 0.5, 0.5, 1),
    "WHITE": (1, 1, 1, 1),
    "BLACK": (0, 0, 0, 1),
}

DATA_ROOT = "./mosaic/data"

# After dropping n held-out diagonals (out of 10) the per-class counts are scaled up so the dataset keeps its size.
_HELD_OUT_SCALE = {1: (10, 9), 3: (10, 7), 5: (10, 5), 8: (5, 1)}


def _scale_for_held_out(distribution, n_held_out):
    num, den = _HELD_OUT_SCALE.get(n_held_out, (1, 1))
    return [d * num // den for d in distribution]


def _held_out_count(subset):
    match = re.search(r"composition(\d+)", subset)
    if not match:
        raise ValueError(f"Unknown composition subset: {subset}")
    return int(match.group(1))


def held_out_pairs(test_mode):
    """Held-out (OOD) label pairs of a composition<N> test_mode, by colour NAME.

    position/count composition: {(colour, k)} with k = position/count 1..10 and colour index (d + k - 1) % 10
    attribute composition:      {(colour1, colour2)} with colour2 index (step + d) % 10
    for d in range(N). Returns an empty set for non-composition modes.
    """
    if not is_composition(test_mode):
        return set()
    n = _held_out_count(test_mode)
    names = list(COLOR_DICT.keys())
    pairs = set()
    for d in range(n):
        for step in range(10):
            if task_of(test_mode) == "attribute":
                pairs.add((names[step], names[(step + d) % 10]))
            else:
                pairs.add((names[(d + step) % 10], step + 1))
    return pairs


def _subset_size(subset):
    if "_" not in subset:
        raise ValueError(f"Provide the dataset size as <subset>_<size>, got {subset}")
    return int(subset.split("_")[-1])


def uniform_distribution(size, divisor=1):
    """Per-class sample counts (10 classes) when `size // divisor` images are spread evenly."""
    return [size // divisor // 10] * 10


def _make_transform(resolution, interpolation):
    return T.Compose([
        T.Resize(resolution, interpolation=interpolation),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])


def _latent_cache_dir(presaved_path, resolution, data_path):
    """<presaved_path>/resolution_<res>/<dataset folder name>. Every dataset folder gets its own cache
    because e.g. comfort_ball_attribution and comfort_ball_attribute_composition_subset reuse image names."""
    if "resolution" not in presaved_path:
        presaved_path = os.path.join(presaved_path, f"resolution_{resolution}")
    return os.path.join(presaved_path, os.path.basename(os.path.normpath(data_path)))


def _sample_with_latent_cache(dataset, idx, condition):
    """Build one training sample. Images are always on disk; the VAE latent is used
    from `<presaved_path>/<image name>.pt` when it exists, otherwise the image is returned
    together with `latent_path` so train.py can create the cache file."""
    image_path = dataset.data[idx]
    sample = {"input_ids": condition}

    latent_path = os.path.join(dataset.presaved_path, os.path.basename(image_path).replace(".png", ".pt"))
    if os.path.exists(latent_path):
        try:
            sample["presaved_latents"] = torch.load(latent_path)
            return sample
        except Exception:
            pass  # corrupted cache file: fall back to the image and let train.py rewrite it

    img = Image.open(image_path).convert("RGB")
    sample["image"] = dataset.transform(img)
    sample["latent_path"] = latent_path
    return sample


def get_dataset(test_mode, resolution=128, interpolation=Image.BICUBIC, presaved_path="./presaved_latents"):
    task = task_of(test_mode)
    composition, complex_ = is_composition(test_mode), is_complex(test_mode)
    kwargs = dict(subset=test_mode, resolution=resolution, interpolation=interpolation, presaved_path=presaved_path)

    if task == "attribute":
        if composition:
            path = f"{DATA_ROOT}/comfort_ball_attribute_composition_subset" # 100k (Full should be 1M, but we only have 100k images in the dataset)
        elif complex_:
            path = f"{DATA_ROOT}/comfort_ball_attribute_complex"
        else:
            path = f"{DATA_ROOT}/comfort_ball_attribution"
        return ComfortBallDataset_attribute(path, **kwargs)

    if task == "position":
        if composition:
            path = f"{DATA_ROOT}/comfort_ball_position_composition_subset" # 100k (Full should be 1M, but we only have 100k images in the dataset)
        elif complex_:
            path = f"{DATA_ROOT}/comfort_ball_position_complex"
        else:
            path = f"{DATA_ROOT}/comfort_ball_position"
        return ComfortBallDataset_position(path, **kwargs)

    # count
    if task == "count":
        if composition:
            path = f"{DATA_ROOT}/comfort_ball_count_composition_subset" # 100k (Full should be 1M, but we only have 100k images in the dataset)
        elif complex_:
            raise ValueError(f"Complex counting dataset does not exist; got {test_mode!r}.")
        else:
            path = f"{DATA_ROOT}/comfort_ball_count"
        return ComfortBallDataset_counting(path, **kwargs)

    raise ValueError(f"Unknown dataset for test mode {test_mode!r}.")


class ComfortBallDataset_counting(Dataset):
    """Images named `<color tuple>_<index>_<count>.png`.
    Condition: one-hot count (1..10), plus one-hot object colour for `composition` subsets."""

    def __init__(self, path, subset, resolution=128, interpolation=Image.BICUBIC, presaved_path=None):
        self.transform = _make_transform(resolution, interpolation)
        self.subset = subset
        self.resolution = resolution
        self.presaved_path = _latent_cache_dir(presaved_path, resolution, path)
        self.composition = is_composition(subset)

        self.num_counts = 10
        self.color_dict = dict(COLOR_DICT)
        self.color_keys = list(self.color_dict.keys())
        self.num_colors = len(self.color_dict)
        color_cond_dict = {str(v): torch.eye(self.num_colors)[j] for j, v in enumerate(self.color_dict.values())}
        counting_condition = torch.eye(self.num_counts)

        def color_of(f): return os.path.basename(f).split("_")[0]
        def count_of(f): return int(os.path.basename(f).split("_")[-1][:-4])
        def index_of(f): return int(os.path.basename(f).split("_")[-2])

        files = [f for f in glob.glob(os.path.join(path, "*.png")) if count_of(f) <= self.num_counts]
        size = _subset_size(subset)
        if self.composition:
            # drop the held-out (colour, count) diagonals
            held_out = {pair for diagonal in self.make_ood_filter_list(subset) for pair in diagonal}
            print(f"Held-out (colour, count) pairs for {subset}: {sorted(held_out)}")
            files = [f for f in files if (color_of(f), count_of(f)) not in held_out]
            per_count = _scale_for_held_out(uniform_distribution(size, divisor=10), _held_out_count(subset))
        else:
            per_count = uniform_distribution(size)
        self.data = [f for f in files if index_of(f) < per_count[count_of(f) - 1]]
        print(f"len(self.data): {len(self.data)}")

        self.countings = [counting_condition[count_of(f) - 1] for f in self.data]
        if self.composition:
            self.colors = [color_cond_dict[color_of(f)] for f in self.data]
            color_names = {str(v): k for k, v in self.color_dict.items()}
            print("images per count:", dict(Counter(count_of(f) for f in self.data)))
            print("images per colour:", dict(Counter(color_names[color_of(f)] for f in self.data)))

    def make_ood_filter_list(self, subset):
        """Held-out (colour value, count) pairs, see held_out_pairs()."""
        return [[(str(self.color_dict[name]), count) for name, count in held_out_pairs(subset)]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        condition = (self.colors[idx], self.countings[idx]) if self.composition else self.countings[idx]
        return _sample_with_latent_cache(self, idx, condition)


class ComfortBallDataset_position(Dataset):
    """Images named `<bg color>_<object color>_position<K>_<index>.png`.
    Condition: one-hot position, plus one-hot object colour for `composition` subsets."""

    def __init__(self, path, subset, resolution=128, interpolation=Image.BICUBIC, presaved_path=None):
        self.transform = _make_transform(resolution, interpolation)
        self.subset = subset
        self.resolution = resolution
        self.presaved_path = _latent_cache_dir(presaved_path, resolution, path)
        self.composition = is_composition(subset)

        self.num_positions = 10
        self.position_cond_dict = {f"position{i + 1}": torch.eye(self.num_positions)[i] for i in range(self.num_positions)}
        self.color_dict = dict(COLOR_DICT)
        self.color_keys = list(self.color_dict.keys())
        self.num_colors = len(self.color_dict)
        color_cond_dict = {str(v): torch.eye(self.num_colors)[j] for j, v in enumerate(self.color_dict.values())}

        def color_of(f): return os.path.basename(f).split("_")[1]
        def position_of(f): return os.path.basename(f).split("_")[-2]
        def index_of(f): return int(os.path.basename(f).split("_")[-1][:-4])

        files = glob.glob(os.path.join(path, "*.png"))
        size = _subset_size(subset)
        if self.composition:
            # drop the held-out (colour, position) diagonals
            held_out = {pair for diagonal in self.make_ood_filter_list(subset) for pair in diagonal}
            print(f"Held-out (colour, position) pairs for {subset}: {sorted(held_out)}")
            files = [f for f in files if (color_of(f), position_of(f)) not in held_out]
            per_position = _scale_for_held_out(uniform_distribution(size, divisor=10), _held_out_count(subset))
        else:
            per_position = uniform_distribution(size)
        self.data = [f for f in files if index_of(f) < per_position[int(position_of(f).replace("position", "")) - 1]]

        self.positions = [self.position_cond_dict[position_of(f)] for f in self.data]
        if self.composition:
            self.colors = [color_cond_dict[color_of(f)] for f in self.data]
            color_names = {str(v): k for k, v in self.color_dict.items()}
            print("images per position:", dict(Counter(position_of(f) for f in self.data)))
            print("images per colour:", dict(Counter(color_names[color_of(f)] for f in self.data)))

    def make_ood_filter_list(self, subset):
        """Held-out (colour value, positionK) pairs, see held_out_pairs()."""
        return [[(str(self.color_dict[name]), f"position{k}") for name, k in held_out_pairs(subset)]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        condition = (self.colors[idx], self.positions[idx]) if self.composition else self.positions[idx]
        return _sample_with_latent_cache(self, idx, condition)


class ComfortBallDataset_attribute(Dataset):
    """Images named `<color1>_<color2>_<index>.png`. Condition: (one-hot color1, one-hot color2)."""

    def __init__(self, path, subset, resolution=128, interpolation=Image.BICUBIC, presaved_path=None):
        self.transform = _make_transform(resolution, interpolation)
        self.subset = subset
        self.resolution = resolution
        self.presaved_path = _latent_cache_dir(presaved_path, resolution, path)

        self.color_dict = dict(COLOR_DICT)
        self.color_keys = list(self.color_dict.keys())
        self.num_colors = len(self.color_dict)
        color_condition = torch.eye(self.num_colors)
        color_dict_reverse = {str(v): k for k, v in self.color_dict.items()}
        color_cond_dict = {str(self.color_dict[k]): color_condition[j] for j, k in enumerate(self.color_keys)}

        # (COLOR1 name, COLOR2 name) -> class index of the 10x10 attribute classifier
        self.classifier_color_cond_dict = {}
        for j, color1 in enumerate(self.color_keys):
            for k, color2 in enumerate(self.color_keys):
                self.classifier_color_cond_dict[(color1, color2)] = j * self.num_colors + k

        def color1_of(f): return os.path.basename(f).split("_")[0]
        def index_of(f): return int(os.path.basename(f).split("_")[-1][:-4])

        files = glob.glob(os.path.join(path, "*.png"))
        distribution = uniform_distribution(_subset_size(subset), divisor=10)
        if is_composition(subset):
            # drop the held-out (color1, color2) diagonals
            held_out = {pair for diagonal in self.make_ood_filter_list(subset) for pair in diagonal}
            print(f"Held-out (color1, color2) pairs for {subset}: {sorted(held_out)}")
            files = [f for f in files if tuple(os.path.basename(f).split("_")[:2]) not in held_out]
            distribution = _scale_for_held_out(distribution, _held_out_count(subset))

        # keep the first N images per first colour
        self.data = [f for f in files if index_of(f) < distribution[list(color_dict_reverse.keys()).index(color1_of(f))]]
        print(f"len(self.data): {len(self.data)}")

        self.conditions = [(color_cond_dict[os.path.basename(f).split("_")[0]], color_cond_dict[os.path.basename(f).split("_")[1]]) for f in self.data]

    def make_ood_filter_list(self, subset):
        """Held-out (colour1 value, colour2 value) pairs, see held_out_pairs()."""
        return [[(str(self.color_dict[c1]), str(self.color_dict[c2])) for c1, c2 in held_out_pairs(subset)]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return _sample_with_latent_cache(self, idx, self.conditions[idx])
