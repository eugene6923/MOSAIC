# When Do Diffusion Models Learn to Generate Multiple Objects?

> **Yujin Jeong, Arnas Uselis, Iro Laina, Seong Joon Oh, Anna Rohrbach**  
> ICML 2026

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2605.00273)
[![Status](https://img.shields.io/badge/Status-Full%20Pipeline-blue)]()

---

## Overview

This repository contains **MOSAIC** (Multi-Object Spatial relations, AttrIbution, Counting), a controlled diagnostic framework for analyzing multi-object generation failures in text-to-image diffusion models, together with the full pipeline used in the paper:

1. **Dataset generation** with Blender (`mosaic/`)
2. **Training** small conditional diffusion models (UNet or DiT) on one MOSAIC task (`train.py`)
3. **Generation** of images from trained checkpoints (`test_generate.py`)
4. **Evaluation** of generated images with pretrained classifiers (`evaluate.py`)

## Key Contributions

- **MOSAIC Framework**: Isolates three compositional factors — **Color Attribution**, **Counting**, and **Spatial Relations** — enabling causal analysis of data effects.
- **Comprehensive Dataset Generation**: Controlled dataset generation pipeline for multi-object scenarios using Blender.
- **Diagnostic Analysis**: Systematic evaluation of diffusion model failures on compositional tasks.

## Tasks and Variants

Every experiment is identified by a `test_mode` string of the form `<task>[_complex|_composition<N>]_<size>`:

| task | variant | `test_mode` example | dataset folder (`mosaic/data/`) | condition fed to the model |
|---|---|---|---|---|
| count | basic | `count_100000` | `comfort_ball_count` | one-hot count (1–10) |
| count | composition | `count_composition1_100000` | `comfort_ball_count_composition_subset` | (one-hot colour, one-hot count) |
| position | basic | `position_100000` | `comfort_ball_position` | one-hot position (1–10) |
| position | complex | `position_complex_10000` | `comfort_ball_position_complex` | one-hot position |
| position | composition | `position_composition1_10000` | `comfort_ball_position_composition_subset` | (one-hot colour, one-hot position) |
| attribute | basic | `attribute_100000` | `comfort_ball_attribution` | (one-hot colour 1, one-hot colour 2) |
| attribute | complex | `attribute_complex_10000` | `comfort_ball_attribute_complex` | (one-hot colour 1, one-hot colour 2) |
| attribute | composition | `attribute_composition1_10000` | `comfort_ball_attribute_composition_subset` | (one-hot colour 1, one-hot colour 2) |

- **basic**: a single object (count), one object pair (attribute) or two objects (position) on a plain background.
- **complex**: the same task with distractor objects in the scene.
- **composition\<N\>**: a compositional-generalization split. The 10 colours × 10 classes (count / position / second colour) grid is split along its diagonals, and `N` diagonals (10 × N label pairs) are **held out** from training. `evaluate.py` reports results separately for held-out and in-distribution pairs. Supported `N`: 1, 3, 5, 8.
- **size**: number of training images. For composition splits the images per remaining pair are scaled up so the total stays close to `size`:

  | N | remaining pairs | images per pair (size 100000) | total |
  |---|---|---|---|
  | 1 | 90 | 1,111 | 99,990 |
  | 3 | 70 | 1,428 | 99,960 |
  | 5 | 50 | 2,000 | 100,000 |
  | 8 | 20 | 5,000 | 100,000 |

Conditions are always one-hot vectors; a small MLP (`condition_encoder` for one token, `two_condition_encoder` for two tokens, see `model.py`) maps them to the cross-attention input of the UNet or the context tokens of the DiT. There is no complex variant of the count task.

## Repository Structure

```
MOSAIC/
├── README.md                     this file
├── environment.yml               conda env for training / generation / evaluation (Python 3.11, torch, accelerate, ...)
├── requirements.txt              the same pip dependencies
├── train.py, train_args.py       training (accelerate), see train.sh
├── train_data.py                 collate function (condition stacking, classifier-free-guidance dropout)
├── data.py                       MOSAIC datasets, composition splits, VAE latent cache
├── validation_prompts.py         prompts generated during training validation
├── model.py                      condition encoders and the classifiers used for scoring
├── utils.py                      test_mode parsing, classifier loading, validation metrics
├── test_generate.py              generate images from trained checkpoints, see test.sh
├── evaluate.py                   score generated images with the classifiers
├── classifier_weights/           pretrained classifiers (*.pth, see below)
├── diffusers/                    vendored diffusers with the two custom pipelines (MyCustomPipeline, MyCustomPipeline_dit)
└── mosaic/                       dataset generation (Blender), see mosaic/README.md
    ├── data_generation/
    ├── generation_configs/
    ├── environment.yml, requirements.txt, setup_bpy_rpath.sh
    └── data/                     generated / downloaded datasets go here
```

Created at run time (git-ignored): `dit_weights/`, `unet_weights/` (runs), `presaved_latents/` (VAE latent cache), `outputs/` (generated images and evaluations), `wandb/`.

## Setup

```bash
conda env create -f environment.yml      # creates the `mosaic` env
conda activate mosaic
```

The vendored `diffusers/` is used directly from source (`sys.path`), no installation needed.

**Classifier weights.** Validation during training and `evaluate.py` score images with pretrained classifiers that must be placed in `classifier_weights/`:

| file | used for |
|---|---|
| `best_classifier_count.pth` | count (20 classes) |
| `best_classifier_count_color.pth` | object colour in count-composition |
| `best_classifier_position_pretrained.pth` | position (10 classes) |
| `best_classifier_position_complex.pth` | position in complex scenes |
| `best_classifier_position_color_pretrained.pth` | object colour in position-composition |
| `best_classifier_attribute_pretrained.pth` | colour pair (100 classes) |
| `best_classifier_attribute_shape_pretrained.pth` | shape check for attribute |

**Data.** Either generate the datasets with the Blender pipeline (see [mosaic/README.md](mosaic/README.md)) or download the pre-generated images linked there, and place each dataset under `mosaic/data/<dataset folder>` using the folder names from the table above.

## Training

`train.sh` shows the configuration used in the paper (4 GPUs, DiT, 20k steps):

```bash
accelerate launch --num_processes 4 train.py \
  --model dit --test_mode position_composition1_10000 \
  --max_train_steps 20000 --checkpointing_steps 2000 \
  --resolution 128 --learning_rate 1e-4 --train_batch_size 512 --gradient_accumulation_steps 1 \
  --val_num_samples 50 --validation_batch_size 4 --test_accuracy --no_image \
  --seed 42 --resume_from_checkpoint latest --report_to wandb
```

- `--model unet|dit` selects the backbone; `--test_mode` selects task, variant and dataset size (table above).
- Validation runs every `--validation_steps` (default 500): images are generated for every class (× `--val_num_samples`) and scored with the classifiers (`--test_accuracy`). The checkpoint with the best validation accuracy is kept in `best_model/`; `--early_stopping` stops once it stops improving.
- VAE latents are cached on first use in `presaved_latents/resolution_128/<dataset folder>/`, so later epochs and runs skip the VAE.
- Runs are written to `<dit|unet>_weights/seed_<seed>/<test_mode>/<param_string>/` with `checkpoint-<step>/`, `best_model/`, the final pipeline (`model_index.json`, `transformer|unet/`, `condition_encoder/`), `args.txt`, `best_val_accuracy.txt` and `completed.txt`. `--resume_from_checkpoint latest` continues an interrupted run.

Run `python train.py --help` for all options.

## Generation

```bash
python test_generate.py --model_dir dit_weights/seed_42/position_composition1_10000
```

- `--model_dir` can be the run directory or a parent of it (it descends to the folder containing the checkpoints).
- By default only `best_model` is used; `--checkpoint_filter all` samples `--num_epochs` evenly spaced checkpoints, `-1` the last one, `<step>` a specific one.
- One prompt per class: counts/positions 1..`--counting`, every colour × class for composition runs, every colour pair for attribute runs, `--num_samples` images each (default 50). `--guidance_scale` > 1 enables classifier-free guidance.
- Images go to `outputs/<dit|unet>/gen_seed_<seed>/<weights>/<seed_dir>/<test_mode>/<param_string>/guidance_scale<g>_condition_scale1.0/images/<checkpoint>/prompt_key_<label>_sample_<n>.png`.

## Evaluation

```bash
python evaluate.py --images_dir outputs/dit/gen_seed_1/dit_weights/seed_42/position_composition1_10000/<param_string>/guidance_scale1.0_condition_scale1.0
```

`evaluate.py` infers the task from the path, loads the matching classifiers and reports per checkpoint:

- accuracy, mean confidence, per-label accuracy and a confusion matrix of the task classifier (count / position / colour pair);
- for composition runs additionally colour accuracy and joint accuracy, each for **all**, **held-out** and **in-distribution** label pairs;
- for attribute runs shape accuracy and joint (colour pair ∧ shape) accuracy.

Results are printed and saved next to the images in `evaluation/` as `evaluation[_<checkpoint>].json`, `.csv`, `_per_label.csv` and `_confusion_<checkpoint>.csv`. Use `--checkpoint_filter` to restrict to one checkpoint and `--no_save` to only print.

## Dataset Generation

See [mosaic/README.md](mosaic/README.md) for the Blender pipeline (configs for count, attribution, position, position-complex and the composition subsets) and the download link for the pre-generated datasets.

## TODO

- [x] MOSAIC dataset generation code
- [x] Training code
- [x] Evaluation code
- [ ] count-composition dataset

## Citation

```bibtex
@inproceedings{jeong2026mosaic,
  title={When Do Diffusion Models Learn to Generate Multiple Objects?},
  author={Jeong, Yujin and Uselis, Arnas and Laina, Iro and Oh, Seong Joon and Rohrbach, Anna},
  booktitle={ICML},
  year={2026}
}
```
