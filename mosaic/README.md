# MOSAIC: Multi-Object Spatial relations, AttrIbution, Counting

## Overview

MOSAIC is a controlled diagnostic framework for analyzing multi-object generation failures in text-to-image diffusion models. It isolates three compositional factors — **Color Attribution**, **Counting**, and **Spatial Relations** — enabling causal analysis of data effects.

![alt text](mosaic.png)

## TODO

- [x] Default dataset generation code
- [ ] Grid dataset generation code
- [ ] Link for downloading pre-generated images

## Repository Layout

- `data_generation/generate_dataset.py`: main entry point for dataset generation
- `data_generation/dataset_generators.py`: task creation and execution
- `data_generation/utils.py`: Blender scene/object/render logic
- `generation_configs/xx.json`: default config
- `generate_dataset.sh`: example CPU/debug/GPU launcher


## Requirements

- Python 3.11 (the `bpy==4.2.21` wheel is only published for Python 3.11)
- Download object assets from [COMFORT](https://github.com/sled-group/COMFORT/tree/main) repository
- Place object assets in `./mosaic/data_generation/assets`
- Place background blend files in `./mosaic/data_generation/background`
- Install Python dependencies: `pip install -r requirements.txt`

### System libraries

`bpy` bundles Blender but expects the X11/OpenGL runtime libraries from the operating system. On a desktop Linux they are usually present; on a headless server/container install them first, otherwise `import bpy` fails with `libX11.so.6: cannot open shared object file`:

```bash
sudo apt-get install -y libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxext6 libsm6 libice6 libgl1 libxkbcommon0
```

Without root access, use the conda environment instead. It installs Python 3.11, the libraries above from conda-forge and the pip requirements in one step; the helper script then points `bpy` at the env's `lib/` folder:

```bash
conda env create -f environment.yml
conda activate mosaic-datagen
bash setup_bpy_rpath.sh
```

The first GPU render compiles the Cycles CUDA kernels (several minutes); later renders are fast.

## How To Run

Generate datasets using the provided script:

```bash
bash generate_dataset.sh
```

Or run individual dataset generation commands:

**Counting Dataset**
```bash
python data_generation/generate_dataset.py --dataset_name generation_configs/count.json --save_path ./data --gpu
```

**Attribution Dataset**
```bash
python data_generation/generate_dataset.py --dataset_name generation_configs/attribution.json --save_path ./data --gpu
```

**Spatial Relations (Position Dataset)**
```bash
python data_generation/generate_dataset.py --dataset_name generation_configs/position.json --save_path ./data --gpu
```

**Spatial Relations with Distractors (complex)**
```bash
python data_generation/generate_dataset.py --dataset_name generation_configs/position_complex.json --save_path ./data --gpu
```

**Spatial Relations, all object colours (composition subset)**
```bash
python data_generation/generate_dataset.py --dataset_name generation_configs/position_composition_subset.json --save_path ./data --gpu
```

**Counting, all object colours (composition subset)**
```bash
python data_generation/generate_dataset.py --dataset_name generation_configs/count_composition_subset.json --save_path ./data --gpu
```

The output folder is named after the config file (`generation_configs/count_composition_subset.json` → `data/comfort_ball_count_composition_subset/`).

Remove `--gpu` flag to run on CPU.

The training code (`../data.py`) expects the datasets under `mosaic/data/` with these folder names: `comfort_ball_count`, `comfort_ball_attribution`, `comfort_ball_attribute_complex`, `comfort_ball_attribute_composition_subset`, `comfort_ball_position`, `comfort_ball_position_complex`, `comfort_ball_position_composition_subset`, `comfort_ball_count_composition_subset`. The `*_composition_subset` datasets contain every (colour, class) pair; the held-out pairs are selected at training time by the `composition<N>` test_mode (see the top-level README).


## Configuration

All dataset generation is controlled via JSON config files in `generation_configs/`:

- **num_images_per_class**: Number of images to generate per class
- **obj_shape**: Object shape (sphere, cube, cylinder)
- **obj_color**: Color(s) — can be a list `["red", "blue"]` or single value `"red"`
- **obj_size**: Size of the object (float, typically 0.5-2.0)
- **max_complexity**: Maximum object count (for counting and attribution) or distractor limit (for spatial relations)
- **cam_position**: Camera position (default: `[0, 0, 10]`)
- **obj2**: Second object properties (for attribution and spatial relations)
- **distractor**: Distractor object properties (for complex scenes)

## Output

- Output root is controlled by `--save_path`
- Generated data is saved under a dataset folder derived from config
- For counting, filenames encode color, index, and count class
- For attribution, images contain two object types with specific color labels
- For spatial relations, images contain two objects positioned within specified angle ranges

## Generated images

If you want to just download the dataset that has been already generated, please find it [here](https://nextcloud-rack.mai.informatik.tu-darmstadt.de/s/PAoH6BqDDLbDwyK) and extract it into `mosaic/data/` (one folder per dataset, e.g. `mosaic/data/comfort_ball_count/`). This is the location the training code expects.

## Notes

- **Multiprocessing**: In CPU mode (without `--debug`), generation uses multiprocessing for faster processing. Logs from multiple workers may interleave.
- **Debugging**: Use `--debug` flag for easier troubleshooting of object placement and count behavior.
- **Scene Bounds**: Objects are constrained to remain visible within camera frame (±4.0 range). Spatial relations use angle ranges to position obj2 relative to obj1.
- **Angle Ranges**: For position datasets, angle ranges (e.g., `[-45, 45]`) define the angular constraint from obj1 for obj2 placement (in degrees).
- **Collision Detection**: Objects are automatically checked for collisions and repositioned if needed.

## Acknowledgement

Most of the code is from the original [COMFORT](https://github.com/sled-group/COMFORT/tree/main) repository. We gratefully acknowledge their contribution. 

