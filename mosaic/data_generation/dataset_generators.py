"""
Dataset generation functions for different path types.
This module contains the logic for generating datasets with different path configurations.
"""

import os
import copy
import random
import multiprocessing
from PIL import Image
from tqdm import tqdm

from data_generation.utils import render_scene_config

def _is_valid_image(image_path):
    """Check if image file exists and is valid."""
    if not os.path.exists(image_path):
        return False
    try:
        with Image.open(image_path) as image:
            image.convert("RGB")
        return True
    except (OSError, IOError):
        return False


def _should_generate_image(save_path):
    """Return True when image should be generated (missing or invalid)."""
    return (not os.path.exists(save_path)) or (not _is_valid_image(save_path))


def execute_tasks(tasks, debug, gpu, description="Generating"):
    """
    Execute tasks sequentially (GPU) or in parallel (CPU).
    
    Args:
        tasks: List of task tuples
        gpu: Whether to use GPU
        description: Progress bar description
    """
    if gpu or debug:
        for task_args in tqdm(tasks, desc=description):
            _worker_wrapper(task_args)
    else:
        # Parallel execution for CPU
        num_workers = min(32, os.cpu_count())
        print(len(tasks), num_workers)
        with multiprocessing.Pool(num_workers) as pool:
            list(tqdm(
                pool.imap(_worker_wrapper, tasks), 
                total=len(tasks), 
                desc=description
            ))

def _worker_wrapper(args):
    """
    Wrapper function for multiprocessing. Must be at module level to be picklable.
    Unpacks: first 5 elements are positional args, last is kwargs dict
    """
    config, step, gpu, kwargs_dict = args

    render_config = dict(config)
    render_kwargs = dict(kwargs_dict)

    save_path = render_config.get('save_path')
    # Re-check at execution time to avoid duplicate work when files were created meanwhile.
    if save_path and not _should_generate_image(save_path):
        return None, None

    mapping, distractors = render_scene_config(
        **render_config,
        render_shadow=True,
        cuda=gpu,
        step=step,
        **render_kwargs
    )
    return mapping, distractors

def generate_count_dataset(config, args):
    """Generate count dataset with all variations."""
    tasks = []
    
    # Normalize obj_color to list
    obj_colors = config['obj_color'] if isinstance(config['obj_color'], list) else [config['obj_color']]
    
    for obj_color in obj_colors:
        for i in range(config['num_images_per_class']):
            for num_object in range(1, config['max_complexity'] + 1):
                save_path = os.path.join(config["save_path"], f'{obj_color}_{i}_{num_object}.png')

                if _should_generate_image(save_path):
                    config_copy = copy.deepcopy(config)
                    config_copy['obj_color'] = obj_color
                    config_copy['save_path'] = save_path
                    
                    tasks.append((
                        config_copy, i, args.gpu,
                        {'num_objects': num_object}
                    ))
    
    print(f"Total tasks to generate: {len(tasks)}")
    random.shuffle(tasks)
    execute_tasks(tasks, args.debug, args.gpu, description="Generating Count Dataset")

def generate_attribution_dataset(config, args):
    """Generate attribution dataset with all variations."""
    tasks = []

    obj_colors = config['obj_color'] if isinstance(config['obj_color'], list) else [config['obj_color']]
    obj2_colors = config['obj2_color'] if isinstance(config['obj2_color'], list) else [config['obj2_color']]

    for obj_color in obj_colors:
        for obj2_color in obj2_colors:
            for i in range(config['num_images_per_class']):
                    save_path = os.path.join(config["save_path"], f'{obj_color}_{obj2_color}_{i}.png')

                    if _should_generate_image(save_path):
                        config_copy = copy.deepcopy(config)
                        config_copy['obj_color'] = obj_color # Set object color to reference color for attribution
                        config_copy['obj2_color'] = obj2_color # Set second object color to reference color for attribution
                        config_copy['save_path'] = save_path  # Remove save_path from config passed to worker
                        
                        tasks.append((
                            config_copy, i, args.gpu, {
                                'num_objects': config['max_complexity'],
                            }
                        ))
                        
    print(f"Total tasks to generate: {len(tasks)}")

    random.shuffle(tasks)
    execute_tasks(tasks, args.debug, args.gpu, description="Generating Attribution Dataset")

def generate_position_dataset(config, args):
    """Generate position dataset with all variations."""
    tasks = []

    obj_colors = config['obj_color'] if isinstance(config['obj_color'], list) else [config['obj_color']]
    obj2_colors = config['obj2_color'] if isinstance(config['obj2_color'], list) else [config['obj2_color']]

    angles = {
        "position1": (0, 18),        # 0° - 18°   (18° range)
        "position2": (36, 54),       # 36° - 54°   (18° range, 18° gap)
        "position3": (72, 90),       # 72° - 90°   (18° range, 18° gap)
        "position4": (108, 126),     # 108° - 126° (18° range, 18° gap)
        "position5": (144, 162),     # 144° - 162° (18° range, 18° gap)
        "position6": (180, 198),     # 180° - 198° (18° range, 18° gap)
        "position7": (216, 234),     # 216° - 234° (18° range, 18° gap)
        "position8": (252, 270),     # 252° - 270° (18° range, 18° gap)
        "position9": (288, 306),     # 288° - 306° (18° range, 18° gap)
        "position10": (324, 342),    # 324° - 342° (18° range, 18° gap)
    }

    for obj_color in obj_colors:
        for obj2_color in obj2_colors:
            for i in range(config['num_images_per_class']):
                for position, angle_range in angles.items():
                    save_path = os.path.join(config["save_path"], f'{obj_color}_{obj2_color}_{position}_{i}.png')

                    if _should_generate_image(save_path):
                        config_copy = copy.deepcopy(config)
                        config_copy['obj_color'] = obj_color # Set object color to reference color for attribution
                        config_copy['obj2_color'] = obj2_color # Set second object color to reference color for attribution
                        config_copy['save_path'] = save_path  # Remove save_path from config passed to worker
                        
                        tasks.append((
                            config_copy, i, args.gpu, {
                                'num_objects': config['max_complexity'],
                                'angle_range': angle_range,
                            }
                        ))
                        
    print(f"Total tasks to generate: {len(tasks)}")

    random.shuffle(tasks)
    execute_tasks(tasks, args.debug, args.gpu, description="Generating Position Dataset")


