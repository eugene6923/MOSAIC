import os
import sys
sys.path.append(os.getcwd())

import json
import argparse

from data_generation.constants import *
from data_generation.config_handler import get_color_from_name, get_shape_from_name
from data_generation.dataset_generators import (
    generate_count_dataset,
    generate_attribution_dataset,
    generate_position_dataset
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Render scene configuration script")
    parser.add_argument(
        '--dataset_name', type=str, required=True, 
        help="Dataset name to specify the configuration (json file)"
    )
    parser.add_argument('--save_path', type=str, default=None, help="Path to save the rendered images")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for rendering")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer samples)")
    return parser.parse_args()


def process_dataset(config_data, args):
    default_config = config_data['default_config']

    config = {**default_config}  # Merge default config with variation

    if "count" in config['dataset_name']:
        generate_count_dataset(config, args)
    elif "position" in config['dataset_name']:
        generate_position_dataset(config, args)
    elif "attribution" in config['dataset_name']:
        generate_attribution_dataset(config, args)

def load_json_config(file_name, save_path):
    """Load configuration from JSON file."""

    try:
        with open(file_name, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {file_name}")
    
    dataset_name = os.path.splitext(os.path.basename(file_name))[0]
    
    default_config = json_data['default_config']
    variations = json_data['variations'] if 'variations' in json_data else None
    
    # Convert color and shape names to actual values
    if isinstance(default_config["obj_color"], str):
        # Convert "available_colors" to actual list
        default_config["obj_color"] = get_color_from_name(default_config["obj_color"])
    elif isinstance(default_config["obj_color"], list):
        # Handle list of color names
        if all(isinstance(color, str) for color in default_config["obj_color"]):
            default_config["obj_color"] = [get_color_from_name(color) if isinstance(color, str) else color 
                                         for color in default_config["obj_color"]]

    # Save original shape values BEFORE any conversion
    original_obj_shape = default_config["obj_shape"]

    # Check if using real objects BEFORE converting shapes
    # Get obj_shape as a list for checking - use ORIGINAL values
    obj_shapes_to_check = original_obj_shape
    if not isinstance(obj_shapes_to_check, list):
        obj_shapes_to_check = [obj_shapes_to_check]
    
    # Check if any of the shapes are real objects (not SPHERE, CUBE, CYLINDER)
    from data_generation.constants import SPHERE, CUBE, CYLINDER
    basic_shapes = {SPHERE, CUBE, CYLINDER}

    basic_shape_names = {"SPHERE", "CUBE", "CYLINDER", "Sphere", "ShapeCube", "ShapeCylinder"}
    
    uses_real_objects = any(
        shape not in basic_shapes and shape not in basic_shape_names
        for shape in obj_shapes_to_check
    )

    if uses_real_objects:
        actual_dataset_name = "comfort_car_ref_facing_left"

    else:
        actual_dataset_name = "comfort_ball_" + dataset_name if "comfort_ball_" not in dataset_name else dataset_name

        if isinstance(original_obj_shape, str):
            default_config["obj_shape"] = get_shape_from_name(original_obj_shape)
    

    if "obj2_color" in default_config:
        if isinstance(default_config["obj2_color"], str):
            # Convert "available_colors" to actual list
            default_config["obj2_color"] = get_color_from_name(default_config["obj2_color"])
        elif isinstance(default_config["obj2_color"], list):
            # Handle list of color names
            if all(isinstance(color, str) for color in default_config["obj2_color"]):
                default_config["obj2_color"] = [get_color_from_name(color) if isinstance(color, str) else color 
                                            for color in default_config["obj2_color"]]

        if isinstance(default_config["obj2_shape"], str):
            default_config["obj2_shape"] = get_shape_from_name(default_config["obj2_shape"])

    if "distractor_color" in default_config:
        if isinstance(default_config["distractor_color"], str):
            default_config["distractor_color"] = get_color_from_name(default_config["distractor_color"])
        elif isinstance(default_config["distractor_color"], list):
            if all(isinstance(color, str) for color in default_config["distractor_color"]):
                default_config["distractor_color"] = [get_color_from_name(color) if isinstance(color, str) else color 
                                            for color in default_config["distractor_color"]]

        if isinstance(default_config["distractor_shape"], str):
            default_config["distractor_shape"] = get_shape_from_name(default_config["distractor_shape"])

    default_config["dataset_name"] = actual_dataset_name  # Update dataset name in config

    return {
        'default_config': default_config,
        'variations': variations,
    }


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set save path
    config = load_json_config(args.dataset_name, args.save_path)
    save_path = os.path.join(args.save_path, config['default_config']['dataset_name'])
    config["default_config"]["save_path"] = save_path

    os.makedirs(save_path, exist_ok=True)

    process_dataset(config, args)

    print("Dataset generation completed!", file=sys.stderr)

if __name__ == "__main__":
    main()