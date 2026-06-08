#!/bin/bash

# Counting
python data_generation/generate_dataset.py --dataset_name generation_configs/count.json --save_path ./data
# --gpu

# Attribution
python data_generation/generate_dataset.py --dataset_name generation_configs/attribution.json --save_path ./data
# --gpu

# Spatial relations without distractors
python data_generation/generate_dataset.py --dataset_name generation_configs/position.json --save_path ./data
# --gpu

# Spatial relations with distractors
python data_generation/generate_dataset.py --dataset_name generation_configs/position_distractor.json --save_path ./data
# --gpu