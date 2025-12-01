"""
This script reads the depth images from the dataset and finds the global minimum and maximum depth values across all images.
"""

import os
import cv2
import numpy as np
import yaml
from tqdm import tqdm

with open("../config.yaml", 'r') as f:
    config = yaml.safe_load(f)

def decode_depth(depth_rgb, scale_factor=256000.0):
    depth_rgb = depth_rgb.astype(np.uint32)
    depth_int = (
        depth_rgb[:, :, 0] * 65536 +
        depth_rgb[:, :, 1] * 256 +
        depth_rgb[:, :, 2]
    )
    depth_float = depth_int.astype(np.float32) / scale_factor
    return depth_float

if __name__ == "__main__":

    dataset_path = "../data/peract_dataset/"
    task_names = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    print(f"Found tasks: {task_names}")

    global_depth_min = float('inf')
    global_depth_max = float('-inf')

    for task_name in tqdm(task_names, desc="Tasks"):
        episodes_path = os.path.join(dataset_path, task_name, "all_variations", "episodes")
        episodes = [f for f in os.listdir(episodes_path) if os.path.isdir(os.path.join(episodes_path, f))]
        for episode in tqdm(episodes, desc=f"Episodes ({task_name})", leave=False):
            depth_image_folder = os.path.join(episodes_path, episode, "front_depth")
            depth_files = [f for f in os.listdir(depth_image_folder) if f.endswith(".png")]
            depth_files.sort()
            for file_name in depth_files:
                file_path = os.path.join(depth_image_folder, file_name)
                depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    continue
                depth_image_rgb = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
                decoded_depth = decode_depth(depth_image_rgb)
                current_min = np.min(decoded_depth)
                current_max = np.max(decoded_depth)
                if current_min < global_depth_min:
                    global_depth_min = current_min
                if current_max > global_depth_max:
                    global_depth_max = current_max

    print(f"Global depth min: {global_depth_min}")
    print(f"Global depth max: {global_depth_max}")

    # write the min and max in the config file
    config['depth_range']['min'] = float(global_depth_min)
    config['depth_range']['max'] = float(global_depth_max)
    with open("../config.yaml", 'w') as f:
        yaml.dump(config, f)