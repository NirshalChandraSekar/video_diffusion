import os
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import shutil

with open("../config.yaml", 'r') as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":

    dataset_path = "../data/peract_dataset/"
    destination_folder = "../data/peract_processed_dataset"
    os.makedirs(destination_folder, exist_ok=True)

    task_names = [
        f for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f)) and not f.startswith(".")
    ]
    task_names.sort()

    episode_count = 0

    for task_name in tqdm(task_names, desc="Tasks"):
        episodes_path = os.path.join(dataset_path, task_name, "all_variations", "episodes")
        if not os.path.isdir(episodes_path):
            continue

        episodes = [
            f for f in os.listdir(episodes_path)
            if os.path.isdir(os.path.join(episodes_path, f)) and not f.startswith(".")
        ]
        episodes.sort()

        for episode in tqdm(episodes, desc=f"Episodes ({task_name})", leave=False):
            source_episode_folder = os.path.join(episodes_path, episode)
            dest_episode_folder = os.path.join(destination_folder, f"episode{episode_count}")
            os.makedirs(dest_episode_folder, exist_ok=True)

            for item in os.listdir(source_episode_folder):
                src = os.path.join(source_episode_folder, item)
                dst = os.path.join(dest_episode_folder, item)

                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

            episode_count += 1