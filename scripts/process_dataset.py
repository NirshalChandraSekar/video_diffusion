"""
This script is used to process the dataset by normalizing depth images and saving them in a specified format.
"""

import os
import cv2
import numpy as np
import yaml

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

def normalize_depth(depth_image, depth_min, depth_max):
    normalized = (depth_image - depth_min) / (depth_max - depth_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized

if __name__ == "__main__":

    depth_image = cv2.imread("../data/peract_dataset/close_jar/all_variations/episodes/episode0/front_depth/0.png", cv2.IMREAD_UNCHANGED)
    depth_image_rgb = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)

    decoded_depth = decode_depth(depth_image_rgb)
    print("decoded depth stats:")
    print(f"Min: {np.min(decoded_depth)}")
    print(f"Max: {np.max(decoded_depth)}")
    cv2.imwrite("decoded_depth.png", (decoded_depth / np.max(decoded_depth) * 255).astype(np.uint8))

    depth_min = config['depth_range']['min']
    depth_max = config['depth_range']['max']

    normalized_depth = normalize_depth(decoded_depth, depth_min, depth_max)
    normalized_depth = normalized_depth.astype(np.float32)
    print("normalized depth stats:")
    print(f"Min: {np.min(normalized_depth)}")
    print(f"Max: {np.max(normalized_depth)}")

    # visualise the normalized depth
    normalized_depth_visual = (normalized_depth * 255).astype(np.uint8)
    cv2.imwrite("normalized_depth.png", normalized_depth_visual)
