"""
This script reads the PerAct2 dataset and processes all depth images in a folder
to make sense for our depth video generation task.
"""

import os
import cv2
import numpy as np


def decode_depth(depth_rgb, scale_factor=256000.0):
    """Decodes RLBench 24-bit depth PNG."""
    depth_rgb = depth_rgb.astype(np.uint32)

    # Combine back into a single 24-bit number
    depth_int = (
        depth_rgb[:, :, 0] * 65536 +  # R
        depth_rgb[:, :, 1] * 256   +  # G
        depth_rgb[:, :, 2]           # B
    )

    # Convert to meters
    depth_float = depth_int.astype(np.float32) / scale_factor
    return depth_float


if __name__ == "__main__":
    folder_path = "../data/peract_dataset/light_bulb_in/all_variations/episodes/episode0/front_depth/"
    
    # List all PNG files
    depth_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    depth_files.sort()  # Optional: process in order

    for file_name in depth_files:
        file_path = os.path.join(folder_path, file_name)
        depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"Failed to read {file_name}")
            continue

        # Convert BGR to RGB exactly like your original code
        depth_image_rgb = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)

        print(f"File: {file_name}")
        # print("Depth image shape:", depth_image_rgb.shape)
        # print("Depth image data type:", depth_image_rgb.dtype)
        # print("Depth image min value:", np.min(depth_image_rgb))
        # print("Depth image max value:", np.max(depth_image_rgb))

        decoded_depth = decode_depth(depth_image_rgb)
        print("Decoded depth shape:", decoded_depth.shape)
        # print("Decoded depth data type:", decoded_depth.dtype)
        print("Decoded depth min value:", np.min(decoded_depth))
        print("Decoded depth max value:", np.max(decoded_depth))

        # visualize the depth map using cv2 gray scale 
        depth_visual = (decoded_depth / np.max(decoded_depth) * 255).astype(np.uint8)
        # cv2.imwrite(f"decoded_depth_visual_{file_name}", depth_visual)
        print("-" * 40)
