"""
Dataset class for loading depth video sequences along with conditioning RGBD inputs.
This script was documented with the help of ChatGPT, and verified by the authors.
"""

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle
import yaml
import cv2
import os

# Load configuration
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)



class VideoDataset(Dataset):
    """
    Dataset class for loading depth video sequences along with conditioning RGBD inputs
    and text variation descriptions.

    Each episode folder has:
        - front_rgb/ : RGB frames (0.png, 1.png, ...)
        - front_depth/ : depth encoded in RGB format
        - variation_descriptions.pkl : list of text descriptions (strings)
    """

    def __init__ (self, 
                  dataset_dir, 
                  num_frames_to_diffuse=config["num_diffusion_frames"]):
        # Path to dataset root folder
        self.dataset_dir = dataset_dir

        # Number of episodes (each episode is one sample)
        self.len_episodes = len(os.listdir(dataset_dir))

        # Number of depth frames sampled uniformly per episode
        self.num_frames_to_diffuse = num_frames_to_diffuse

    def __len__(self):
        return self.len_episodes
    
    def decode_depth(self, depth_rgb, scale_factor=256000.0):
        """
        Decode 24-bit RGB-encoded depth map into floating-point depth.

        depth_rgb: (H, W, 3) uint8 image
        encoding: depth = R*65536 + G*256 + B
        """
        depth_rgb = depth_rgb.astype(np.uint32)

        depth_int = (
            depth_rgb[:, :, 0] * 65536 +
            depth_rgb[:, :, 1] * 256 +
            depth_rgb[:, :, 2]
        )

        depth_float = depth_int.astype(np.float32) / scale_factor
        return depth_float
    
    def normalize_depth(self, depth):
        """
        Normalize depth values to [-1, 1] using global dataset statistics.
        depth_min/max loaded from config YAML.
        """
        depth_min = config['depth_range']['min']                                  
        depth_max = config['depth_range']['max']

        # Normalize to [-1, 1]
        depth_normalized = 2.0 * ( (depth - depth_min) / (depth_max - depth_min) ) - 1.0
        depth_normalized = np.clip(depth_normalized, -1.0, 1.0)

        return depth_normalized
    
    def normalize_rgb(self, rgb):
        """
        Normalize RGB frame to [0, 1] float32.
        """
        rgb_normalized = rgb.astype(np.float32) / 255.0
        return rgb_normalized
    
    def __getitem__(self, idx):
        """
        Returns:
            text (str): randomly sampled description
            rgbd_first_frame (H, W, 4): RGB + depth of the first frame
            frames_to_diffuse (T, H, W): depth frames only, for diffusion model
        """
        # Episode folder path (episode0, episode1, ...)
        episode_dir = os.path.join(self.dataset_dir, f"episode{idx}")

        # Total number of frames in this episode's RGB directory
        num_total_frames = len(os.listdir(os.path.join(episode_dir, "front_rgb")))

        # Uniformly sample the frames that will be diffused
        frame_indices = np.linspace(0, num_total_frames - 1, self.num_frames_to_diffuse, dtype=int)

        # -------------------------------
        # Load FIRST RGB frame
        # -------------------------------
        first_frame_rgb = cv2.imread(os.path.join(episode_dir, "front_rgb", f"{frame_indices[0]}.png"))
        first_frame_rgb = cv2.cvtColor(first_frame_rgb, cv2.COLOR_BGR2RGB)
        first_frame_rgb = self.normalize_rgb(first_frame_rgb) # (H, W, 3)
        
        # -------------------------------
        # Load FIRST depth frame
        # -------------------------------
        first_frame_depth = cv2.imread(os.path.join(episode_dir, "front_depth", f"{frame_indices[0]}.png"))
        first_frame_depth = cv2.cvtColor(first_frame_depth, cv2.COLOR_BGR2RGB)
        first_frame_depth = self.decode_depth(first_frame_depth)
        first_frame_depth = self.normalize_depth(first_frame_depth) # (H, W)

        # Concatenate RGB + depth → (H, W, 4)
        rgbd_first_frame = np.concatenate([first_frame_rgb, first_frame_depth[..., None]], axis=2)

        # -------------------------------
        # Load depth frames for diffusion
        # -------------------------------
        frames_to_diffuse = []
        frames_to_diffuse.append(first_frame_depth)  # First depth frame is always included

        # Load remaining frames
        for i in range(1, len(frame_indices)):
            depth = cv2.imread(os.path.join(episode_dir, "front_depth", f"{frame_indices[i]}.png"))
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            depth = self.decode_depth(depth)
            depth = self.normalize_depth(depth)
            frames_to_diffuse.append(depth)

        # Shape → (T, H, W)
        frames_to_diffuse = np.stack(frames_to_diffuse, axis=0)

        # -------------------------------
        # Load the variation description list
        # -------------------------------
        with open(os.path.join(episode_dir, "variation_descriptions.pkl"), "rb") as f:
            variational_descriptions = pickle.load(f)

        # Randomly sample one text instruction
        random_idx = np.random.randint(0, len(variational_descriptions))
        text = variational_descriptions[random_idx]

        # Return:
        # text: (str)
        # rgbd_first_frame: (H, W, 4)
        # frames_to_diffuse: (T, H, W)
        return text, rgbd_first_frame, frames_to_diffuse