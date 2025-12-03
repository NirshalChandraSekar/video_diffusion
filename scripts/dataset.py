import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle
import yaml
import cv2
import os

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)



class VideoDataset(Dataset):
    def __init__ (self, 
                  dataset_dir, 
                  num_frames_to_diffuse=config["num_diffusion_frames"]):
        self.dataset_dir = dataset_dir
        self.len_episodes = len(os.listdir(dataset_dir))
        self.num_frames_to_diffuse = num_frames_to_diffuse

    def __len__(self):
        return self.len_episodes
    
    def decode_depth(self, depth_rgb, scale_factor=256000.0):
        depth_rgb = depth_rgb.astype(np.uint32)
        depth_int = (
            depth_rgb[:, :, 0] * 65536 +
            depth_rgb[:, :, 1] * 256 +
            depth_rgb[:, :, 2]
        )
        depth_float = depth_int.astype(np.float32) / scale_factor
        return depth_float
    
    def normalize_depth(self, depth):
        depth_min = config['depth_range']['min']
        depth_max = config['depth_range']['max']
        # Normalize to [0, 1]
        # depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        # depth_normalized = np.clip(depth_normalized, 0.0, 1.0)

        depth_normalized = 2.0 * ( (depth - depth_min) / (depth_max - depth_min) ) - 1.0
        depth_normalized = np.clip(depth_normalized, -1.0, 1.0) # clip to [-1, 1]


        return depth_normalized
    
    def normalize_rgb(self, rgb):
        rgb_normalized = rgb.astype(np.float32) / 255.0
        return rgb_normalized
    
    def __getitem__(self, idx):
        episode_dir = os.path.join(self.dataset_dir, f"episode{idx}")

        num_total_frames = len(os.listdir(os.path.join(episode_dir, "front_rgb")))
        # uniformly sample "num_frames_to_diffuse" frames from the episode
        frame_indices = np.linspace(0, num_total_frames - 1, self.num_frames_to_diffuse, dtype=int)

        first_frame_rgb = cv2.imread(os.path.join(episode_dir, "front_rgb", f"{frame_indices[0]}.png"))
        first_frame_rgb = cv2.cvtColor(first_frame_rgb, cv2.COLOR_BGR2RGB)
        first_frame_rgb = self.normalize_rgb(first_frame_rgb)
        
        first_frame_depth = cv2.imread(os.path.join(episode_dir, "front_depth", f"{frame_indices[0]}.png"))
        first_frame_depth = cv2.cvtColor(first_frame_depth, cv2.COLOR_BGR2RGB)
        first_frame_depth = self.decode_depth(first_frame_depth)
        first_frame_depth = self.normalize_depth(first_frame_depth)

        rgbd_first_frame = np.concatenate([first_frame_rgb, first_frame_depth[..., None]], axis=2)  # (H, W, 4)

        frames_to_diffuse = []
        frames_to_diffuse.append(first_frame_depth)  # include the first frame depth
        for i in range(1, len(frame_indices)):
            depth = cv2.imread(os.path.join(episode_dir, "front_depth", f"{frame_indices[i]}.png"))
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            depth = self.decode_depth(depth)
            depth = self.normalize_depth(depth)
            frames_to_diffuse.append(depth)
        frames_to_diffuse = np.stack(frames_to_diffuse, axis=0)  # (num_frames_to_diffuse, H, W)

        # Read the .pkl file 
        with open(os.path.join(episode_dir, "variation_descriptions.pkl"), "rb") as f:
            variational_descriptions = pickle.load(f)

        text = variational_descriptions[0]

        return text, rgbd_first_frame, frames_to_diffuse
    


# sample usage
if __name__ == "__main__":
    dataset = VideoDataset(
        dataset_dir="../data/peract_processed_dataset",
        num_frames_to_diffuse=4
    )
    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    depth_min = config["depth_range"]["min"]
    depth_max = config["depth_range"]["max"]
    depth_range = depth_max - depth_min

    for batch in dataloader:
        text, rgbd_first_frame, frames_to_diffuse = batch
        print(f"Text length: {len(text)}")
        print(f"RGBD First Frame shape: {rgbd_first_frame.shape}")
        print(f"Frames to Diffuse shape: {frames_to_diffuse.shape}")

        for i in range(1):
            episode_dir = f"debug_episode_{i}"
            os.makedirs(episode_dir, exist_ok=True)

            first_frame_rgb = rgbd_first_frame[i, :, :, :3].cpu().numpy()
            first_frame_depth = rgbd_first_frame[i, :, :, 3].cpu().numpy()

            first_frame_depth_unnormalized = ((first_frame_depth + 1.0) / 2.0) * depth_range + depth_min
            depth_to_save = np.clip(
                (first_frame_depth_unnormalized - depth_min) / depth_range,
                0.0,
                1.0,
            )

            cv2.imwrite(
                os.path.join(episode_dir, "first_frame_rgb.png"),
                (first_frame_rgb * 255).astype(np.uint8),
            )
            cv2.imwrite(
                os.path.join(episode_dir, "first_frame_depth.png"),
                (depth_to_save * 255).astype(np.uint8),
            )

            for t in range(frames_to_diffuse.shape[1]):
                frame_depth = frames_to_diffuse[i, t, :, :].cpu().numpy()
                frame_depth_unnormalized = ((frame_depth + 1.0) / 2.0) * depth_range + depth_min
                depth_to_save = np.clip(
                    (frame_depth_unnormalized - depth_min) / depth_range,
                    0.0,
                    1.0,
                )
                cv2.imwrite(
                    os.path.join(episode_dir, f"frame_{t}_depth.png"),
                    (depth_to_save * 255).astype(np.uint8),
                )

        break
