import os
import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scripts.dataset import VideoDataset
from train_pl import DepthVideoDiffusionLightning  # rename file if needed


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["device"])


def load_networks(ckpt_path, device):
    model = DepthVideoDiffusionLightning.load_from_checkpoint(
        ckpt_path,
        map_location=device,
    )
    model.to(device)
    model.eval()
    global_encoder = model.global_encoder.eval()
    unet = model.noise_prediction_network.eval()
    networks = torch.nn.ModuleDict(
        {
            "global_encoder": global_encoder,
            "noise_prediction_network": unet,
        }
    ).to(device)
    return networks


def sample_depth_videos(networks):
    dataset = VideoDataset(
        dataset_dir=config["dataset_dir"],
        num_frames_to_diffuse=config["num_diffusion_frames"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        persistent_workers=config["training"]["persistent_workers"],
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["diffusion"]["num_timesteps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    num_inference_steps = config["diffusion"]["num_timesteps"]
    noise_scheduler.set_timesteps(num_inference_steps)

    networks.eval()

    save_root = "diffused_frames"
    os.makedirs(save_root, exist_ok=True)

    # Fix the random seed for reproducibility
    # torch.manual_seed(42)
    

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            text, rgbd_first_frame, frames_to_diffuse = batch

            rgbd_first_frame = rgbd_first_frame.to(device, dtype=torch.float32)
            frames_to_diffuse = frames_to_diffuse.to(device, dtype=torch.float32)

            global_encoder = networks["global_encoder"]
            noise_prediction_network = networks["noise_prediction_network"]

            cond_embedding = global_encoder(text, rgbd_first_frame).to(device)
            
            # null the conditioning for ablation
            # cond_embedding = torch.zeros_like(cond_embedding)

            B, T, H, W = frames_to_diffuse.shape
            sample_shape = (B, 1, T, H, W)
            samples = torch.randn(sample_shape, device=device)

            for k in tqdm(noise_scheduler.timesteps):
                t_int = int(k)
                # t_tensor = torch.tensor([t_int], device=device).long()
                t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long) # (B,)
                # noise_pred = noise_prediction_network.forward_with_cond_scale(samples, t_tensor, cond_embedding, cond_scale=config["diffusion"]["cond_scale"])
                noise_pred = noise_prediction_network(samples, t_tensor, cond_embedding)
                samples = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t_int,
                    sample=samples,
                ).prev_sample


            samples = samples.permute(0, 2, 3, 4, 1) # (B, T, H, W, 1)
            samples = samples[0].detach().cpu().numpy() # (T, H, W, 1)
            samples = np.squeeze(samples, axis=-1) # (T, H, W) (normalised between -1 and 1)
            
            max_depth = config["depth_range"]["max"]
            min_depth = config["depth_range"]["min"]
            
            # convert form -1 to 1 to 0 to 1
            samples = (samples + 1.0) / 2.0

            episode_dir = os.path.join(save_root, f"episode_{idx:04d}")
            os.makedirs(episode_dir, exist_ok=True)
            for t in range(samples.shape[0]):
                frame = np.clip(samples[t], 0.0, 1.0)
                frame_path = os.path.join(episode_dir, f"frame_{t:03d}.png")
                cv2.imwrite(frame_path, (frame * 255).astype(np.uint8))

            # Save first frame RGB for reference
            first_frame_rgb = rgbd_first_frame[0, :, :, :3].cpu().numpy()
            first_frame_rgb = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(episode_dir, "first_frame_rgb.png"),
                (first_frame_rgb * 255).astype(np.uint8),
            )

            # save the actual depth frames for reference
            actual_depth_frames = frames_to_diffuse[0].cpu().numpy()  # (T, H, W)
            for t in range(actual_depth_frames.shape[0]):
                depth_frame = actual_depth_frames[t]
                depth_frame = (depth_frame + 1.0) / 2.0  # normalize to 0-1
                
                frame_path = os.path.join(episode_dir, f"actual_depth_frame_{t:03d}.png")
                cv2.imwrite(frame_path, (depth_frame * 255).astype(np.uint8))

            # print the text prompt for reference
            print(f"Saved diffused frames for episode {idx:04d} with prompt: {text[0]}")


if __name__ == "__main__":
    ckpt_path = "models/latest-v1.ckpt"
    networks = load_networks(ckpt_path, device)
    sample_depth_videos(networks)