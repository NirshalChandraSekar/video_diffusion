import os
import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scripts.dataset import VideoDataset
from train_pl import DepthVideoDiffusionLightning


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
    networks = torch.nn.ModuleDict(
        {
            "global_encoder": model.global_encoder.eval(),
            "noise_prediction_network": model.noise_prediction_network.eval(),
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

    noise_scheduler.set_timesteps(config["diffusion"]["num_timesteps"])

    save_root = "diffused_frames"
    os.makedirs(save_root, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            text, rgbd_first_frame, frames_to_diffuse = batch

            rgbd_first_frame = rgbd_first_frame.to(device, dtype=torch.float32)
            frames_to_diffuse = frames_to_diffuse.to(device, dtype=torch.float32)

            global_encoder = networks["global_encoder"]
            noise_prediction_network = networks["noise_prediction_network"]

            cond_embedding = global_encoder(text, rgbd_first_frame).to(device)

            B, T, H, W = frames_to_diffuse.shape
            samples = torch.randn((B, 1, T, H, W), device=device)

            for k in tqdm(noise_scheduler.timesteps):
                t_int = int(k)
                t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)
                noise_pred = noise_prediction_network(samples, t_tensor, cond_embedding)
                samples = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t_int,
                    sample=samples,
                ).prev_sample

            samples = samples.permute(0, 2, 3, 4, 1)[0].cpu().numpy()
            samples = np.squeeze(samples, axis=-1)
            samples = (samples + 1.0) / 2.0

            actual_depth_frames = frames_to_diffuse[0].cpu().numpy()
            actual_depth_frames = (actual_depth_frames + 1.0) / 2.0

            gen_u8 = (np.clip(samples, 0.0, 1.0) * 255).astype(np.uint8)
            gt_u8  = (np.clip(actual_depth_frames, 0.0, 1.0) * 255).astype(np.uint8)

            gen_color = np.stack([cv2.applyColorMap(gen_u8[t], cv2.COLORMAP_TURBO) for t in range(T)], axis=0)
            gt_color  = np.stack([cv2.applyColorMap(gt_u8[t],  cv2.COLORMAP_TURBO) for t in range(T)], axis=0)

            LABEL_HEIGHT = 40
            grid_height = LABEL_HEIGHT + 2 * H
            grid_width  = T * W
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

            for t in range(T):
                x_start = t * W
                x_end   = (t + 1) * W
                grid[LABEL_HEIGHT : LABEL_HEIGHT + H, x_start:x_end] = gt_color[t]
                grid[LABEL_HEIGHT + H : LABEL_HEIGHT + 2*H, x_start:x_end] = gen_color[t]
                cv2.rectangle(grid, (x_start, 0), (x_end, LABEL_HEIGHT), (255, 255, 255), -1)
                cv2.putText(grid, f"t = {t}", (x_start + 10, LABEL_HEIGHT - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            episode_dir = os.path.join(save_root, f"episode_{idx:04d}")
            os.makedirs(episode_dir, exist_ok=True)

            cv2.imwrite(os.path.join(episode_dir, "depth_grid_labeled.png"), grid)
            cv2.imwrite(os.path.join(episode_dir, "actual_depth_frame_000.png"), gt_color[0])

            first_frame_rgb = rgbd_first_frame[0, :, :, :3].cpu().numpy()
            first_frame_rgb = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(episode_dir, "first_frame_rgb.png"),
                (first_frame_rgb * 255).astype(np.uint8),
            )

            print(f"Saved diffused frames for episode {idx:04d} with prompt: {text[0]}")


if __name__ == "__main__":
    ckpt_path = "models/latest.ckpt"
    networks = load_networks(ckpt_path, device)
    sample_depth_videos(networks)
