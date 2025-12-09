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
            
            # convert from [-1, 1] to [0, 1]
            samples = (samples + 1.0) / 2.0

            episode_dir = os.path.join(save_root, f"episode_{idx:04d}")
            os.makedirs(episode_dir, exist_ok=True)

            # --- get actual depth frames and normalize to [0, 1] ---
            actual_depth_frames = frames_to_diffuse[0].cpu().numpy()  # (T, H, W)
            actual_depth_frames = (actual_depth_frames + 1.0) / 2.0   # [0, 1]

            # --- convert both to uint8 once ---
            # --- convert both to uint8 once ---
            gen_u8 = (np.clip(samples, 0.0, 1.0) * 255).astype(np.uint8)   # (T, H, W)
            gt_u8  = (np.clip(actual_depth_frames, 0.0, 1.0) * 255).astype(np.uint8)

            T, H, W = gen_u8.shape  # T should be 10

            LABEL_HEIGHT = 40
            grid_height = LABEL_HEIGHT + 2 * H
            grid_width  = T * W

            grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

            for t in range(T):
                x_start = t * W
                x_end   = (t + 1) * W

                # --- TOP ROW = ACTUAL ---
                grid[LABEL_HEIGHT : LABEL_HEIGHT + H, x_start:x_end] = gt_u8[t]

                # --- BOTTOM ROW = GENERATED ---
                grid[LABEL_HEIGHT + H : LABEL_HEIGHT + 2*H, x_start:x_end] = gen_u8[t]

                # -----------------------------
                # Draw white background for label
                # -----------------------------
                label = f"t = {t}"

                # Rectangle background
                cv2.rectangle(
                    grid,
                    (x_start, 0),
                    (x_end, LABEL_HEIGHT),
                    color=(255,),  # white background
                    thickness=-1,   # filled rectangle
                )

                # Put black text on white background
                cv2.putText(
                    grid,
                    label,
                    (x_start + 10, LABEL_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,),  # black text
                    2,
                    cv2.LINE_AA,
                )

            # Save final grid
            out_path = os.path.join(episode_dir, "depth_grid_labeled.png")
            cv2.imwrite(out_path, grid)


            # (optional) if you STILL want individual frames, keep this loop;
            # otherwise you can delete it.
            for t in range(1):
                # frame_path_gen = os.path.join(episode_dir, f"gen_frame_{t:03d}.png")
                # cv2.imwrite(frame_path_gen, gen_u8[t])

                frame_path_gt = os.path.join(episode_dir, f"actual_depth_frame_{t:03d}.png")
                cv2.imwrite(frame_path_gt, gt_u8[t])

            # Save first frame RGB for reference (unchanged)
            first_frame_rgb = rgbd_first_frame[0, :, :, :3].cpu().numpy()
            first_frame_rgb = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(episode_dir, "first_frame_rgb.png"),
                (first_frame_rgb * 255).astype(np.uint8),
            )

            # print the text prompt for reference
            print(f"Saved diffused frames for episode {idx:04d} with prompt: {text[0]}")


if __name__ == "__main__":
    ckpt_path = "models/latest-v1.ckpt"
    networks = load_networks(ckpt_path, device)
    sample_depth_videos(networks)