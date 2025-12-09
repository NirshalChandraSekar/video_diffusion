import os
import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scripts.dataset import VideoDataset
from scripts.encoder import globalEncoder
from train_pl import DepthVideoDiffusionLightning


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# get a batch from the dataloader

dataset = VideoDataset(
    dataset_dir=config["dataset_dir"],
    num_frames_to_diffuse=config["num_diffusion_frames"],
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=config["training"]["num_workers"],
    pin_memory=config["training"]["pin_memory"],
    persistent_workers=config["training"]["persistent_workers"],
)

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

if __name__ == "__main__":

    ckpt_path = "models/latest.ckpt"
    networks = load_networks(ckpt_path, device)

    batch = next(iter(dataloader))
    text, rgbd_first_frame, frames_to_diffuse = batch
    frames_to_diffuse = frames_to_diffuse.unsqueeze(2)
    print("Text:", text)
    print("RGBD First Frame Shape:", rgbd_first_frame.shape)
    print("Frames to Diffuse Shape:", frames_to_diffuse.shape)

    encoder = networks["global_encoder"]
    
    with torch.no_grad():
        cond_embedding = encoder(text, rgbd_first_frame.to(device))

    print("Conditioning Embedding Shape:", cond_embedding.shape)
    # difference between the global embeddings for the two samples

    embedding_diff = torch.norm(cond_embedding[0] - cond_embedding[1]).item()
    print("L2 Norm between the two conditioning embeddings:", embedding_diff)
    