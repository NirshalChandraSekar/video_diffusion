import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.encoder import globalEncoder
from scripts.dataset import VideoDataset
from scripts.unet import UNet3D

from tqdm import tqdm
import numpy as np
import yaml

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = config["device"]





def forward_pass(batch, networks, noise_scheduler, device, num_frames_to_diffuse):
    
    text, rgbd_first_frame, frames_to_diffuse = batch
    """
    text: list of strings, length B
    rgbd_first_frame: (B, H, W, 4) tensor
    frames_to_diffuse: (B, num_frames_to_diffuse, H, W) tensor
    """
    global_encoder = networks["global_encoder"]
    noise_prediction_network = networks["noise_prediction_network"]

    # Extract the global conditioning embedding
    cond_embedding = global_encoder(text, 
                                    rgbd_first_frame.to(device))
    

    # Diffusion process
    B, T, H, W = frames_to_diffuse.shape
    frames_to_diffuse = frames_to_diffuse.unsqueeze(2)  # (B, T, 1, H, W)
    frames_to_diffuse = frames_to_diffuse.permute(0, 2, 1, 3, 4)  # (B, 1, T, H, W) unet expects channel dim at 1
    frames_to_diffuse = frames_to_diffuse.to(device)
    noise = torch.randn(frames_to_diffuse.shape, device=device)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
    noisy_frames = noise_scheduler.add_noise(frames_to_diffuse, noise, timesteps)
    noise_pred = noise_prediction_network(noisy_frames, timesteps, cond_embedding)

    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    return loss
    



def train_epoch(dataloader,
                networks,
                optimizer,
                lr_scheduler,
                ema,
                noise_scheduler,
                device,
                num_frames_to_diffuse):
    
    epoch_loss = []
    networks.train()

    with tqdm(dataloader, desc='Training') as tepoch:
        for batch in tepoch:
            loss = forward_pass(batch,
                                networks,
                                noise_scheduler,
                                device,
                                num_frames_to_diffuse
                                )
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(networks.parameters())

            loss = loss.item()
            epoch_loss.append(loss)

            tepoch.set_postfix(loss=loss)

    return np.mean(epoch_loss)



def train():

    device = DEVICE

    dataset = VideoDataset(dataset_dir=config["dataset_dir"],
                           num_frames_to_diffuse=config["num_diffusion_frames"])
    
    dataloader = DataLoader(
                            dataset,
                            batch_size=config["training"]["batch_size"],
                            shuffle=config["training"]["shuffle"],
                            num_workers=config["training"]["num_workers"],
                            pin_memory=config["training"]["pin_memory"],
                            persistent_workers=config["training"]["persistent_workers"]
                            )
    
    global_encoder = globalEncoder().to(device)
    noise_prediction_network = UNet3D(
                                      dim=config["unet"]["dim"],
                                      cond_dim=config["encoder"]["global_embedding_dim"],
                                      out_dim=1,
                                      dim_mults=config["unet"]["dim_mults"],
                                      channels=1,
                                      attn_heads=config["unet"]["attn_heads"],
                                      attn_dim_head=config["unet"]["attn_dim_head"],
                                      init_kernel_size=7,
                                      use_sparse_linear_attn=True
                                      ).to(device)
    
    networks = torch.nn.ModuleDict({
        "global_encoder": global_encoder,
        "noise_prediction_network": noise_prediction_network
    }).to(device)
    
    optimizer = torch.optim.AdamW(
        params=networks.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config["training"]["num_warmup_steps"],
        num_training_steps=len(dataloader) * config["training"]["num_epochs"]
    )

    ema = EMAModel(parameters=networks.parameters(), power=0.75)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["diffusion"]["num_timesteps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon"
    )

    for epoch in range(config["training"]["num_epochs"]):

        epoch_loss = train_epoch(
            dataloader,
            networks,
            optimizer,
            lr_scheduler,
            ema,
            noise_scheduler,
            device,
            config["num_diffusion_frames"]
        )
    

train()