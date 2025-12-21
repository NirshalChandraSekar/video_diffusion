import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from scripts.encoder import globalEncoder
from scripts.dataset import VideoDataset
from scripts.unet import UNet3D

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import yaml
import os
import wandb

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = config["device"]



def forward_pass(batch, networks, noise_scheduler, device, num_frames_to_diffuse):
    '''
    text = batch of text embeddings (B, text_dim)
    rgbd_first_frame = first RGBD frame of the video (B, H, W, 4)
    frames_to_diffuse = depth frames to diffuse (B, T, H, W)
    '''
    text, rgbd_first_frame, frames_to_diffuse = batch
    global_encoder = networks["global_encoder"]
    noise_prediction_network = networks["noise_prediction_network"]

    cond_embedding = global_encoder(text, rgbd_first_frame.to(device))

    B, T, H, W = frames_to_diffuse.shape
    frames_to_diffuse = frames_to_diffuse.unsqueeze(2) # (B, T, 1, H, W)
    frames_to_diffuse = frames_to_diffuse.permute(0, 2, 1, 3, 4) # (B, 1, T, H, W)
    frames_to_diffuse = frames_to_diffuse.to(device) 

    noise = torch.randn(frames_to_diffuse.shape, device=device) # (B, 1, T, H, W)
    
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (B,), device=device
    ).long() # (B,)
    
    noisy_frames = noise_scheduler.add_noise(frames_to_diffuse, noise, timesteps)
    noise_pred = noise_prediction_network(noisy_frames, timesteps, cond_embedding, null_cond_prob=config["unet"]["null_cond_prob"])

    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    return loss


class DepthVideoDiffusionLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.global_encoder = globalEncoder()
        self.noise_prediction_network = UNet3D(
            dim=config["unet"]["dim"],
            cond_dim=config["encoder"]["global_embedding_dim"],
            out_dim=1,
            dim_mults=config["unet"]["dim_mults"],
            channels=1,
            attn_heads=config["unet"]["attn_heads"],
            attn_dim_head=config["unet"]["attn_dim_head"],
            init_kernel_size=7,
            use_sparse_linear_attn=True,
        )

        self.networks = nn.ModuleDict(
            {
                "global_encoder": self.global_encoder,
                "noise_prediction_network": self.noise_prediction_network,
            }
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["diffusion"]["num_timesteps"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.ema = None
        self._train_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self._train_dataset = VideoDataset(
                dataset_dir=config["dataset_dir"],
                num_frames_to_diffuse=config["num_diffusion_frames"],
            )

    def on_fit_start(self):
        if self.ema is None:
            self.ema = EMAModel(parameters=self.networks.parameters(), power=0.75)

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=config["training"]["shuffle"],
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"],
            persistent_workers=config["training"]["persistent_workers"],
        )

    def training_step(self, batch, batch_idx):
        loss = forward_pass(
            batch=batch,
            networks=self.networks,
            noise_scheduler=self.noise_scheduler,
            device=self.device,
            num_frames_to_diffuse=config["num_diffusion_frames"],
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.step(self.networks.parameters())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.networks.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        dataset_len = len(
            VideoDataset(
                dataset_dir=config["dataset_dir"],
                num_frames_to_diffuse=config["num_diffusion_frames"],
            )
        )
        steps_per_epoch = (dataset_len + config["training"]["batch_size"] - 1) // config[
            "training"]["batch_size"]
        num_training_steps = steps_per_epoch * config["training"]["num_epochs"]

        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=config["training"]["num_warmup_steps"],
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }


if __name__ == "__main__":
    model = DepthVideoDiffusionLightning()

    wandb.require("service")


    wandb_logger = WandbLogger(
        project="depth-video-diffusion",
        name="video-ddpm-correct-normalization",
        config=config,
    )

    os.makedirs("models", exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename="latest",
        save_top_k=1,
        save_weights_only=False,
        every_n_epochs=1,
        monitor="epoch", # to save every epoch
        mode="max",
    )


    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[5, 6, 7],
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    ckpt_path = "models/latest.ckpt"  # path to checkpoint, if resuming training
    if os.path.exists(ckpt_path):
        trainer.fit(model, ckpt_path=ckpt_path)
    else:
        trainer.fit(model)