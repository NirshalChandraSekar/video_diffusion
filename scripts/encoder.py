"""
This script has the text and image encoders used to encode the conditioning information
This script was documented with the help of ChatGPT, and verified by the authors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from sentence_transformers import SentenceTransformer
import yaml

from einops import rearrange
from transformers import BertModel, BertTokenizer
from typing import List, Union, Optional, Tuple


# Load configuration
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Device and embedding dimensions from config
DEVICE = config['device']
text_embedding_dim = config['encoder']['text_embedding_dim']
image_embedding_dim = config['encoder']['image_embedding_dim']
global_embedding_dim = config['encoder']['global_embedding_dim']


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def forward(self, texts):
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings # (B, 384)



class RGBDImageEncoder(nn.Module):
    """
    Lightweight CNN encoder that maps an RGB-D image (4 channels: R, G, B, depth)
    to a single global conditioning embedding vector.
    """
    def __init__(self, output_embedding_dim=image_embedding_dim):
        super().__init__()

        # Input expected as (B, 4, H, W)
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # Spatial downsampling: H/2, W/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # Spatial downsampling: H/4, W/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # → global pooling: (B, 256, 1, 1)
        )

        # Final fully-connected layer → global image embedding
        self.fc = nn.Linear(256, output_embedding_dim)

    def forward(self, rgbd_images):
        """
        Forward pass for RGB-D encoder.

        Args:
            rgbd_images (torch.Tensor): (B, H, W, 4) tensor, channels-last.

        Returns:
            embedding (torch.Tensor): (B, output_embedding_dim) global vector.
        """
        # Convert to channels-first (NCHW) for Conv2d
        x = rgbd_images.permute(0, 3, 1, 2).contiguous()

        features = self.conv_net(x)            # (B, 256, 1, 1)
        features = features.flatten(1)         # (B, 256)

        embedding = self.fc(features)          # (B, output_embedding_dim)
        return embedding


class globalEncoder(nn.Module):
    """
    Combines text and RGB-D encoders into a single global conditioning vector.

    - TextEncoder → text_embedding_dim
    - RGBDImageEncoder → image_embedding_dim
    - Concatenate and fuse → global_embedding_dim
    """
    def __init__(self):
        super().__init__()
        # Text encoder using BERT + projection
        self.text_encoder = TextEncoder()
        # RGB-D encoder using small CNN backbone
        self.rgbd_encoder = RGBDImageEncoder(output_embedding_dim=image_embedding_dim)
        
        # Fusion MLP to combine text + image embeddings into a global embedding
        # Input dim: text_embedding_dim + image_embedding_dim
        # Hidden dim: 2 * (text + image)
        # Output dim: global_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(text_embedding_dim + image_embedding_dim, (text_embedding_dim + image_embedding_dim) * 2),
            nn.ReLU(inplace=True),
            nn.Linear((image_embedding_dim + text_embedding_dim) * 2, global_embedding_dim)
        )

    def forward(self, texts, rgbd_images):
        """        
        Jointly encode text and RGB-D image(s) into a single global embedding.

        Args:
            texts: list of strings, length B
            rgbd_images (torch.Tensor): (B, H, W, 4) tensor

        Returns:
            global_embedding (torch.Tensor): (B, global_embedding_dim) tensor
        """
        # Encode text into dense embeddings
        text_embeddings = self.text_encoder(texts)  # (B, text_embedding_dim)
        # Encode RGB-D images into dense embeddings
        rgbd_embeddings = self.rgbd_encoder(rgbd_images)   # (B, image_embedding_dim)

        # Normalize both embeddings to unit norm
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        rgbd_embeddings = F.normalize(rgbd_embeddings, dim=-1)

        # Concatenate along feature dimension
        combined = torch.cat((text_embeddings, rgbd_embeddings), dim=1)  # (B, text_embedding_dim + image_embedding_dim)

        # Pass through fusion MLP to obtain final global embedding
        global_embedding = self.fusion(combined)  # (B, global_embedding_dim)

        return global_embedding