"""
This script has the text and image encoders used to encode the conditioning information
This script was documented with the help of ChatGPT, and verified by the authors.
"""

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from sentence_transformers import SentenceTransformer
import yaml

from typing import Callable
from einops import rearrange
from transformers import BertModel, BertTokenizer
from typing import List, Union, Optional, Tuple


# Load configuration
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Device and embedding dimensions from config
DEVICE = config['device']
text_embedding_dim = config['encoder']['text_embedding_dim'] # 384
image_embedding_dim = config['encoder']['image_embedding_dim'] # 1024
global_embedding_dim = config['encoder']['global_embedding_dim'] # 384 + 1024 = 1408



def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module



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
    def __init__(self):
        super().__init__()

        # RGB encoder: standard ResNet expects 3 channels
        self.rgb_encoder = replace_bn_with_gn(get_resnet("resnet18"))

        # Depth encoder: make ResNet accept 1 channel
        self.depth_encoder = replace_bn_with_gn(get_resnet("resnet18"))

        old_conv = self.depth_encoder.conv1  # Conv2d(3, 64, 7, 2, 3, bias=False)
        self.depth_encoder.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,      # 64
            kernel_size=old_conv.kernel_size,        # (7, 7)
            stride=old_conv.stride,                  # (2, 2)
            padding=old_conv.padding,                # (3, 3)
            bias=(old_conv.bias is not None)
        )

    def forward(self, rgbd_images):
        """
        Forward pass for RGB-D encoder.

        Args:
            rgbd_images (torch.Tensor): (B, H, W, 4) tensor, channels-last.

        Returns:
            embedding (torch.Tensor): (B, 512+512).
        """

        rgb_image = rgbd_images[:, :, :, :3]  # (B, H, W, 3)
        depth_image = rgbd_images[:, :, :, 3:]  # (B, H, W, 1)

        # Encode RGB image
        rgb_image = rgb_image.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        rgb_features = self.rgb_encoder(rgb_image)  # (B, 512)

        # Encode Depth image
        depth_image = depth_image.permute(0, 3, 1, 2).contiguous()  # (B, 1, H, W)
        depth_features = self.depth_encoder(depth_image)  # (B, 512)

        # Concatenate RGB and Depth features
        embedding = torch.cat((rgb_features, depth_features), dim=1)  # (B, 1024)

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
        # RGB-D encoder using ResNet
        self.rgbd_encoder = RGBDImageEncoder()

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

        return combined