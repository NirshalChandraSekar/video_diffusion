"""
This script has the text and image encoders used to encode the conditioning information
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

DEVICE = config['device']
text_embedding_dim = config['encoder']['text_embedding_dim']
image_embedding_dim = config['encoder']['image_embedding_dim']
global_embedding_dim = config['encoder']['global_embedding_dim']



class TextEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.model.eval()

    def encode(self, texts):
        encoded_input = self.tokenizer(texts, 
                                       padding=True,
                                       return_tensors='pt'
                                    ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_input)

        output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        return output


class RGBDImageEncoder(nn.Module):
    """
    Lightweight CNN encoder that maps an RGB-D image (4 channels)
    to a single global conditioning embedding vector.
    """
    def __init__(self, output_embedding_dim=image_embedding_dim):
        super().__init__()

        # Input: (B, 4, H, W)
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/2, W/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/4, W/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (B, 256, 1, 1)
        )

        # Final MLP → global embedding
        self.fc = nn.Linear(256, output_embedding_dim)

    def forward(self, rgbd_images):
        """
        rgbd_images: (B, H, W, 4)
        returns: (B, output_embedding_dim)
        """
        # Convert to NCHW
        x = rgbd_images.permute(0, 3, 1, 2).contiguous()

        features = self.conv_net(x)            # (B, 256, 1, 1)
        features = features.flatten(1)         # (B, 256)

        embedding = self.fc(features)          # (B, output_embedding_dim)
        return embedding


class globalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.rgbd_encoder = RGBDImageEncoder(output_embedding_dim=image_embedding_dim)
        
        self.linear = nn.Linear(text_embedding_dim + image_embedding_dim,
                                global_embedding_dim)  # Combine text and image embeddings

    def forward(self, texts, rgbd_images):
        """        
        texts: list of strings, length B
        rgbd_images: (B, H, W, 4) tensor
        returns: (B, global_embedding_dim) tensor
        """
        text_embeddings = self.text_encoder.encode(texts)  # (B, hidden_size)
        rgbd_embeddings = self.rgbd_encoder(rgbd_images)   # (B, output_embedding_dim)

        combined = torch.cat((text_embeddings, rgbd_embeddings), dim=1)  # (B, hidden_size + output_embedding_dim)
        global_embedding = self.linear(combined)  # (B, global_embedding_dim)

        return global_embedding

        



# Debug usage
if __name__ == "__main__":
    text_encoder = TextEncoder()
    sample_text = ["This is a sample text for encoding."]

    embeddings = text_encoder.encode(sample_text)
    print(embeddings.shape)
    print(type(embeddings))

    rgbd_encoder = RGBDImageEncoder(output_embedding_dim=512)
    dummy_images = torch.randn(1, 240, 320, 4)  # (B, H, W, 4)
    rgbd_embeddings = rgbd_encoder(dummy_images)
    print(rgbd_embeddings.shape)

    global_encoder = globalEncoder().to(DEVICE)
    dummy_images = dummy_images.to(DEVICE)
    global_embeddings = global_encoder(sample_text, dummy_images)
    print(global_embeddings.shape)