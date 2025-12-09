"""
This script has the text and image encoders used to encode the conditioning information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
import yaml

from einops import rearrange
from transformers import BertModel, BertTokenizer
from typing import List, Union, Optional, Tuple


with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

DEVICE = config['device']
text_embedding_dim = config['encoder']['text_embedding_dim']
image_embedding_dim = config['encoder']['image_embedding_dim']
global_embedding_dim = config['encoder']['global_embedding_dim']



# Helper function to check if a value exists (is not None)
def exists(val: Optional[Union[torch.Tensor, any]]) -> bool:
    return val is not None

# Singleton globals to hold preloaded BERT model and tokenizer
MODEL: Optional[BertModel] = None
TOKENIZER: Optional[BertTokenizer] = None
BERT_MODEL_DIM: int = 768  # Dimension size of BERT model output

# Function to retrieve the BERT tokenizer
def get_tokenizer() -> BertTokenizer:
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')
    return TOKENIZER

# Function to retrieve the BERT model, initializes if not loaded yet
def get_bert() -> BertModel:
    global MODEL
    if not exists(MODEL):
        MODEL = BertModel.from_pretrained('bert-base-cased')
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()  # Move to GPU if available
    return MODEL

# Function to tokenize a list or string of texts
def tokenize(texts: Union[str, List[str], Tuple[str]]) -> torch.Tensor:
    if not isinstance(texts, (list, tuple)):
        texts = [texts]  # Convert a single string to a list

    tokenizer = get_tokenizer()

    # Tokenize the input texts, returning PyTorch tensor of token ids
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,  # Add [CLS] and [SEP] tokens
        padding=True,  # Pad to the max sequence length
        return_tensors='pt'  # Return as PyTorch tensor
    )

    return encoding.input_ids

# Function to obtain BERT embeddings for tokenized texts
@torch.no_grad()  # No gradients are calculated in this function
def bert_embed(
    token_ids: torch.Tensor,
    return_cls_repr: bool = False,
    eps: float = 1e-8,
    pad_id: int = 0
) -> torch.Tensor:
    """
    Given tokenized input, returns embeddings from BERT.
    
    Parameters:
        token_ids (torch.Tensor): Tensor of token IDs.
        return_cls_repr (bool): If True, return the [CLS] token representation.
        eps (float): Small epsilon for numerical stability.
        pad_id (int): Token ID to treat as padding (defaults to 0).

    Returns:
        torch.Tensor: BERT embeddings (mean or [CLS] token).
    """
    model = get_bert()
    mask = token_ids != pad_id  # Create a mask to ignore padding tokens

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    # Obtain the hidden states from the BERT model
    outputs = model(
        input_ids=token_ids,
        attention_mask=mask,
        output_hidden_states=True  # Get hidden states from all layers
    )

    hidden_state = outputs.hidden_states[-1]  # Get the last layer of hidden states

    # If return_cls_repr is True, return the representation of the [CLS] token
    if return_cls_repr:
        return hidden_state[:, 0]  # [CLS] token is the first token

    # If no mask provided, return the mean of all tokens in the sequence
    if not exists(mask):
        return hidden_state.mean(dim=1)

    # Calculate the mean of non-padding tokens by applying the mask
    mask = mask[:, 1:]  # Ignore the [CLS] token in the mask
    mask = rearrange(mask, 'b n -> b n 1')  # Reshape for broadcasting

    # Weighted sum of token representations (ignoring padding)
    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)  # Normalize by sum of mask (to avoid division by zero)

    return masked_mean



class TextEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device

        # BERT model + tokenizer are handled by the global helpers:
        # get_tokenizer(), get_bert(), tokenize(), bert_embed()

        # BERT outputs 768-dim embeddings; we project to text_embedding_dim (e.g. 512)
        self.proj = nn.Sequential(
            nn.Linear(BERT_MODEL_DIM, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, text_embedding_dim),
        ).to(self.device)

    def encode(self, texts):
        """
        texts: str or list[str]
        returns: (B, text_embedding_dim) tensor on self.device
        """
        # 1. Tokenize with your helper
        token_ids = tokenize(texts)  # (B, L) on CPU

        # 2. Get BERT embeddings using your bert_embed() helper
        #    return_cls_repr=True → use [CLS] token representation
        emb = bert_embed(token_ids, return_cls_repr=True)  # (B, 768)

        # bert_embed already moves to CUDA if available,
        # but we ensure it's on self.device for safety:
        emb = emb.to(self.device)

        # 3. (Optional but nice) normalize BERT embedding
        emb = F.normalize(emb, dim=-1)

        # 4. Project to text_embedding_dim (e.g. 512)
        emb = self.proj(emb)  # (B, text_embedding_dim)

        return emb


# class TextEncoder(nn.Module):
#     def __init__(self, device=DEVICE):
#         super().__init__()
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
#         self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
#         self.model.eval()

#         self.aligner = nn.Sequential(
#             nn.Linear(text_embedding_dim, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, text_embedding_dim)
#         ).to(self.device)

#     def encode(self, texts):
#         encoded_input = self.tokenizer(texts, 
#                                        padding=True,
#                                        return_tensors='pt'
#                                     ).to(self.device)

#         with torch.no_grad():
#             outputs = self.model(**encoded_input)

#         output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
#         output = F.normalize(output, dim=-1)
#         output = self.aligner(output)  # (batch_size, hidden_size)
        
#         return output


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
        
        # self.linear = nn.Linear(text_embedding_dim + image_embedding_dim,
        #                         global_embedding_dim)  # Combine text and image embeddings

        self.fusion = nn.Sequential(
            nn.Linear(text_embedding_dim + image_embedding_dim, (text_embedding_dim + image_embedding_dim) * 2),
            nn.ReLU(inplace=True),
            nn.Linear((image_embedding_dim + text_embedding_dim) * 2, global_embedding_dim)
        )

    def forward(self, texts, rgbd_images):
        """        
        texts: list of strings, length B
        rgbd_images: (B, H, W, 4) tensor
        returns: (B, global_embedding_dim) tensor
        """
        text_embeddings = self.text_encoder.encode(texts)  # (B, hidden_size)
        rgbd_embeddings = self.rgbd_encoder(rgbd_images)   # (B, output_embedding_dim)

        text_embeddings = F.normalize(text_embeddings, dim=-1)
        rgbd_embeddings = F.normalize(rgbd_embeddings, dim=-1)

        combined = torch.cat((text_embeddings, rgbd_embeddings), dim=1)  # (B, hidden_size + output_embedding_dim)
        global_embedding = self.fusion(combined)  # (B, global_embedding_dim)

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