"""
This script has the text and image encoders used to encode the conditioning information
This script was documented with the help of ChatGPT, and verified by the authors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
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



# Helper function to check if a value exists (is not None)
def exists(val: Optional[Union[torch.Tensor, any]]) -> bool:
    return val is not None

# Singleton globals to hold preloaded BERT model and tokenizer
MODEL: Optional[BertModel] = None
TOKENIZER: Optional[BertTokenizer] = None
BERT_MODEL_DIM: int = 768  # Dimension size of BERT model output

# Function to retrieve the BERT tokenizer
def get_tokenizer() -> BertTokenizer:
    """
    Lazily load and return a singleton BERT tokenizer.
    Ensures that the tokenizer is only loaded once.
    """
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')
    return TOKENIZER

# Function to retrieve the BERT model, initializes if not loaded yet
def get_bert() -> BertModel:
    """
    Lazily load and return a singleton BERT base model.
    Moves the model to CUDA if available.
    """
    global MODEL
    if not exists(MODEL):
        MODEL = BertModel.from_pretrained('bert-base-cased')
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()  # Move to GPU if available
    return MODEL

# Function to tokenize a list or string of texts
def tokenize(texts: Union[str, List[str], Tuple[str]]) -> torch.Tensor:
    """
    Tokenize input text(s) using the BERT tokenizer.

    Args:
        texts: A single string, or list/tuple of strings.

    Returns:
        token_ids (torch.Tensor): Long tensor of shape (B, L) with token IDs.
    """
    if not isinstance(texts, (list, tuple)):
        texts = [texts]  # Convert a single string to a list

    tokenizer = get_tokenizer()

    # Tokenize the input texts, returning PyTorch tensor of token ids
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,  # Add [CLS] and [SEP] tokens
        padding=True,             # Pad to the max sequence length in the batch
        return_tensors='pt'       # Return as PyTorch tensor
    )

    return encoding.input_ids

# Function to obtain BERT embeddings for tokenized texts
@torch.no_grad()
def bert_embed(
    token_ids: torch.Tensor,
    return_cls_repr: bool = False,
    eps: float = 1e-8,
    pad_id: int = 0
) -> torch.Tensor:
    """
    Given tokenized input, returns embeddings from BERT.
    
    Parameters:
        token_ids (torch.Tensor): Tensor of token IDs of shape (B, L).
        return_cls_repr (bool): If True, return the [CLS] token representation (B, D).
                                Otherwise, return masked mean over tokens (excluding CLS).
        eps (float): Small epsilon for numerical stability when normalizing.
        pad_id (int): Token ID to treat as padding (defaults to 0).

    Returns:
        torch.Tensor: BERT embeddings of shape (B, D) where D is hidden size.
    """
    model = get_bert()
    mask = token_ids != pad_id  # Create a mask to ignore padding tokens

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    outputs = model(
        input_ids=token_ids,
        attention_mask=mask,
        output_hidden_states=True  # Get hidden states from all layers
    )

    # Last layer hidden states: (B, L, D)
    hidden_state = outputs.hidden_states[-1]  # Get the last layer of hidden states

    if return_cls_repr:
        return hidden_state[:, 0]  # [CLS] token is the first token

    if not exists(mask):
        return hidden_state.mean(dim=1)

    # Calculate the mean of non-padding tokens by applying the mask
    mask = mask[:, 1:]  # Ignore the [CLS] token in the mask
    mask = rearrange(mask, 'b n -> b n 1')  # Reshape for broadcasting to (B, L-1, 1)

    # Weighted sum of token representations (ignoring padding)
    numer = (hidden_state[:, 1:] * mask).sum(dim=1)  # (B, D)
    denom = mask.sum(dim=1)                          # (B, 1)
    masked_mean = numer / (denom + eps)              # Normalize by sum of mask (avoid division by zero)

    return masked_mean



class TextEncoder(nn.Module):
    """
    Wraps BERT + a small projection network to map text into
    a fixed-size embedding of dimension `text_embedding_dim`.
    """
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device

        self.proj = nn.Sequential(
            nn.Linear(BERT_MODEL_DIM, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, text_embedding_dim),
        ).to(self.device)

    def encode(self, texts):
        """
        Encode raw text(s) into dense embeddings.

        Args:
            texts: str or list[str] of length B

        Returns:
            emb (torch.Tensor): (B, text_embedding_dim) tensor on self.device.
        """
        token_ids = tokenize(texts)  # (B, L) on CPU

        emb = bert_embed(token_ids, return_cls_repr=True)  # (B, 768)

        emb = emb.to(self.device)

        emb = F.normalize(emb, dim=-1)

        emb = self.proj(emb)  # (B, text_embedding_dim)

        return emb


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
        text_embeddings = self.text_encoder.encode(texts)  # (B, text_embedding_dim)
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