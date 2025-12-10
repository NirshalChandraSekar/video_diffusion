# unet3d_backbone.py
# 3D U-Net backbone with time embedding + optional FiLM global conditioning.
# Dependencies: torch, einops, einops_exts

import math
from typing import Optional, Callable, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from einops_exts import rearrange_many
import yaml

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Load configuration (e.g., device) from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = config["device"]

# -----------------------
# Helper utilities
# -----------------------

# Check if a value is not None
def exists(x):
    return x is not None

# Return val if it exists, else default d (callable or value)
def default(val, d):
    return val if exists(val) else d() if callable(d) else d

# Create a boolean mask of given shape with probability 'prob' of being True
def prob_mask_like(shape, prob, device=None, dtype=torch.bool):
    if prob <= 0:
        return torch.zeros(shape, dtype=dtype, device=device)
    if prob >= 1:
        return torch.ones(shape, dtype=dtype, device=device)
    return torch.rand(shape, device=device) < prob

# Check if an integer is odd
def is_odd(n: int) -> bool:
    return n % 2 == 1

# -----------------------
# Relative Position Bias (temporal)
# -----------------------

class RelativePositionBias(nn.Module):
    """
    T5-style relative position bias, applied along temporal dimension (frames).
    Used to provide temporal awareness in attention over time.
    """
    def __init__(self, heads: int = 8, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # Learnable bias embeddings per bucket and per head
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> Tensor:
        # T5-style relative position bucketing
        n = -relative_position
        num_buckets //= 2
        # Sign bucket: positive vs negative relative position
        ret = (n < 0).long() * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # Log-bucket for large distances
        val_if_large = max_exact + (
            (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact)) *
            (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1)
        )
        # Combine sign and magnitude buckets
        return ret + torch.where(is_small, n.long(), val_if_large)

    def forward(self, n: int, device: torch.device) -> Tensor:
        # n = number of frames
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        # relative positions (i,j)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        # map to bucket indices
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        # lookup bias per bucket and head
        values = self.relative_attention_bias(rp_bucket)  # (n, n, heads)
        return rearrange(values, 'i j h -> h i j')        # (heads, n, n)

# -----------------------
# Sinusoidal time embedding
# -----------------------

class SinusoidalPosEmb(nn.Module):
    """
    Standard 1D sinusoidal positional / time embedding.
    Used to encode diffusion timestep into a vector.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B,) or (B, 1) time / diffusion steps
        returns: (B, dim)
        """
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.squeeze(-1)

        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        # frequencies
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        # outer product of times and frequencies
        emb = x[:, None] * emb[None, :]
        # sin/cos concat
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# -----------------------
# Upsample / Downsample (3D, spatial-only)
# -----------------------

def Upsample(dim: int) -> nn.ConvTranspose3d:
    # upsample H, W by 2; keep F (frames) unchanged using transposed conv
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim: int) -> nn.Conv3d:
    # downsample H, W by 2; keep F (frames) unchanged using stride-2 conv
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# -----------------------
# Norm layers
# -----------------------

class LayerNorm3D(nn.Module):
    """
    LayerNorm-style normalization over channel dimension of 3D feature maps.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable per-channel scale
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # Compute mean and variance over channels
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    """
    RMSNorm variant operating over channels for 3D feature maps.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        # Per-channel scale parameter
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # L2-normalize along channel dimension and scale
        return F.normalize(x, dim=1) * self.scale * self.gamma

# -----------------------
# PreNorm wrapper
# -----------------------

class PreNorm(nn.Module):
    """
    Apply LayerNorm3D before a given module (e.g., attention).
    """
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm3D(dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.norm(x)
        return self.fn(x, **kwargs)

# -----------------------
# FiLM (Feature-wise Linear Modulation)
# -----------------------

class FiLM(nn.Module):
    """
    FiLM modulation: generate per-channel scale and shift
    from a conditioning vector.
    """
    def __init__(self, cond_dim: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2)
        )

    def forward(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        cond: (B, cond_dim)
        returns:
          scale, shift: (B, C, 1, 1, 1)
        """
        out = self.layers(cond)  # (B, 2*C)
        scale, shift = out.chunk(2, dim=-1)
        scale = scale.view(scale.shape[0], -1, 1, 1, 1)
        shift = shift.view(shift.shape[0], -1, 1, 1, 1)
        return scale, shift

# -----------------------
# Basic Conv Block + ResNet Block
# -----------------------

class Block(nn.Module):
    """
    Basic 3D conv block with RMSNorm + SiLU.
    Supports optional FiLM scale/shift modulation.
    """
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        # Apply FiLM if provided
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)

class ResnetBlock(nn.Module):
    """
    Two Block()s with a residual skip connection.
    Optionally FiLM-conditioned with an external cond vector.
    """
    def __init__(self, dim: int, dim_out: int, cond_dim: Optional[int] = None):
        super().__init__()
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        # Match dimensions for residual
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        # FiLM for conditioning if cond_dim is provided
        self.film = FiLM(cond_dim, dim_out) if exists(cond_dim) else None

    def forward(self, x: Tensor, global_cond: Optional[Tensor] = None) -> Tensor:
        scale_shift = None
        if exists(self.film):
            assert exists(global_cond), "global_cond must be provided when cond_dim is set"
            # Compute FiLM modulation from global condition
            scale_shift = self.film(global_cond)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

# -----------------------
# Attention modules
# -----------------------

class SpatialLinearAttention(nn.Module):
    """
    Linear attention over spatial dimensions HxW, applied per frame.
    Implemented as a 2D attention on each (F) slice.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, F, H, W)
        """
        b, c, f, h, w = x.shape
        # Reshape to group frames into batch
        x2 = rearrange(x, 'b c f h w -> (b f) c h w')
        qkv = self.to_qkv(x2).chunk(3, dim=1)
        # Project into multi-head q,k,v
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)
        # Normalize across "d" for queries and across "n" for keys
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        # Compute context = K * V^T in "linear" form
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        # Then reconstruct output via context * Q
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        # Restore original grouping of frames
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

class Attention(nn.Module):
    """
    Standard multi-head attention over an arbitrary sequence.
    Can be used for temporal or spatial tokens depending on reshape.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        pos_bias: Optional[Tensor] = None,
        focus_present_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        x: (..., n, dim)
        pos_bias: (heads, n, n) optional temporal relative position bias
        focus_present_mask: (B,) optional mask controlling whether to attend
                            only to present frame (self) or all frames.
        """
        n = x.shape[-2]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # (..., n, dim) -> (..., h, n, d)
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)
        q = q * self.scale

        # Compute similarity scores
        sim = torch.einsum('... h i d, ... h j d -> ... h i j', q, k)

        # Add relative position bias if given
        if exists(pos_bias):
            sim = sim + rearrange(pos_bias, 'h i j -> 1 h i j')

        # Optional focus mask: for some samples only allow self-attention
        if exists(focus_present_mask) and not (~focus_present_mask).all():
            device = x.device
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)
            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j')
            )
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # Stabilize before softmax
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        # Weighted sum of values
        out = torch.einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# -----------------------
# Einops wrapper + Residual
# -----------------------

class EinopsToAndFrom(nn.Module):
    """
    Wrap a function that expects a different tensor layout.
    Uses einops to go from one shape description to another and back.
    """
    def __init__(self, from_einops: str, to_einops: str, fn: Callable[[Tensor], Tensor]):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        # Map current shape to named axes dict
        shape_dict = dict(zip(self.from_einops.split(' '), x.shape))
        # Rearrange to target pattern
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        # Apply wrapped function
        x = self.fn(x, **kwargs)
        # Rearrange back to original pattern
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **shape_dict)
        return x

class Residual(nn.Module):
    """
    Standard residual wrapper: y = fn(x) + x
    """
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, **kwargs) + x

# -----------------------
# Main 3D U-Net backbone
# -----------------------

class UNet3D(nn.Module):
    """
    3D U-Net backbone for video inputs.

    Args:
        dim: base channel dimension.
        cond_dim: dimension of external global conditioning (e.g. text + image). Optional.
        out_dim: number of output channels (defaults to input `channels`).
        dim_mults: list of multipliers for channels at each UNet level.
        channels: number of input channels.
        attn_heads: number of attention heads.
        attn_dim_head: per-head dimension.
        init_kernel_size: spatial kernel size for initial conv (must be odd).
        use_sparse_linear_attn: whether to use SpatialLinearAttention for spatial attention.
        time_dim_factor: factor for time embedding dimension = dim * factor.
    """
    def __init__(
        self,
        dim: int = 64,
        cond_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        channels: int = 3,
        attn_heads: int = 8,
        attn_dim_head: int = 32,
        init_kernel_size: int = 7,
        use_sparse_linear_attn: bool = True,
        time_dim_factor: int = 4,
    ):
        super().__init__()
        assert is_odd(init_kernel_size), "init_kernel_size must be odd"

        self.channels = channels
        self.dim = dim
        self.has_cond = exists(cond_dim)
        self.attn_heads = attn_heads

        # initial conv (spatial only) to lift input to base feature dim
        init_dim = dim
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels,
            init_dim,
            (1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        # time embedding MLP: maps scalar t -> time_dim
        time_dim = dim * time_dim_factor
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.time_dim = time_dim

        # combined FiLM conditioning dimension: time + (optional) global cond
        film_cond_dim = time_dim + (cond_dim or 0)
        self.film_cond_dim = film_cond_dim

        # classifier-free guidance null embedding (for external cond part only)
        if self.has_cond:
            self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim))
        else:
            self.null_cond_emb = None

        # dims per level (encoder/decoder)
        dims = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        # temporal attention factory: operate along time dimension
        def temporal_attn(d):
            return EinopsToAndFrom(
                'b c f h w',
                'b (h w) f c',
                Attention(d, heads=attn_heads, dim_head=attn_dim_head),
            )

        # relative position bias for temporal attention
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, num_buckets=32, max_distance=32
        )

        # initial temporal attention block after first conv
        self.init_temporal = Residual(
            PreNorm(
                init_dim,
                temporal_attn(init_dim),
            )
        )

        # -----------------------
        # Downsampling path
        # -----------------------
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            # 2x ResNet blocks with FiLM conditioning (time + global cond)
            block1 = ResnetBlock(dim_in, dim_out, cond_dim=self.film_cond_dim)
            block2 = ResnetBlock(dim_out, dim_out, cond_dim=self.film_cond_dim)
            # Spatial attention (linear variant) or identity
            spatial_attn = (
                Residual(
                    PreNorm(
                        dim_out,
                        SpatialLinearAttention(
                            dim_out, heads=attn_heads, dim_head=attn_dim_head
                        ),
                    )
                )
                if use_sparse_linear_attn
                else nn.Identity()
            )
            # Temporal attention per level
            temp_attn = Residual(
                PreNorm(dim_out, temporal_attn(dim_out))
            )
            # Downsample spatially except at last level
            downsample = Downsample(dim_out) if not is_last else nn.Identity()
            self.downs.append(nn.ModuleList([block1, block2, spatial_attn, temp_attn, downsample]))

        # -----------------------
        # Bottleneck / middle
        # -----------------------
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim=self.film_cond_dim)
        # Spatial attention at bottleneck (across tokens (h*w))
        self.mid_spatial_attn = Residual(
            PreNorm(
                mid_dim,
                EinopsToAndFrom(
                    'b c f h w',
                    'b f (h w) c',
                    Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head),
                ),
            )
        )
        # Temporal attention at bottleneck
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim))
        )
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim=self.film_cond_dim)

        # -----------------------
        # Upsampling path
        # -----------------------
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            # Note: on up path, dim_out * 2 from skip connection concat
            block1 = ResnetBlock(dim_out * 2, dim_in, cond_dim=self.film_cond_dim)
            block2 = ResnetBlock(dim_in, dim_in, cond_dim=self.film_cond_dim)
            spatial_attn = (
                Residual(
                    PreNorm(
                        dim_in,
                        SpatialLinearAttention(
                            dim_in, heads=attn_heads, dim_head=attn_dim_head
                        ),
                    )
                )
                if use_sparse_linear_attn
                else nn.Identity()
            )
            temp_attn = Residual(
                PreNorm(dim_in, temporal_attn(dim_in))
            )
            # Upsample spatially except at final level
            upsample = Upsample(dim_in) if not is_last else nn.Identity()
            self.ups.append(nn.ModuleList([block1, block2, spatial_attn, temp_attn, upsample]))

        # final conv stage: combine with early-res skip and project to out channels
        final_dim = dim
        self.final_res_block = ResnetBlock(
            final_dim * 2, final_dim, cond_dim=self.film_cond_dim
        )
        self.final_conv = nn.Conv3d(final_dim, out_dim or channels, kernel_size=1)

    # Optional helper for CFG at inference
    def forward_with_cond_scale(
        self,
        x: Tensor,
        time: Tensor,
        cond: Optional[Tensor] = None,
        cond_scale: float = 1.0,
        **kwargs,
    ) -> Tensor:
        """
        Apply classifier-free guidance at inference:
        eps = eps_uncond + (eps_cond - eps_uncond) * cond_scale
        """
        if not self.has_cond or cond_scale == 1.0:
            return self.forward(x, time, cond=cond, null_cond_prob=0.0, **kwargs)

        # Conditional prediction
        eps_cond = self.forward(x, time, cond=cond, null_cond_prob=0.0, **kwargs)
        # Unconditional prediction (mask out cond)
        eps_uncond = self.forward(x, time, cond=cond, null_cond_prob=1.0, **kwargs)
        # Guided combination
        return eps_uncond + (eps_cond - eps_uncond) * cond_scale

    def forward(
        self,
        x: Tensor,                      # (B, C, T, H, W)
        time: Tensor,                   # (B,) or (B, 1)
        cond: Optional[Tensor] = None,  # (B, cond_dim) if used
        null_cond_prob: float = 0.0,
        focus_present_mask: Optional[Tensor] = None,
        prob_focus_present: float = 0.0,
    ) -> Tensor:
        """
        Forward pass through the 3D UNet.

        Args:
            x: input video tensor (B, C, T, H, W).
            time: diffusion timestep tensor (B,) or (B,1).
            cond: global conditioning tensor (B, cond_dim) if used.
            null_cond_prob: probability of dropping cond for classifier-free guidance during training.
            focus_present_mask: optional boolean mask to force some samples to attend only to present time.
            prob_focus_present: probability of enabling focus_present_mask when not provided.

        Returns:
            Tensor of shape (B, out_channels, T, H, W)
        """
        assert x.ndim == 5, "x must be (B, C, T, H, W)"
        b, _, t, _, _ = x.shape
        device = x.device

        if self.has_cond:
            assert exists(cond) and cond.shape[0] == b, \
                "Global conditioning `cond` must be provided with shape (B, cond_dim)"

        # temporal attention bias for all temporal attentions
        time_rel_pos_bias = self.time_rel_pos_bias(t, device=device)

        # initial conv + temporal attention
        x = self.init_conv(x)
        x = self.init_temporal(x, pos_bias=time_rel_pos_bias)

        # store initial features for final long skip connection
        r = x.clone()

        # time embedding
        t_emb = self.time_mlp(time)  # (B, time_dim)

        # classifier-free guidance masking for external cond part
        if self.has_cond:
            # decide which samples drop condition (use null embedding)
            mask = prob_mask_like((b,), null_cond_prob, device=device)
            cond_used = torch.where(mask[:, None], self.null_cond_emb.to(device), cond)
            film_cond = torch.cat([t_emb, cond_used], dim=-1)  # (B, film_cond_dim)
        else:
            # only time embedding used for FiLM
            film_cond = t_emb  # (B, time_dim)

        # focus-present mask (per-sample toggle)
        focus_present_mask = default(
            focus_present_mask,
            lambda: prob_mask_like((b,), prob_focus_present, device=device),
        )

        # -----------------------
        # Downsampling path
        # -----------------------
        skips = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            # Two FiLM-conditioned ResNet blocks
            x = block1(x, film_cond)
            x = block2(x, film_cond)
            # Spatial attention
            x = spatial_attn(x)
            # Temporal attention with relative bias and focus mask
            x = temporal_attn(
                x,
                pos_bias=time_rel_pos_bias,
                focus_present_mask=focus_present_mask,
            )
            # Save for skip connection
            skips.append(x)
            # Downsample spatially
            x = downsample(x)

        # -----------------------
        # Middle / bottleneck
        # -----------------------
        x = self.mid_block1(x, film_cond)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x,
            pos_bias=time_rel_pos_bias,
            focus_present_mask=focus_present_mask,
        )
        x = self.mid_block2(x, film_cond)

        # -----------------------
        # Upsampling path
        # -----------------------
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            # Pop corresponding skip from encoder
            skip = skips.pop()
            # Channel concat with skip connection
            x = torch.cat((x, skip), dim=1)
            # Two FiLM-conditioned ResNet blocks
            x = block1(x, film_cond)
            x = block2(x, film_cond)
            # Spatial attention
            x = spatial_attn(x)
            # Temporal attention
            x = temporal_attn(
                x,
                pos_bias=time_rel_pos_bias,
                focus_present_mask=focus_present_mask,
            )
            # Upsample spatially
            x = upsample(x)

        # -----------------------
        # Final head
        # -----------------------
        # Concatenate with early residual features
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, film_cond)
        x = self.final_conv(x)
        return x