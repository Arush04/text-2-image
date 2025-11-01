import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
import random
import math
import pdb
from torch import device
from torch.utils.checkpoint import checkpoint

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int, dropout_prob: float, embed_dim: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        self.embed_proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x, embeddings):
        emb_proj = self.embed_proj(embeddings)
        residual = self.skip_proj(x)
        x = self.conv1(x)
        x = self.relu(self.gnorm1(x + emb_proj))
        x = self.dropout(x)
        x = self.conv2(self.gnorm2(x))
        return x + residual

class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

class UnetLayer(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 upscale: bool, 
                 attention: bool, 
                 num_groups: int, 
                 dropout_prob: float,
                 num_heads: int,
                 embed_dim: int,
                 is_last: bool = False):
        super().__init__()
        self.ResBlock1 = ResBlock(in_channels, out_channels, num_groups, dropout_prob, embed_dim)
        self.ResBlock2 = ResBlock(out_channels, out_channels, num_groups, dropout_prob, embed_dim)
        if upscale and not is_last:
            self.conv = nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=4, stride=2, padding=1)
            self.next_channels = out_channels//2
        elif upscale and is_last:
            self.conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            self.next_channels = out_channels
        else:
            self.conv = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1)
            self.next_channels = out_channels*2
        if attention:
            self.attention_layer = Attention(out_channels, num_heads=num_heads, dropout_prob=dropout_prob)
        else:
            self.attention_layer = None

    def forward(self, x, embeddings):
        x = checkpoint(self.ResBlock1, x, embeddings, use_reentrant=False)
        if self.attention_layer is not None:
            x = checkpoint(self.attention_layer, x, use_reentrant=False)
        x = checkpoint(self.ResBlock2, x, embeddings, use_reentrant=False)
        return self.conv(x), x

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int, device: torch.device = None):
        super().__init__()
        position = torch.arange(time_steps, device=device).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2, device=device).float() * -(math.log(10000.0) / embed_dim)
        )
        embeddings = torch.zeros(time_steps, embed_dim, device=device, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.register_buffer("embeddings", embeddings, persistent=False)

    def forward(self, t):
        embeds = self.embeddings[t]
        return embeds[:, :, None, None]

class UNET(nn.Module):
    def __init__(self,
                 Channels: List = [32, 64, 128, 128, 64, 32],
                 Attentions: List = [False, False, True, True, False, False],
                 Upscales: List = [False, False, False, True, True, True],
                 num_groups: int = 8,
                 dropout_prob: float = 0.1,
                 num_heads: int = 8,
                 input_channels: int = 3,  # RGB input
                 output_channels: int = 3,  # RGB output
                 device: str = 'cuda',
                 time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels), device=device)

        # Encoder for 32x32 images
        self.enc_layers = nn.ModuleList()
        in_ch = Channels[0]
        for i in range(self.num_layers//2):
            out_ch = Channels[i]
            self.enc_layers.append(
                UnetLayer(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    upscale=False,
                    attention=Attentions[i],
                    num_groups=num_groups,
                    dropout_prob=dropout_prob,
                    num_heads=num_heads,
                    embed_dim=max(Channels)
                )
            )
            in_ch = out_ch * 2  # Downsampling doubles channels

        # Decoder
        self.dec_layers = nn.ModuleList()
        for i in range(self.num_layers//2, self.num_layers):
            skip_ch = Channels[self.num_layers-1-i]
            in_ch = in_ch + skip_ch
            out_ch = Channels[i]
            is_last = (i == self.num_layers - 1)
            self.dec_layers.append(
                UnetLayer(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    upscale=True,
                    attention=Attentions[i],
                    num_groups=num_groups,
                    dropout_prob=dropout_prob,
                    num_heads=num_heads,
                    embed_dim=max(Channels),
                    is_last=is_last
                )
            )
            in_ch = out_ch if is_last else out_ch // 2

        # Final convs
        self.late_conv = nn.Conv2d(Channels[-1], Channels[-1], kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(Channels[-1], output_channels, kernel_size=1)

    def forward(self, x, t):
        # Input x is expected to be of shape [batch, 3, 32, 32]
        x = self.shallow_conv(x)
        embeddings = self.embeddings(t)
        residuals = []

        # Encoder: downsampling reduces spatial size (32x32 -> 16x16 -> 8x8)
        for layer in self.enc_layers:
            x, r = layer(x, embeddings)
            residuals.append(r)

        # Decoder: upsampling restores spatial size (8x8 -> 16x16 -> 32x32)
        for i, layer in enumerate(self.dec_layers):
            skip = residuals[-(i+1)]
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x, _ = layer(x, embeddings)

        x = self.relu(self.late_conv(x))
        x = self.output_conv(x)
        # Output shape: [batch, 3, 32, 32]
        return x
