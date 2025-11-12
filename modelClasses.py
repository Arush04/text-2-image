import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from timm.utils import ModelEmaV3
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.register_buffer("embeddings", embeddings)

    def forward(self, x, t):
        C = x.shape[1]
        embeds = self.embeddings[t, :C].to(x.device)  # match input channels dynamically
        return embeds[:, :, None, None]


# Residual Blocks
class ResBlock(nn.Module):
    def __init__(self, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.num_groups = num_groups
        self.dropout_prob = dropout_prob
        self.gnorm1 = nn.GroupNorm(num_groups=min(self.num_groups, C), num_channels=C).to(x.device)
        self.gnorm2 = nn.GroupNorm(num_groups=min(self.num_groups, C), num_channels=C).to(x.device)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1).to(x.device)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1).to(x.device)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=False)

    def forward(self, x, embeddings):
        C = x.shape[1]  # dynamic channels
        if self.gnorm1 is None or self.conv1 is None:
            self.gnorm1 = nn.GroupNorm(num_groups=min(self.num_groups, C), num_channels=C).to(x.device)
            self.gnorm2 = nn.GroupNorm(num_groups=min(self.num_groups, C), num_channels=C).to(x.device)
            self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1).to(x.device)
            self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1).to(x.device)

        x = x + embeddings[:, :C, :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


# Cross attention block
class CrossAttention(nn.Module):
    def __init__(self, C, text_dim=512, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.proj = nn.Linear(text_dim, C)
        self.attn = nn.MultiheadAttention(embed_dim=C, num_heads=num_heads, dropout=dropout_prob)
        self.norm = nn.LayerNorm(C)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(2, 0, 1)  # [HW, B, C]

        context = self.proj(context)                # [B, seq_len, C]
        context = context.permute(1, 0, 2)          # [seq_len, B, C]

        attn_out, _ = self.attn(x_flat, context, context)
        out = self.norm(x_flat + attn_out)
        out = out.permute(1, 2, 0).view(B, C, H, W)
        return out

class UnetLayer(nn.Module):
    def __init__(self, 
                 upscale: bool, 
                 attention: bool, 
                 num_groups: int, 
                 dropout_prob: float,
                 num_heads: int,
                 C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(num_groups=num_groups, dropout_prob=dropout_prob)

        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)

        self.cross_attention_layer = CrossAttention(C, num_heads=num_heads, dropout_prob=dropout_prob)
    
    def forward(self, x, embeddings, text_emb=None):
        x = self.ResBlock1(x, embeddings)
        if text_emb is not None:
            x = self.cross_attention_layer(x, context=text_emb) # cross-attention
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x

class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Attentions: List = [False, True, False, False, False, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 3,
            output_channels: int = 3,
            time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t, text_emb=None):
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings, text_emb=text_emb)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x, _ = layer(x, embeddings, text_emb=text_emb)
            x= torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]
