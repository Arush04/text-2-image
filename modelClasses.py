import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
import random
from layers import CrossEmbedLayer 
from utils import cast_tuple

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
            Channels: List = [64, 128, 256, 512],
            attentions: List = [False, True, False, False, False, True, False, True],
            cross_attention: List = [True, False, True, True, True, False, True, False],
            # Upscales: List = [False, False, False, True, True, True],
            num_resnet_blocks = 1,
            num_groups: int = 32,
            text_embed_dim: int = 512,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 3,
            output_channels: int = 3,
            time_steps: int = 1000):
        super().__init__()
        
        # Contants
        ATTN_HEADS = 4
        ATTN_DIM_HEAD = 64
        NUM_TIME_TOKENS = 2
        RESNET_GROUPS = 8

        self.num_layers = len(Channels)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        for i in range(self.num_layers):
            
            dim = Channels[i]
            # Time Conditioning
            time_cond_dim = dim*4
            
            # Map time to time hidden state
            self.to_time_hiddens = nn.Sequential(
                embeddings,
                nn.Linear(dim, time_cond_dim),
                nn.SiLU()
            )

            # Map time hidden states to time conditioning (non-attention)
            self.to_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )
            
            # Map time hidden states to time time tokens for main conditioning tokens (attention)
            self.to_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, dim*NUM_TIME_TOKENS),
                rearrange('b (r d) -> b r d', r=NUM_TIME_TOKENS)
            )
            
            # Text Conditioning
            self.norm_cond = nn.LayerNorm(dim)

            # Projection from text embedding dim to cond_dim
            self.text_embed_dim = text_embed_dim
            self.text_to_cond = nn.Linear(self.text_embed_dim, cond_dim)

            # Create null tokens for classifier-free guidance. See
            max_text_len = 256
            self.max_text_len = max_text_len
            self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
            self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

            # For injecting text information into time conditioning (non-attention)
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim)
            )
            # Initial convolution that brings input images to proper number of channels for the Unet
            self.init_conv = CrossEmbedLayer(channels if not lowres_cond else channels * 2,
                             dim_out=dim,
                             kernel_sizes=(2, 6, 12),
                             stride=2)
            
            dims = [Channels[0], *Channels]
            # in/out pair based on the proivded Channel sizes
            in_out = list(zip(dims[:-1], dims[1:]))
            # Number of resolutions/layers in the UNet
            num_resolutions = len(in_out)
            
            num_resnet_blocks = cast_tuple(num_resnet_blocks, num_resolutions)
            resnet_groups = cast_tuple(RESNET_GROUPS, num_resolutions)
            layer_attns = cast_tuple(attentions[i], num_resolutions)
            layer_cross_attns = cast_tuple(cross_attention[i], num_resolutions)

            # Make sure relevant tuples have one elt for each layer in the UNet (if tuples rather than single values passed
            #   in as arguments)
            assert all(
                [layers == num_resolutions for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

            # Scale for resnet skip connections
            self.skip_connect_scale = 2 ** -0.5

            # Downsampling and Upsampling modules of the Unet
            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])

            # Parameter lists for downsampling and upsampling trajectories
            layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_cross_attns]
            reversed_layer_params = list(map(reversed, layer_params))
            
            # Downsampling Layers
            #Keep track of skip connections channels depth
            skip_connection_dims = []
            
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

