import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .layers import (
        SinsiodalPosEmb,
        CrossEmbedLayer
    )
from utils import (
        cast_tuple
    )

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

class Unet(nn.Module):
    """
    Base class for the UNet model
    params:
    dim = number of features at the highest spatial resolution
    dim_mults = feature mulitplier for each layer of the UNet
    channels = number of channels (colors) in the input image
    channels_out = number of channels in the output image
    cond_dim = conditional dimensionality (time conditioning)
    text_embed_dim = dimensionality from the text embeddings
    num_resnet_blocks = number of resnet blocks for each layer in the Unet. (int, tuple)
    layer_attns = weather to add self attention at end of a layer. (bool, tuple)
    layer_cross_attns = weather to add cross attention between images and text at end of a layer. (bool, tuple)
    attn_heads: Numner of attention heads. Needs to be >1, ideally 4 or 8
    lowres_cond: Whether the Unet is conditioned on low resolution images. :code:`True` for super-resolution models.
    memory_efficient: Whether to downsample at the beginning rather than end of a given layer in the U-Net. Saves memory.
    attend_at_middle: Whether to have an :class:`.minimagen.layers.Attention` at the bottleneck.    
    """
    def __init__(
            self,
            *,
            dim: int = 128,
            dim_mults: tuple = (1, 2, 4),
            channels: int = 3,
            channels_out: int = None,
            cond_dim: int = None,
            text_embed_dim=get_encoded_dim('t5_small'),
            num_resnet_blocks: Union[int, tuple] = 1,
            layer_attns: Union[bool, tuple] = True,
            layer_cross_attns: Union[bool, tuple] = True,
            attn_heads: int = 8,
            lowres_cond: bool = False,
            memory_efficient: bool = False,
            attend_at_middle: bool = False

    ):
        super.__init__()

        # Constants
        ATTN_DIM_HEAD = 64
        NUM_TIME_TOKENS = 2
        RESNET_GROUPS = 8
        # Model constants
        init_conv_to_final_conv_residual = False  # Whether to add skip connection between Unet input and output
        final_resnet_block = True  # Whether to add a final resnet block to the output of the Unet

        # TIME CONDITIONING

        # Double conditioning dimensionality for super-res models due to concatenation of low-res images
        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)
        
        # Maps time hidden state to time conditioning (non-attention)
        # Global time conditioning gets applied to all layers in the network
        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # Maps time hidden states to time tokens for main conditioning tokens (attention)
        # produces NUM_TIME_TOKENS seperate token embeddings that participate in cross-attention mechanism
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * NUM_TIME_TOKENS),
            Rearrange('b (r d) -> b r d', r=NUM_TIME_TOKENS)
        )

        # TEXT CONDITIONING
        self.norm_cond = nn.LayerNorm(cond_dim)

        # Projection from text embedding dim to cond_dim
        self.text_embed_dim = text_embed_dim
        self.text_to_cond = nn.Linear(self.text_embed_dim, cond_dim)

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

        # UNET LAYERS

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # Initial convolution that brings input images to proper number of channels for the Unet
        self.init_conv = CrossEmbedLayer(channels if not lowres_cond else channels * 2,
                                         dim_out=dim,
                                         kernel_sizes=(3, 7, 15),
                                         stride=1)

        # Determine channel numbers for UNet descent/ascent and then zip into in/out pairs
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Number of resolutions/layers in the UNet
        num_resolutions = len(in_out)

        # Cast relevant arguments to tuples (with one element for each Unet layer) if a single value rather than tuple
        # was input for the argument
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_resolutions)
        resnet_groups = cast_tuple(RESNET_GROUPS, num_resolutions)
        layer_attns = cast_tuple(layer_attns, num_resolutions)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_resolutions)
        
       
        # Validate that per-resolution configuration tuples have one value per UNet resolution level
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

        # DOWNSAMPLING LAYERS

        # Keep track of skip connection channel depths for concatenation later
        skip_connect_dims = []
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(
                zip(in_out, *layer_params)):

            is_last = ind == (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn else None

            # Potentially use Transformer encoder at end of layer
            transformer_block_klass = TransformerBlock if layer_attn else Identity

            current_dim = dim_in

            # Whether to downsample at the beginning of the layer - cuts image spatial size-length
            pre_downsample = None
            if memory_efficient:
                pre_downsample = Downsample(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # Downsample at the end of the layer if not `pre_downsample`
            post_downsample = None
            if not memory_efficient:
                post_downsample = Downsample(current_dim, dim_out) if not is_last else Parallel(
                    nn.Conv2d(dim_in, dim_out, 3, padding=1), nn.Conv2d(dim_in, dim_out, 1))

            # Create the layer
            self.downs.append(nn.ModuleList([
                pre_downsample,
                # ResnetBlock that conditions, in addition to time, on the main tokens via cross attention.
                ResnetBlock(current_dim,
                            current_dim,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups),
                # Sequence of ResnetBlocks that condition only on time
                nn.ModuleList(
                    [
                        ResnetBlock(current_dim,
                                    current_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups
                                    )
                        for _ in range(layer_num_resnet_blocks)
                    ]
                ),
                # Transformer encoder for multi-headed self attention
                transformer_block_klass(dim=current_dim,
                                        heads=attn_heads,
                                        dim_head=ATTN_DIM_HEAD),
                post_downsample,
            ]))
