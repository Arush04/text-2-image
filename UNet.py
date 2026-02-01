import torch
import torch.nn as nn
import torch.nn.functional as F


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
