import torch
import torch.nn as nn
import math
from einops import rearrange

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


class CrossEmbedLayer(nn.Module):
    '''
    Module that performs cross embedding on an input image (essentially an Inception module) which maintains channel
        depth.
    Essentially we are running multiple kernels of differente sizes on the same input to get more information.
    E.g. If input a 64x64 image with 128 channels and use kernel_sizes = (3, 7, 15) and stride=1, then 3 convolutions
        will be performed:

        1: 64 filters, (3x3) kernel, stride=(1x1), padding=(1x1) -> 64x64 output
        2: 32 filters, (7x7) kernel, stride=(1x1), padding=(3x3) -> 64x64 output
        3: 32 filters, (15x15) kernel, stride=(1x1), padding=(7x7) -> 64x64 output

        Concatenate them for a resulting 64x64 image with 128 output channels
    '''

    def __init__(
            self,
            dim_in: int,
            kernel_sizes: tuple[int, ...],
            dim_out: int = None,
            stride: int = 2
    ):
        """
        params:
        dim_in: Number of channels in the input image.
        kernel_sizes: Tuple of kernel sizes to use for convolutions.
        dim_out: Number of channels in output image. Defaults to `dim_in`.
        stride: Stride of convolutions.
        """
        super().__init__()
        # Ensures stride and all kernels are either all odd or all even
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])

        # Set output dimensionality to be same as input if not provided
        dim_out = default(dim_out, dim_in)

        # Sort the kernels by size and determine number of kernels
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # The following code determines the number of filters for each kernel and the sum of filters of each kernel = dim_out and list in descending order
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        # Create the convolution objects
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Perform each convolution and then concatenate the results along the channel dim.
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)

class SinsiodalPosEmb(nn.Module):
    """
    Generates sinusiodal postional embedding tensor, here position corresponds to time. For more on sinusiodal postional embeddings refere here: https://stats.stackexchange.com/questions/409629/what-is-the-intuition-behind-the-positional-cosine-encoding-in-the-transformer-n/409635#409635?newreg=f7b48e085190413ab4933bb6a9301019
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1) # 10000 here is for scale as done in DDPM scheduler
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
