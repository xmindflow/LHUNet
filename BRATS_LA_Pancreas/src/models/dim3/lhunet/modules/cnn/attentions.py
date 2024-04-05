from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from ..deform_conv import DeformConvPack, DeformConvPack_Depth
from ...blocks.cnn import UnetResBlock


class LKA3D_571(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 32 or dim == 64 or True:
            kernel_dwd = 7
            dilation_dwd = 3
            padding_dwd = 9
            kernel_dw = 5
            padding_dw = 2
        elif dim == 128:
            kernel_dwd = 5
            dilation_dwd = 3
            padding_dwd = 6
            kernel_dw = 5
            padding_dw = 2
        elif dim == 256:
            kernel_dwd = 3
            dilation_dwd = 2
            padding_dwd = 2
            kernel_dw = 3
            padding_dw = 1
        else:
            raise ValueError("Unknown dim: {}".format(dim))

        self.conv5 = nn.Conv3d(
            dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim
        )
        self.conv_spatial = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_dwd,
            stride=1,
            padding=padding_dwd,
            groups=dim,
            dilation=dilation_dwd,
        )
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv5(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class LKA3D_5731(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv5 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_lk = nn.Conv3d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv3 = nn.Conv3d(dim, dim, 3, stride=1, padding=1)
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv5(x)
        attn = self.conv_lk(attn)
        attn = self.conv3(attn)
        attn = self.conv1(attn)
        return u * attn


class DLKA3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim < 33:
            kernel_dwd = 7
            dilation_dwd = 3
            padding_dwd = 9
            kernel_dw = 5
            padding_dw = 2
        elif dim < 65:
            kernel_dwd = 5
            dilation_dwd = 3
            padding_dwd = 6
            kernel_dw = 5
            padding_dw = 2
        elif dim < 512:
            kernel_dwd = 3
            dilation_dwd = 2
            padding_dwd = 2
            kernel_dw = 3
            padding_dw = 1
        else:
            raise ValueError("Unknown dim: {}".format(dim))

        self.conv0 = nn.Conv3d(
            dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim
        )
        self.conv_spatial = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_dwd,
            stride=1,
            padding=padding_dwd,
            groups=dim,
            dilation=dilation_dwd,
        )
        self.deform_conv = DeformConvPack(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
        )
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = attn.contiguous()
        attn = self.deform_conv(attn)
        attn = self.conv1(attn)
        return u * attn


class DLKA3D_Static(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim in [32, 64] or True:
            kernel_dwd = 7
            dilation_dwd = 3
            padding_dwd = 9
            kernel_dw = 5
            padding_dw = 2
        elif dim == 128:
            kernel_dwd = 5
            dilation_dwd = 3
            padding_dwd = 6
            kernel_dw = 5
            padding_dw = 2
        elif dim == 256:
            kernel_dwd = 3
            dilation_dwd = 2
            padding_dwd = 2
            kernel_dw = 3
            padding_dw = 1
        else:
            raise ValueError("Unknown dim: {}".format(dim))

        self.conv0 = nn.Conv3d(
            dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim
        )
        self.conv_spatial = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_dwd,
            stride=1,
            padding=padding_dwd,
            groups=dim,
            dilation=dilation_dwd,
        )
        self.deform_conv = DeformConvPack(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
        )
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = attn.contiguous()
        attn = self.deform_conv(attn)
        attn = self.conv1(attn)
        return u * attn


class DLKA3D_Block_onTensor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DLKA3D(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class LKA3D_Block(nn.Module):
    def __init__(self, d_model, lka_module=LKA3D_5731):
        super().__init__()
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = lka_module(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        if x.shape != 5:  # [B, N, C]
            x = x.permute(0, 2, 1).reshape(
                B, C, H, W, D
            )  # B N C --> B C N --> B C H W D
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x


DLKA3D_Block = partial(LKA3D_Block, lka_module=DLKA3D)
DLKA3D_Static_Block = partial(LKA3D_Block, lka_module=DLKA3D_Static)
LKA3D_571_Block = partial(LKA3D_Block, lka_module=LKA3D_571)
LKA3D_5731_Block = partial(LKA3D_Block, lka_module=LKA3D_5731)


###########################
#
# Transformer Block 2D deformable convolution
#
###########################
import torchvision


class DeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        groups,
        kernel_size=(3, 3),
        padding=1,
        stride=1,
        dilation=1,
        bias=True,
    ):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kernel_size[0] * kernel_size[1],
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=False,
        )

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(
            dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)  # B N C --> B C N --> B C H W D
        shorcut = x.clone()
        x_copy = x.clone()

        # Shape B C H W D
        # Extract Depths
        for i in range(x.size(-1)):
            x_temp = x[:, :, :, :, i]
            x_temp = self.proj_1(x_temp)
            x_temp = self.activation(x_temp)
            x_temp = self.spatial_gating_unit(x_temp)
            x_temp = self.proj_2(x_temp)
            x_copy[:, :, :, :, i] = x_temp

        x = x_copy + shorcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)  # B N C
        return x


class TransformerBlock_2Dsingle(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        # print("Using LKA Attention")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = deformable_LKA_Attention(d_model=hidden_size)
        self.conv51 = UnetResBlock(
            3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch"
        )
        self.conv8 = nn.Sequential(
            nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1)
        )

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        y = self.norm(x)
        z = self.epa_block(y, B, C, H, W, D)
        attn = x + self.gamma * z
        attn_skip = attn.reshape(B, H, W, D, C).permute(
            0, 4, 1, 2, 3
        )  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x
