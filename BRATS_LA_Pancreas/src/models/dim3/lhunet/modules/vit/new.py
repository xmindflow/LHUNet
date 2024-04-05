import torch
import torch.nn as nn
from torch.nn import functional as F


from ..cnn import *


class AttBlock_ViT_Parallel_DLKA3D(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and spatial attention
    """

    def __init__(
        self,
        vit_block: nn.Module,
        lka_block: nn.Module,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *args,
        **kwargs
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
        # print(f"Using {epa_block}")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.bnorm = nn.BatchNorm3d(hidden_size)

        self.gamma = nn.Parameter(torch.ones(hidden_size, 1, 1, 1), requires_grad=True)
        self.attn = vit_block(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate / 2,
        )

        self.delta = nn.Parameter(torch.ones(hidden_size, 1, 1, 1), requires_grad=True)
        self.lka = lka_block(d_model=hidden_size)

        self.conv3 = UnetResBlock(
            3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch"
        )
        if dropout_rate:
            self.conv1 = nn.Sequential(
                nn.Dropout3d(dropout_rate, False),
                nn.Conv3d(hidden_size, hidden_size, 1),
            )
        else:
            self.conv1 = nn.Conv3d(hidden_size, hidden_size, 1)

        self.pos_embed = nn.Parameter(1e-6 + torch.zeros(1, input_size, hidden_size))

    def vit_attn(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        x = x + self.pos_embed
        attn = self.attn(self.norm(x), B, C, H, W, D)
        return attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)

    def forward(self, x):
        x_lka = self.delta * self.lka(x)
        x_vit = self.gamma * self.vit_attn(x)
        # x = x+ x_lka + x_vit
        x = x * (1 + x_lka + x_vit)
        x = self.bnorm(x)
        x = x + self.conv1(self.conv3(x))
        return x


class ChannelAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=4,
        qkv_bias=False,
        use_norm=False,
        use_temperature=False,
        dropout=0,
        *args,
        **kwargs
    ):
        super().__init__()

        self.num_heads = num_heads
        self.use_norm = use_norm
        self.use_dropout = dropout
        self.use_temperature = use_temperature

        if dropout:
            self.dropout = nn.Dropout(dropout)

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out = nn.Linear(hidden_size, hidden_size)

        if use_norm:
            self.norm = nn.LayerNorm(hidden_size)
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def vit_attention(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, v_CA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        attn_CA = query @ key.transpose(-2, -1)
        if self.use_temperature:
            attn_CA *= self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        if self.use_dropout:
            attn_CA = self.dropout(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        return self.norm(x_CA) if self.use_norm else x_CA

    def forward(self, x, B_in, C_in, H, W, D):
        x = self.vit_attention(x)
        return self.out(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature"}


class SpatialAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads=4,
        qkv_bias=False,
        proj_size=8**3,
        use_norm=False,
        use_temperature=False,
        dropout=0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_norm = use_norm
        self.use_dropout = dropout
        self.use_temperature = use_temperature

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channlka)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out = nn.Linear(hidden_size, hidden_size)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        if use_norm:
            self.norm = nn.LayerNorm(hidden_size)

    def vit_attention(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, v_SA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_projected = self.E(key)
        v_SA_projected = self.F(v_SA)

        query = torch.nn.functional.normalize(query, dim=-1)
        # key = torch.nn.functional.normalize(key, dim=-1)

        attn_SA = query.permute(0, 1, 3, 2) @ k_projected
        if self.use_temperature:
            attn_SA *= self.temperature
        attn_SA = attn_SA.softmax(dim=-1)
        if self.use_dropout:
            attn_SA = self.dropout(attn_SA)
        x_SA = (
            (attn_SA @ v_SA_projected.transpose(-2, -1))
            .permute(0, 3, 1, 2)
            .reshape(B, N, C)
        )
        return self.norm(x_SA) if self.use_norm else x_SA

    def forward(self, x, B_in, C_in, H, W, D):
        x = self.vit_attention(x)
        x = self.out(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature"}


####################(Sequential)#####################
HybAttn_DLKA3D_Parallel_SpatialViT = partial(
    AttBlock_ViT_Parallel_DLKA3D,
    vit_block=partial(SpatialAttention, use_norm=True, use_temperature=True),
    lka_block=DLKA3D_Block_onTensor,
)

HybAttn_DLKA3D_Parallel_ChannelViT = partial(
    AttBlock_ViT_Parallel_DLKA3D,
    vit_block=partial(ChannelAttention, use_norm=True, use_temperature=True),
    lka_block=DLKA3D_Block_onTensor,
)
