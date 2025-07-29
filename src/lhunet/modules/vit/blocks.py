import torch
import torch.nn as nn
from torch.nn import functional as F

from ..cnn import *


class TransformerBlock_LKA3D(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and spatial attention
    """

    def __init__(
        self,
        epa_block: nn.Module,
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

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa = epa_block(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_heads=num_heads,
        )
        self.conv51 = UnetResBlock(
            3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch"
        )

        if dropout_rate:
            self.conv8 = nn.Sequential(
                nn.Dropout3d(dropout_rate, False),
                nn.Conv3d(hidden_size, hidden_size, 1),
            )
        else:
            self.conv8 = nn.Conv3d(hidden_size, hidden_size, 1)

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(
            0, 4, 1, 2, 3
        )  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


class ChannelAttention_LKA3D(nn.Module):
    """
    Channel attention parallel to 3d LKA
    """

    def __init__(
        self,
        hidden_size,
        num_heads=4,
        qkv_bias=False,
        use_norm_ch=False,
        use_norm_sp=False,
        use_temperature_ch=False,
        use_temperature_sp=False,
        temperature_sp_use_head_nums=False,
        lka_block=LKA3D_571_Block,
        sequential=False,
        channel_attn_drop=0,
        spatial_attn_drop=0,
        *args,
        **kwargs
    ):
        super().__init__()

        self.num_heads = num_heads
        self.sequential = sequential
        self.use_norm_ch = use_norm_ch
        self.use_norm_sp = use_norm_sp
        self.channel_attn_drop = channel_attn_drop
        self.spatial_attn_drop = spatial_attn_drop
        self.use_temperature_ch = use_temperature_ch
        self.use_temperature_sp = use_temperature_sp

        if channel_attn_drop:
            self.attn_drop_ch = nn.Dropout(channel_attn_drop)

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        if use_norm_ch:
            self.norm_ch = nn.LayerNorm(hidden_size)
        if use_temperature_ch:
            self.temperature_ch = nn.Parameter(torch.ones(num_heads, 1, 1))

        if lka_block:
            self.use_lka = True
            self.lka = lka_block(d_model=hidden_size)
            if spatial_attn_drop:
                self.attn_drop_sp = nn.Dropout(spatial_attn_drop)
            if use_temperature_sp:
                tem_sp_num = num_heads if temperature_sp_use_head_nums else 1
                self.temperature_sp = nn.Parameter(torch.ones(tem_sp_num, 1, 1))
            if use_norm_sp:
                self.norm_sp = nn.LayerNorm(hidden_size)
            if sequential:
                self.out = nn.Linear(hidden_size, hidden_size)
            else:
                self.out_sp = nn.Linear(hidden_size, int(hidden_size // 2))
                self.out_ch = nn.Linear(hidden_size, int(hidden_size // 2))
        else:
            self.use_lka = False
            self.out = nn.Linear(hidden_size, hidden_size)

    def spatial_attention(self, x, B, C, spatial_shapes):
        x_SA = self.lka(x, B, C, *spatial_shapes)
        if self.use_temperature_sp:
            x_SA *= self.temperature_sp
        if self.spatial_attn_drop:
            x_SA = self.attn_drop_sp(x_SA)
        return self.norm_sp(x_SA) if self.use_norm_sp else x_SA

    def channel_attention(self, x):
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
        if self.use_temperature_ch:
            attn_CA *= self.temperature_ch
        attn_CA = attn_CA.softmax(dim=-1)
        if self.channel_attn_drop:
            attn_CA = self.attn_drop_ch(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        return self.norm_ch(x_CA) if self.use_norm_ch else x_CA

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape
        special_shape = (H, W, D)

        # Channel Attention (ViT)
        x_CA = self.channel_attention(x)

        if not self.use_lka:
            return self.out(x_CA)

        if self.sequential:
            x = x_CA.permute(0, 2, 1).reshape(B, C, *special_shape)

        # Spatial Attention (LKA3D)
        x_SA = self.spatial_attention(x, B, C, special_shape)

        if self.sequential:
            x = self.out(x_SA)
        else:
            x_CA = self.out_ch(x_CA)
            x_SA = self.out_sp(x_SA)
            x = torch.cat((x_SA, x_CA), dim=-1)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature_ch", "temperature_sp"}


class SpatialAttention_LKA3D(nn.Module):
    """
    Spatial attention parallel to 3d LKA
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads=4,
        qkv_bias=False,
        proj_size=32,
        use_norm_spa=False,
        use_norm_lka=False,
        use_temperature_spa=False,
        use_temperature_lka=False,
        temperature_lka_use_head_nums=False,
        lka_block=LKA3D_571_Block,
        sequential=False,
        spa_attn_drop=0,
        lka_attn_drop=0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.sequential = sequential
        self.use_norm_lka = use_norm_lka
        self.use_norm_spa = use_norm_spa
        self.lka_attn_drop = lka_attn_drop
        self.spa_attn_drop = spa_attn_drop
        self.use_temperature_lka = use_temperature_lka
        self.use_temperature_spa = use_temperature_spa

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        if lka_attn_drop:
            self.attn_drop_lka = nn.Dropout(lka_attn_drop)
        if spa_attn_drop:
            self.attn_drop_spa = nn.Dropout(spa_attn_drop)

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channlka)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.lka = lka_block(d_model=hidden_size)

        if use_temperature_lka:
            tem_sp_num = num_heads if temperature_lka_use_head_nums else 1
            self.temperature_lka = nn.Parameter(torch.ones(tem_sp_num, 1, 1))
        if use_temperature_spa:
            self.temperature_spa = nn.Parameter(torch.ones(num_heads, 1, 1))

        if use_norm_spa:
            self.norm_spa = nn.LayerNorm(hidden_size)
        if use_norm_lka:
            self.norm_lka = nn.LayerNorm(hidden_size)

        if sequential:
            self.out = nn.Linear(hidden_size, hidden_size)
        else:
            self.out_spa = nn.Linear(hidden_size, int(hidden_size // 2))
            self.out_lka = nn.Linear(hidden_size, int(hidden_size // 2))

    def spatial_lka_attention(self, x, special_shape):
        if self.sequential:
            B, C, H, W, D = x.shape
        else:
            B, N, C = x.shape

        x_LKA = self.lka(x, B, C, *special_shape)
        if self.use_temperature_lka:
            x_LKA *= self.temperature_lka
        if self.lka_attn_drop:
            x_LKA = self.attn_drop_lka(x_LKA)
        return self.norm_lka(x_LKA) if self.use_norm_lka else x_LKA

    def spatial_vit_attention(self, x):
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

        attn_SA = query.permute(0, 1, 3, 2) @ k_projected
        if self.use_temperature_spa:
            attn_SA *= self.temperature_spa
        attn_SA = attn_SA.softmax(dim=-1)
        if self.spa_attn_drop:
            attn_SA = self.attn_drop_spa(attn_SA)
        x_SA = (
            (attn_SA @ v_SA_projected.transpose(-2, -1))
            .permute(0, 3, 1, 2)
            .reshape(B, N, C)
        )

        return self.norm_spa(x_SA) if self.use_norm_spa else x_SA

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape
        special_shape = (H, W, D)

        # Spatial Attention (ViT)
        x_SA = self.spatial_vit_attention(x)

        if self.sequential:
            x = x_SA.permute(0, 2, 1).reshape(
                B, C, *special_shape
            )  # B N C --> B C N --> B C H W D

        # Spatial Attention (LKA3D)
        x_LKA = self.spatial_lka_attention(x, special_shape=(H, W, D))

        if self.sequential:
            x = self.out(x_LKA)
        else:
            x_SA = self.out_spa(x_SA)
            x_LKA = self.out_lka(x_LKA)
            x = torch.cat((x_SA, x_LKA), dim=-1)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature_lka", "temperature_spa"}


# ================================================


TransformerBlock_LKA3D_571 = partial(TransformerBlock_LKA3D, epa_block=LKA3D_571_Block)
TransformerBlock_LKA3D_5731 = partial(
    TransformerBlock_LKA3D, epa_block=LKA3D_5731_Block
)
TransformerBlock_DLKA3D_single = partial(
    TransformerBlock_LKA3D, epa_block=DLKA3D_Static_Block
)


####################(Sequential)#####################
SpatialAttention_DLKA3D_sequential = partial(
    SpatialAttention_LKA3D,
    use_norm_spa=True,
    use_norm_lka=True,
    use_temperature_spa=True,
    use_temperature_lka=False,
    temperature_lka_use_head_nums=False,
    lka_block=DLKA3D_Block,
    sequential=True,
)

TransformerBlock_DLKA3D_SpatialSequential = partial(
    TransformerBlock_LKA3D, epa_block=SpatialAttention_DLKA3D_sequential
)


ChannelAttention_DLKA3D_sequential = partial(
    ChannelAttention_LKA3D,
    use_norm_ch=True,
    use_norm_sp=True,
    use_temperature_ch=True,
    use_temperature_sp=False,
    temperature_sp_use_head_nums=False,
    lka_block=DLKA3D_Block,
    sequential=True,
)

TransformerBlock_DLKA3D_ChannelSequential = partial(
    TransformerBlock_LKA3D, epa_block=ChannelAttention_DLKA3D_sequential
)


ChannelAttention_ONLY = partial(
    ChannelAttention_LKA3D,
    use_norm_ch=True,
    use_temperature_ch=True,
    lka_block=None,
)

TransformerBlock_3D_ChannelAtt_ONLY = partial(
    TransformerBlock_LKA3D, epa_block=ChannelAttention_ONLY
)


####################(Parallel)#####################
SpatialAttention_LKA3D_parallel = partial(
    SpatialAttention_LKA3D,
    use_norm_spa=False,
    use_norm_lka=False,
    use_temperature_spa=True,
    use_temperature_lka=False,
    temperature_lka_use_head_nums=False,
    lka_block=LKA3D_571_Block,
    sequential=False,
    spa_attn_drop=0,
    lka_attn_drop=0,
)

TransformerBlock_LKA3D_SpatialParallel = partial(
    TransformerBlock_LKA3D, epa_block=SpatialAttention_LKA3D_parallel
)


SpatialAttention_DLKA3D_parallel = partial(
    SpatialAttention_LKA3D,
    use_norm_spa=False,
    use_norm_lka=False,
    use_temperature_spa=True,
    use_temperature_lka=False,
    temperature_lka_use_head_nums=False,
    lka_block=DLKA3D_Block,
    sequential=False,
    spa_attn_drop=0,
    lka_attn_drop=0,
)

TransformerBlock_DLKA3D_SpatialParallel = partial(
    TransformerBlock_LKA3D, epa_block=SpatialAttention_DLKA3D_parallel
)


ChannelAttention_LKA3D_normParallel = partial(
    ChannelAttention_LKA3D,
    lka_block=LKA3D_571_Block,
    use_temperature_ch=True,
    use_temperature_sp=True,
    use_norm_sp=True,
    use_norm_ch=True,
    sequential=False,
)

TransformerBlock_LKA3D_ChannelNormParallel = partial(
    TransformerBlock_LKA3D, epa_block=ChannelAttention_LKA3D_normParallel
)


ChannelAttention_DLKA3D_parallel = partial(
    ChannelAttention_LKA3D,
    lka_block=DLKA3D_Static_Block,
    use_temperature_ch=True,
    use_temperature_sp=False,
    use_norm_sp=False,
    use_norm_ch=False,
    sequential=False,
)

TransformerBlock_DLKA3D_ChannelParallel = partial(
    TransformerBlock_LKA3D, epa_block=ChannelAttention_DLKA3D_parallel
)


ChannelAttention_LKA3D_tempsphead_parallel = partial(
    ChannelAttention_LKA3D,
    lka_block=LKA3D_571_Block,
    use_temperature_ch=True,
    use_temperature_sp=True,
    temperature_sp_use_head_nums=True,
    use_norm_sp=False,
    use_norm_ch=False,
    sequential=False,
)

TransformerBlock_LKA3D_ChannelParallel_tempsphead = partial(
    TransformerBlock_LKA3D, epa_block=ChannelAttention_LKA3D_tempsphead_parallel
)


# -------------------------(Others)--------------------------
class EPA(nn.Module):
    """
    Efficient Paired Attention Block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        proj_size,
        num_heads=4,
        qkv_bias=False,
        channel_attn_drop=0.1,
        spatial_attn_drop=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)
        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (
            q_shared.permute(0, 1, 3, 2) @ k_shared_projected
        ) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (
            (attn_SA @ v_SA_projected.transpose(-2, -1))
            .permute(0, 3, 1, 2)
            .reshape(B, N, C)
        )

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)

        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature", "temperature2"}


TransformerBlock_3D_EPA = partial(TransformerBlock_LKA3D, epa_block=EPA)


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, N, C]
    output ->   [B, N, C]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, input_size, hidden_size, head_count=4, qkv_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_count = head_count

        self.key_lin = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.query_lin = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.value_lin = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.reprojection = nn.Linear(hidden_size, hidden_size)
        self.temperature = nn.Parameter(torch.ones(head_count, 1, 1))

    def forward(self, input_):
        B, N, C = input_.shape

        queries = self.query_lin(input_).permute(0, 2, 1)

        keys = self.key_lin(input_).permute(0, 2, 1)

        values = self.value_lin(input_).permute(0, 2, 1)

        head_key_channels = self.hidden_size // self.head_count
        head_value_channels = self.hidden_size // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(
                keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2
            )

            query = F.softmax(
                queries[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )

            value = values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            context = key @ value.transpose(1, 2)  # dk*dv

            attended_value = context.transpose(1, 2) @ query  # n*dv

            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)

        attention = self.reprojection(aggregated_values.transpose(1, 2))

        return attention

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature", "temperature2"}


TransformerBlock_3D_EA = partial(TransformerBlock_LKA3D, epa_block=EfficientAttention)


#########################
#
# 3D LKA with SE Module
#
#########################
class SEModule(nn.Module):
    """SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """

    def __init__(self, channels, rd_ratio=1.0 / 4, rd_channels=None, bias=True):
        super(SEModule, self).__init__()
        if rd_channels is None:
            rd_channels = int(channels * rd_ratio)
        self.fc1 = nn.Conv3d(channels, rd_channels, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()
        print("Using SE Module")

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)  # B C H W D --> B C 1 1 1
        x_se = self.fc1(x_se)  # B C 1 1 1
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class TransformerBlock_3D_SE(nn.Module):
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
        self.se = SEModule(channels=hidden_size, rd_ratio=1.0 / 4)
        self.LKA_block = LKA3D_571_Block(d_model=hidden_size)
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

        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)  # B N C --> B C N --> B C H W D
        x = self.se(x)
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        attn = x + self.gamma * self.LKA_block(self.norm(x), B, C, H, W, D)
        attn_skip = attn.reshape(B, H, W, D, C).permute(
            0, 4, 1, 2, 3
        )  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x
