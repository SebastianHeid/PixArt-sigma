import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from diffusion.model.builder import MODELS
from diffusion.model.nets.PixArt import PixArt, get_2d_sincos_pos_embed
from diffusion.model.nets.PixArt_blocks import (
    AttentionKVCompress,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    t2i_modulate,
)
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple



class PixArtMSBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        input_size=None,
        sampling=None,
        sr_ratio=1,
        qk_norm=False,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionKVCompress(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            sampling=sampling,
            sr_ratio=sr_ratio,
            qk_norm=qk_norm,
            **block_kwargs,
        )
        self.cross_attn = MultiHeadCrossAttention(
            hidden_size, num_heads, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

    def forward(self, x, y, t, mask=None, HW=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x = x + self.drop_path(
            gate_msa
            * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW)
        )
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(
            gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        )

        return x