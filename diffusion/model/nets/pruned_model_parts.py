import math
from copy import deepcopy
from typing import List, Literal, Optional, Union

import torch
import torch as th
import torch.nn as nn
import xformers.ops
from diffusion.model.nets.PixArt_blocks import AttentionKVCompress
from diffusion.model.nets.PixArtMS import PixArtMSBlock
from diffusion.model.utils import decompose_linear_to_svd, grasp_decompose_linear_to_svd
from timm.models.vision_transformer import Attention as Attention_


class SVDLinear(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], sigma_fuse: Literal["UV", "U", "V"] = "UV"):
        '''
        **__Args__:**
            U: Left Singular Vectors after rank truncation, which is shape of [rank, out_features]
            S: Diagonal Matrix of singular values, which is shape of [rank, rank]
            Vh: Right Singular Vectors after rank truncation, which is shape of [in_features, rank]
            bias: bias
        '''
        super(SVDLinear, self).__init__()
        
        in_features = Vh.shape[1]
        out_features = U.shape[0]
        hidden_size = S.shape[0]

        self.InLinear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)
        self.OutLinear = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True if bias is not None else False)

        if bias is not None:
            self.OutLinear.bias.data = bias
        
        if sigma_fuse == "UV":
            self.InLinear.weight.data = Vh.mul(S.sqrt().view(-1, 1)).contiguous()
            self.OutLinear.weight.data = U.mul(S.sqrt()).contiguous()
        elif sigma_fuse == "U":
            self.InLinear.weight.data = Vh.contiguous()
            self.OutLinear.weight.data = U.mul(S).contiguous()
        elif sigma_fuse == "V":
            self.InLinear.weight.data = Vh.mul(S.view(-1, 1)).contiguous()
        else:
            raise ValueError(f"value of sigma_fuse {sigma_fuse} not support")
    
    def forward(self, x: torch.Tensor):
        output = self.OutLinear(self.InLinear(x))
        return output


class GRASPLayer(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], compression_ratio: Optional[float]):
        super(GRASPLayer, self).__init__()
        self.U = nn.Parameter(U.clone().detach().requires_grad_(False))
        self.S = nn.Parameter(S.clone().detach().requires_grad_(True))
        self.Vh = nn.Parameter(Vh.clone().detach().requires_grad_(False))

        self.in_features = self.Vh.shape[1]
        self.out_features = self.U.shape[0]

        self.bias =  nn.Parameter(bias.clone().detach().requires_grad_(False))
        self.compression_ratio = compression_ratio

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        sigma = torch.diag(self.S)
        W_reconstructed =  torch.mm(self.U, torch.mm(sigma, self.Vh))
        return (torch.mm(x.reshape(-1, x.shape[-1]), W_reconstructed.t()) + self.bias).view(b, s, -1)
    
   
   
   
    
class GraspdMLP(nn.Module):
    def __init__(self, block: PixArtMSBlock):
        super().__init__()
        
        original_layer = block.mlp.fc1
        W = block.mlp.fc1.weight.data
        bias = original_layer.bias.data if original_layer.bias is not None else None
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.fc1 = GRASPLayer(U=U, S=S, Vh=Vh, bias=bias, compression_ratio=0.5)
        
        original_layer2 = block.mlp.fc2
        W2 = block.mlp.fc2.weight.data
        bias2 = original_layer2.bias.data if original_layer2.bias is not None else None
        U2, S2, Vh2 = torch.linalg.svd(W2, full_matrices=False)
        self.fc2 = GRASPLayer(U=U2, S=S2, Vh=Vh2, bias=bias2, compression_ratio=0.5)
        
        self.approx_gelu = nn.GELU(approximate="tanh")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.approx_gelu(x)
        x = self.fc2(x)
        return x
    
class GraspAttentionKVCompress(nn.Module):
    """Multi-head Attention block with KV token compression and qk norm."""

    def __init__(
        self,
        block: PixArtMSBlock
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.sampling=block.attn.sampling    # ['conv', 'ave', 'uniform', 'uniform_every']
        self.sr_ratio = block.attn.sr_ratio
        self.attn_drop = block.attn.attn_drop
        self.proj = block.attn.proj
        self.proj_drop = block.attn.proj_drop
        self.sampling = block.attn.sampling
        self.qk_norm = block.attn.qk_norm
        
        if self.sr_ratio > 1 and self.sampling == 'conv':
            # Avg Conv Init.
            self.sr = block.attn.sr
            self.sr.weight.data.fill_(1/block.attn.sr_ratio**2)
            self.sr.bias.data.zero_()
            self.norm = block.attn.norm
        if self.qk_norm:
            self.q_norm =block.attn.q_norm
            self.k_norm = block.attn.k_norm
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            
        self.num_heads = block.attn.num_heads

        q_linear,k_linear,v_linear = self.split_qkv_linear(block.attn.qkv)
     
        
        original_q_layer = q_linear
        W_q = original_q_layer.weight.data
        bias_q = original_q_layer.bias.data if original_q_layer.bias is not None else None
        U_q, S_q, Vh_q = torch.linalg.svd(W_q, full_matrices=False)
        self.q_proj = GRASPLayer(U=U_q, S=S_q, Vh=Vh_q, bias=bias_q, compression_ratio=0.5)
        
        
        
        original_k_layer = k_linear
        W_k = original_k_layer.weight.data
        bias_k = original_k_layer.bias.data if original_k_layer.bias is not None else None
        U_k, S_k, Vh_k = torch.linalg.svd(W_k, full_matrices=False)

        # Ersetzt k_linear durch den GRASPLayer
        self.k_proj = GRASPLayer(U=U_k, S=S_k, Vh=Vh_k, bias=bias_k, compression_ratio=0.5)
        
        original_v_layer = v_linear
        W_v = original_v_layer.weight.data
        bias_v = original_v_layer.bias.data if original_v_layer.bias is not None else None
        U_v, S_v, Vh_v = torch.linalg.svd(W_v, full_matrices=False)

        # Ersetzt v_linear durch den GRASPLayer
        self.v_proj = GRASPLayer(U=U_v, S=S_v, Vh=Vh_v, bias=bias_v, compression_ratio=0.5)

        
    def downsample_2d(self, tensor, H, W, scale_factor, sampling=None):
        if sampling is None or scale_factor == 1:
            return tensor
        B, N, C = tensor.shape

        if sampling == 'uniform_every':
            return tensor[:, ::scale_factor], int(N // scale_factor)

        tensor = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == 'ave':
            tensor = F.interpolate(
                tensor, scale_factor=1 / scale_factor, mode='nearest'
            ).permute(0, 2, 3, 1)
        elif sampling == 'uniform':
            tensor = tensor[:, :, ::scale_factor, ::scale_factor].permute(0, 2, 3, 1)
        elif sampling == 'conv':
            tensor = self.sr(tensor).reshape(B, C, -1).permute(0, 2, 1)
            tensor = self.norm(tensor)
        else:
            raise ValueError
   
    def split_qkv_linear(self, qkv_layer: nn.Linear):
        in_dim = qkv_layer.in_features   # = dim
        out_dim = qkv_layer.out_features # = 3*dim
        dim = out_dim // 3

        W = qkv_layer.weight.data        # [3*dim, dim]
        b = qkv_layer.bias.data if qkv_layer.bias is not None else None

        # slice weights
        W_q, W_k, W_v = W[:dim, :], W[dim:2*dim, :], W[2*dim:, :]
        b_q = b[:dim] if b is not None else None
        b_k = b[dim:2*dim] if b is not None else None
        b_v = b[2*dim:] if b is not None else None

        # build independent linears
        q = nn.Linear(in_dim, dim, bias=b_q is not None)
        k = nn.Linear(in_dim, dim, bias=b_k is not None)
        v = nn.Linear(in_dim, dim, bias=b_v is not None)

        # copy weights/bias
        q.weight.data.copy_(W_q);  k.weight.data.copy_(W_k);  v.weight.data.copy_(W_v)
        if b_q is not None:
            q.bias.data.copy_(b_q); k.bias.data.copy_(b_k); v.bias.data.copy_(b_v)

        return q, k, v

    
    def forward(self, x, mask=None, HW=None, block_id=None):
        B, N, C = x.shape
        new_N = N
        if HW is None:
            H = W = int(N ** 0.5)
        else:
            H, W = HW
        
        # using the lower rank matrices 
        q = self.q_proj(x)
        v = self.v_proj(x)
        k = self.k_proj(x)
        
        
    
        
        dtype = q.dtype
        q = self.q_norm(q)
        k = self.k_norm(k)

        # KV compression
        if self.sr_ratio > 1:
            k, new_N = self.downsample_2d(k, H, W, self.sr_ratio, sampling=self.sampling)
            v, new_N = self.downsample_2d(v, H, W, self.sr_ratio, sampling=self.sampling)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = th.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class GraspMultiHeadCrossAttention(nn.Module):
    def __init__(self, block):
        super().__init__()

        self.d_model = block.cross_attn.d_model
        self.num_heads = block.cross_attn.num_heads
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // self.num_heads

        k_linear, v_linear = self.split_kv_linear(block.cross_attn.kv_linear)
        q_linear = block.cross_attn.q_linear
        
        original_q_layer = q_linear
        W_q = original_q_layer.weight.data
        bias_q = original_q_layer.bias.data if original_q_layer.bias is not None else None
        U_q, S_q, Vh_q = torch.linalg.svd(W_q, full_matrices=False)
        self.q_proj = GRASPLayer(U=U_q, S=S_q, Vh=Vh_q, bias=bias_q, compression_ratio=0.5)
        
        
        
        original_k_layer = k_linear
        W_k = original_k_layer.weight.data
        bias_k = original_k_layer.bias.data if original_k_layer.bias is not None else None
        U_k, S_k, Vh_k = torch.linalg.svd(W_k, full_matrices=False)

        # Ersetzt k_linear durch den GRASPLayer
        self.k_proj = GRASPLayer(U=U_k, S=S_k, Vh=Vh_k, bias=bias_k, compression_ratio=0.5)
        
        original_v_layer = v_linear
        W_v = original_v_layer.weight.data
        bias_v = original_v_layer.bias.data if original_v_layer.bias is not None else None
        U_v, S_v, Vh_v = torch.linalg.svd(W_v, full_matrices=False)

        # Ersetzt v_linear durch den GRASPLayer
        self.v_proj = GRASPLayer(U=U_v, S=S_v, Vh=Vh_v, bias=bias_v, compression_ratio=0.5)

        
        self.attn_drop = block.cross_attn.attn_drop
        self.proj = block.cross_attn.proj
        self.proj_drop = block.cross_attn.proj_drop

    def split_kv_linear(self, kv_linear: nn.Linear):
        d_model = kv_linear.in_features
        W = kv_linear.weight.data        # [2*d_model, d_model]
        b = kv_linear.bias.data if kv_linear.bias is not None else None

        # Slice into halves
        W_k, W_v = W[:d_model, :], W[d_model:, :]
        b_k = b[:d_model] if b is not None else None
        b_v = b[d_model:] if b is not None else None

        # Create two nn.Linear layers
        k_linear = nn.Linear(d_model, d_model, bias=b is not None)
        v_linear = nn.Linear(d_model, d_model, bias=b is not None)

        # Copy weights/bias
        k_linear.weight.data.copy_(W_k)
        v_linear.weight.data.copy_(W_v)
        if b is not None:
            k_linear.bias.data.copy_(b_k)
            v_linear.bias.data.copy_(b_v)

        return k_linear, v_linear

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape


        q = self.q_proj(x).view(1, -1, self.num_heads, self.head_dim)

        v = self.v_proj(cond).view(1, -1, self.num_heads, self.head_dim)

        k = self.k_proj(cond).view(1, -1, self.num_heads, self.head_dim)
        
        
        
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
    
    
    
    
class GraspCompressedMLP(nn.Module):
    def __init__(self, block: PixArtMSBlock, rank: int):
        super().__init__()
        
        U1, S1, Vh1 = grasp_decompose_linear_to_svd(block.mlp.fc1, r=rank)
        self.fc1 = SVDLinear(U1, S1, Vh1, block.mlp.fc1.bias)
        
        U2, S2, Vh2 = grasp_decompose_linear_to_svd(block.mlp.fc2, r=rank)
        self.fc2 = SVDLinear(U2, S2, Vh2, block.mlp.fc2.bias)
        
        self.approx_gelu = nn.GELU(approximate="tanh")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.approx_gelu(x)
        x = self.fc2(x)
        return x

class GraspCompressedMultiHeadCrossAttention(nn.Module):
    def __init__(self, block, rank):
        super().__init__()

        self.d_model = block.cross_attn.d_model
        self.num_heads = block.cross_attn.num_heads
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // self.num_heads

        k_linear, v_linear = self.split_kv_linear(block.cross_attn.kv_linear)
        q_linear = block.cross_attn.q_linear
        
        Uq, Sq, Vhq = grasp_decompose_linear_to_svd(q_linear, r=rank)
        self.q_proj = SVDLinear(Uq, Sq, Vhq, q_linear.bias)
        
        Uk, Sk, Vhk = grasp_decompose_linear_to_svd(k_linear, r=rank)
        self.k_proj = SVDLinear(Uk, Sk, Vhk, k_linear.bias)
        
        Uv, Sv, Vhv = grasp_decompose_linear_to_svd(v_linear, r=rank)
        self.v_proj = SVDLinear(Uv, Sv, Vhv, v_linear.bias)

        
        self.attn_drop = block.cross_attn.attn_drop
        self.proj = block.cross_attn.proj
        self.proj_drop = block.cross_attn.proj_drop

    def split_kv_linear(self, kv_linear: nn.Linear):
        d_model = kv_linear.in_features
        W = kv_linear.weight.data        # [2*d_model, d_model]
        b = kv_linear.bias.data if kv_linear.bias is not None else None

        # Slice into halves
        W_k, W_v = W[:d_model, :], W[d_model:, :]
        b_k = b[:d_model] if b is not None else None
        b_v = b[d_model:] if b is not None else None

        # Create two nn.Linear layers
        k_linear = nn.Linear(d_model, d_model, bias=b is not None)
        v_linear = nn.Linear(d_model, d_model, bias=b is not None)

        # Copy weights/bias
        k_linear.weight.data.copy_(W_k)
        v_linear.weight.data.copy_(W_v)
        if b is not None:
            k_linear.bias.data.copy_(b_k)
            v_linear.bias.data.copy_(b_v)

        return k_linear, v_linear

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape
        
        q = self.q_proj(x).view(1, -1, self.num_heads, self.head_dim)
        v = self.v_proj(cond).view(1, -1, self.num_heads, self.head_dim)
        k = self.k_proj(cond).view(1, -1, self.num_heads, self.head_dim)

        
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x    
    
class GraspCompressedAttentionKV(nn.Module):
    """Multi-head Attention block with KV token compression and qk norm."""

    def __init__(
        self,
        block: PixArtMSBlock, 
        rank: int,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.sampling=block.attn.sampling    # ['conv', 'ave', 'uniform', 'uniform_every']
        self.sr_ratio = block.attn.sr_ratio
        self.attn_drop = block.attn.attn_drop
        self.proj = block.attn.proj
        self.proj_drop = block.attn.proj_drop
        self.sampling = block.attn.sampling
        self.qk_norm = block.attn.qk_norm
        
        if self.sr_ratio > 1 and self.sampling == 'conv':
            # Avg Conv Init.
            self.sr = block.attn.sr
            self.sr.weight.data.fill_(1/block.attn.sr_ratio**2)
            self.sr.bias.data.zero_()
            self.norm = block.attn.norm
        if self.qk_norm:
            self.q_norm =block.attn.q_norm
            self.k_norm = block.attn.k_norm
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            
        self.num_heads = block.attn.num_heads

        q_linear,k_linear,v_linear = self.split_qkv_linear(block.attn.qkv)
        
        Uq, Sq, Vhq = grasp_decompose_linear_to_svd(q_linear, r=rank)
        self.q_proj = SVDLinear(Uq, Sq, Vhq, q_linear.bias)
        
        Uk, Sk, Vhk = grasp_decompose_linear_to_svd(k_linear, r=rank)
        self.k_proj = SVDLinear(Uk, Sk, Vhk, k_linear.bias)
        
        Uv, Sv, Vhv = grasp_decompose_linear_to_svd(v_linear, r=rank)
        self.v_proj = SVDLinear(Uv, Sv, Vhv, v_linear.bias)
     


        
    def downsample_2d(self, tensor, H, W, scale_factor, sampling=None):
        if sampling is None or scale_factor == 1:
            return tensor
        B, N, C = tensor.shape

        if sampling == 'uniform_every':
            return tensor[:, ::scale_factor], int(N // scale_factor)

        tensor = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == 'ave':
            tensor = F.interpolate(
                tensor, scale_factor=1 / scale_factor, mode='nearest'
            ).permute(0, 2, 3, 1)
        elif sampling == 'uniform':
            tensor = tensor[:, :, ::scale_factor, ::scale_factor].permute(0, 2, 3, 1)
        elif sampling == 'conv':
            tensor = self.sr(tensor).reshape(B, C, -1).permute(0, 2, 1)
            tensor = self.norm(tensor)
        else:
            raise ValueError
   
    def split_qkv_linear(self, qkv_layer: nn.Linear):
        in_dim = qkv_layer.in_features   # = dim
        out_dim = qkv_layer.out_features # = 3*dim
        dim = out_dim // 3

        W = qkv_layer.weight.data        # [3*dim, dim]
        b = qkv_layer.bias.data if qkv_layer.bias is not None else None

        # slice weights
        W_q, W_k, W_v = W[:dim, :], W[dim:2*dim, :], W[2*dim:, :]
        b_q = b[:dim] if b is not None else None
        b_k = b[dim:2*dim] if b is not None else None
        b_v = b[2*dim:] if b is not None else None

        # build independent linears
        q = nn.Linear(in_dim, dim, bias=b_q is not None)
        k = nn.Linear(in_dim, dim, bias=b_k is not None)
        v = nn.Linear(in_dim, dim, bias=b_v is not None)

        # copy weights/bias
        q.weight.data.copy_(W_q);  k.weight.data.copy_(W_k);  v.weight.data.copy_(W_v)
        if b_q is not None:
            q.bias.data.copy_(b_q); k.bias.data.copy_(b_k); v.bias.data.copy_(b_v)

        return q, k, v

    
    def forward(self, x, mask=None, HW=None, block_id=None):
        B, N, C = x.shape
        new_N = N
        if HW is None:
            H = W = int(N ** 0.5)
        else:
            H, W = HW
        
        # using the lower rank matrices 
        q = self.q_proj(x)
        v = self.v_proj(x)
        k = self.k_proj(x)
        
        
    
        
        dtype = q.dtype
        q = self.q_norm(q)
        k = self.k_norm(k)

        # KV compression
        if self.sr_ratio > 1:
            k, new_N = self.downsample_2d(k, H, W, self.sr_ratio, sampling=self.sampling)
            v, new_N = self.downsample_2d(v, H, W, self.sr_ratio, sampling=self.sampling)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = th.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x







class PrunedMLP(nn.Module):
    def __init__(self, block: PixArtMSBlock, rank: int):
        super().__init__()
        
        linear1_A, linear1_B = decompose_linear_to_svd(block.mlp.fc1, r=rank)
        self.linear1_A, self.linear1_B = nn.Parameter(linear1_A), nn.Parameter(linear1_B)
        self.bias1 = nn.Parameter(deepcopy(block.mlp.fc1.bias))
        
        linear2_A, linear2_B = decompose_linear_to_svd(block.mlp.fc2, r=rank)
        self.linear2_A, self.linear2_B = nn.Parameter(linear2_A), nn.Parameter(linear2_B)
        self.bias2 = nn.Parameter(deepcopy(block.mlp.fc2.bias))
        
        self.approx_gelu = nn.GELU(approximate="tanh")
    
    def forward(self, x):
        x = (x @ self.linear1_A @ self.linear1_B) + self.bias1
        x = self.approx_gelu(x)
        x = (x @ self.linear2_A @ self.linear2_B) + self.bias2
        return x
    
class AttentionKVCompressPruned(nn.Module):
    """Multi-head Attention block with KV token compression and qk norm."""

    def __init__(
        self,
        block: PixArtMSBlock, 
        rank: int,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.sampling=block.attn.sampling    # ['conv', 'ave', 'uniform', 'uniform_every']
        self.sr_ratio = block.attn.sr_ratio
        self.attn_drop = block.attn.attn_drop
        self.proj = block.attn.proj
        self.proj_drop = block.attn.proj_drop
        self.sampling = block.attn.sampling
        self.qk_norm = block.attn.qk_norm
        
        if self.sr_ratio > 1 and self.sampling == 'conv':
            # Avg Conv Init.
            self.sr = block.attn.sr
            self.sr.weight.data.fill_(1/block.attn.sr_ratio**2)
            self.sr.bias.data.zero_()
            self.norm = block.attn.norm
        if self.qk_norm:
            self.q_norm =block.attn.q_norm
            self.k_norm = block.attn.k_norm
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            
        self.num_heads = block.attn.num_heads

        q_linear,k_linear,v_linear = self.split_qkv_linear(block.attn.qkv)
     

        linear_q_A, linear_q_B = decompose_linear_to_svd(q_linear, r=rank)
        self.linear_q_A, self.linear_q_B = nn.Parameter(linear_q_A), nn.Parameter(linear_q_B)
        self.bias_q = nn.Parameter(deepcopy(q_linear.bias.data))
        
        linear_k_A, linear_k_B = decompose_linear_to_svd(k_linear, r=rank)
        self.linear_k_A, self.linear_k_B = nn.Parameter(linear_k_A), nn.Parameter(linear_k_B)
        self.bias_k = nn.Parameter(deepcopy(k_linear.bias.data))
        
        linear_v_A, linear_v_B = decompose_linear_to_svd(v_linear, r=rank)
        self.linear_v_A, self.linear_v_B = nn.Parameter(linear_v_A), nn.Parameter(linear_v_B)
        self.bias_v = nn.Parameter(deepcopy(v_linear.bias.data))

        
    def downsample_2d(self, tensor, H, W, scale_factor, sampling=None):
        if sampling is None or scale_factor == 1:
            return tensor
        B, N, C = tensor.shape

        if sampling == 'uniform_every':
            return tensor[:, ::scale_factor], int(N // scale_factor)

        tensor = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == 'ave':
            tensor = F.interpolate(
                tensor, scale_factor=1 / scale_factor, mode='nearest'
            ).permute(0, 2, 3, 1)
        elif sampling == 'uniform':
            tensor = tensor[:, :, ::scale_factor, ::scale_factor].permute(0, 2, 3, 1)
        elif sampling == 'conv':
            tensor = self.sr(tensor).reshape(B, C, -1).permute(0, 2, 1)
            tensor = self.norm(tensor)
        else:
            raise ValueError
   
    def split_qkv_linear(self, qkv_layer: nn.Linear):
        in_dim = qkv_layer.in_features   # = dim
        out_dim = qkv_layer.out_features # = 3*dim
        dim = out_dim // 3

        W = qkv_layer.weight.data        # [3*dim, dim]
        b = qkv_layer.bias.data if qkv_layer.bias is not None else None

        # slice weights
        W_q, W_k, W_v = W[:dim, :], W[dim:2*dim, :], W[2*dim:, :]
        b_q = b[:dim] if b is not None else None
        b_k = b[dim:2*dim] if b is not None else None
        b_v = b[2*dim:] if b is not None else None

        # build independent linears
        q = nn.Linear(in_dim, dim, bias=b_q is not None)
        k = nn.Linear(in_dim, dim, bias=b_k is not None)
        v = nn.Linear(in_dim, dim, bias=b_v is not None)

        # copy weights/bias
        q.weight.data.copy_(W_q);  k.weight.data.copy_(W_k);  v.weight.data.copy_(W_v)
        if b_q is not None:
            q.bias.data.copy_(b_q); k.bias.data.copy_(b_k); v.bias.data.copy_(b_v)

        return q, k, v

    
    def forward(self, x, mask=None, HW=None, block_id=None):
        B, N, C = x.shape
        new_N = N
        if HW is None:
            H = W = int(N ** 0.5)
        else:
            H, W = HW
        
        # using the lower rank matrices 
        q = (x @ self.linear_q_A @ self.linear_q_B) + self.bias_q
        v = (x @ self.linear_v_A @ self.linear_v_B) + self.bias_v
        k = (x @ self.linear_k_A @ self.linear_k_B) + self.bias_k
        
        
    
        
        dtype = q.dtype
        q = self.q_norm(q)
        k = self.k_norm(k)

        # KV compression
        if self.sr_ratio > 1:
            k, new_N = self.downsample_2d(k, H, W, self.sr_ratio, sampling=self.sampling)
            v, new_N = self.downsample_2d(v, H, W, self.sr_ratio, sampling=self.sampling)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = th.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class MultiHeadCrossAttentionPruned(nn.Module):
    def __init__(self, block, rank):
        super().__init__()

        self.d_model = block.cross_attn.d_model
        self.num_heads = block.cross_attn.num_heads
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // self.num_heads

        k_linear, v_linear = self.split_kv_linear(block.cross_attn.kv_linear)
        q_linear = block.cross_attn.q_linear
        
        linear_q_A, linear_q_B = decompose_linear_to_svd(q_linear, r=rank)
        self.linear_q_A, self.linear_q_B = nn.Parameter(linear_q_A), nn.Parameter(linear_q_B)
        self.bias_q = nn.Parameter(deepcopy(q_linear.bias.data))
        
        linear_k_A, linear_k_B = decompose_linear_to_svd(k_linear, r=rank)
        self.linear_k_A, self.linear_k_B = nn.Parameter(linear_k_A), nn.Parameter(linear_k_B)
        self.bias_k = nn.Parameter(deepcopy(k_linear.bias.data))
        
        linear_v_A, linear_v_B = decompose_linear_to_svd(v_linear, r=rank)
        self.linear_v_A, self.linear_v_B = nn.Parameter(linear_v_A), nn.Parameter(linear_v_B)
        self.bias_v = nn.Parameter(deepcopy(v_linear.bias.data))

        
        self.attn_drop = block.cross_attn.attn_drop
        self.proj = block.cross_attn.proj
        self.proj_drop = block.cross_attn.proj_drop

    def split_kv_linear(self, kv_linear: nn.Linear):
        d_model = kv_linear.in_features
        W = kv_linear.weight.data        # [2*d_model, d_model]
        b = kv_linear.bias.data if kv_linear.bias is not None else None

        # Slice into halves
        W_k, W_v = W[:d_model, :], W[d_model:, :]
        b_k = b[:d_model] if b is not None else None
        b_v = b[d_model:] if b is not None else None

        # Create two nn.Linear layers
        k_linear = nn.Linear(d_model, d_model, bias=b is not None)
        v_linear = nn.Linear(d_model, d_model, bias=b is not None)

        # Copy weights/bias
        k_linear.weight.data.copy_(W_k)
        v_linear.weight.data.copy_(W_v)
        if b is not None:
            k_linear.bias.data.copy_(b_k)
            v_linear.bias.data.copy_(b_v)

        return k_linear, v_linear

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = ((x @ self.linear_q_A @ self.linear_q_B) + self.bias_q).view(1, -1, self.num_heads, self.head_dim)
        v = ((cond @ self.linear_v_A @ self.linear_v_B) + self.bias_v).view(1, -1, self.num_heads, self.head_dim)
        k = ((cond @ self.linear_k_A @ self.linear_k_B) + self.bias_k).view(1, -1, self.num_heads, self.head_dim)

        
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x