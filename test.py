import warnings
from typing import Tuple, Union

import torch as th
import torch.nn as nn
from diffusion.model.nets import PixArtMS, PixArtMSBlock
from diffusion.model.nets.pruned_model_parts import (
    AttentionKVCompressPruned,
    MultiHeadCrossAttentionPruned,
    PrunedMLP,
)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from diffusion.model.modify_model import modify_model
from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.utils.misc import read_config

config_path = "/home/hd/hd_hd/hd_om233/partially_removal/PixArt-sigma/configs/pixart_sigma_config/partially_block_removal/First_Iteration.py"
config = read_config(config_path)
model = PixArtMS_XL_2()
print("Model: ", sum(p.numel() for p in model.parameters()))
m_model = modify_model(model, config)
print("Modified Model: ", sum(p.numel() for p in m_model.parameters()))



# def decompose_linear_to_svd(
#         linear_layer: nn.Linear,
#         r: int,
#         reverse: bool = False,
#         return_full: bool = False,
# ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
#     """
#     Decomposes a torch.nn.Linear layer into two LoRA-style matrices using truncated SVD.

#     Args:
#         linear_layer (nn.Linear): The original linear layer to decompose.
#         r (int): Rank for decomposition (r < min(in_features, out_features))

#     Returns:
#         A (nn.Parameter): Left matrix (in_features x r)
#         B (nn.Parameter): Right matrix (r x out_features)
#     """
#     # Get original weight (shape: out_features x in_features)
#     W = linear_layer.weight.data  # shape: [out_features, in_features]

#     # Perform full SVD on the transposed weight to get shape (in_features x out_features)
#     # This lets us get A (in_features x r) and B (r x out_features)
#     U, S, Vh = th.linalg.svd(W.T, full_matrices=False)

#     # Truncate to rank-r
#     if not reverse:
#         U_r = U[:, :r]  # shape: [in_features, r]
#         S_r = S[:r]  # shape: [r]
#         Vh_r = Vh[:r, :]  # shape: [r, out_features]
#     else:
#         U_r = U[:, r:]  # shape: [in_features, full - r]
#         S_r = S[r:]  # shape: [full - r]
#         Vh_r = Vh[r:, :]  # shape: [full - r, out_features]

#     if return_full:
#         return (U_r @ th.diag(S_r) @ Vh_r).T

#     # A = U_r
#     A = U_r @ th.diag(th.sqrt(S_r))  # shape: [in_features, r]
#     B = th.diag(th.sqrt(S_r)) @ Vh_r  # shape: [r, out_features]
    
#     return A.T, B.T





# ckpt_file = "/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/PixArt-Sigma-XL-2-512-MS.pth" 
# hidden_size=1152
# num_heads=16
# block = PixArtMSBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=4.0)

# block = block.to("cuda")
# x = th.ones((2, 1024,1152)).to("cuda")
# y = th.ones((2, 3904, 1152)).to("cuda")
# t = th.ones((2,6912)).to("cuda")
# out = block(x,y,t)
# print(out.shape)

# block1 = PixArtMSBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=4.0)
# block1 = block1.to("cuda")
# att_pruned = AttentionKVCompressPruned( block1, 128)

# att_pruned = att_pruned.to("cuda")
# block1.attn = att_pruned
# out_pruned = block1(x,y,t)
# print(out_pruned.shape)


# mlp_pruned = PrunedMLP(block1, 512)
# mlp_pruned = mlp_pruned.to("cuda")
# block1.mlp = mlp_pruned


# cross_pruned = MultiHeadCrossAttentionPruned(block1, 128)
# cross_pruned = cross_pruned.to("cuda")
# block1.cross_attn = cross_pruned
# # out_pruned = block1(x,y,t)
# print(out_pruned.shape)

# print("param block", sum(p.numel() for p in block.parameters()))
# print("param block1", sum(p.numel() for p in block1.parameters()))

# print("param attn", sum(p.numel() for p in block.attn.parameters()))
# print("param attn1", sum(p.numel() for p in block1.attn.parameters()))

# print("param mlp", sum(p.numel() for p in block.mlp.parameters()))
# print("param mlp1", sum(p.numel() for p in block1.mlp.parameters()))

# print("param cross", sum(p.numel() for p in block.cross_attn.parameters()))
# print("param cross", sum(p.numel() for p in block1.cross_attn.parameters()))


# print("Smaller Block")
# print("Fc1 weight:", block.mlp.fc1.weight.shape)
# print("Fc1 bias:",block.mlp.fc1.bias.shape)
# print("Fc2 weight:",block.mlp.fc2.weight.shape)
# print("Fc1 bias:",block.mlp.fc2.bias.shape)

# checkpoint = th.load(ckpt_file, map_location="cpu")

# params_to_check = [
#     'blocks.27.scale_shift_table',
#     'blocks.27.attn.qkv.weight',
#     'blocks.27.attn.qkv.bias',
#     'blocks.27.attn.proj.weight',
#     'blocks.27.attn.proj.bias',
#     'blocks.27.mlp.fc1.weight',
#     'blocks.27.mlp.fc1.bias',
#     'blocks.27.mlp.fc2.weight',
#     'blocks.27.mlp.fc2.bias',
#     'blocks.27.cross_attn.q_linear.weight',
#     'blocks.27.cross_attn.q_linear.bias',
#     'blocks.27.cross_attn.kv_linear.weight',
#     'blocks.27.cross_attn.kv_linear.bias',
#     'blocks.27.cross_attn.proj.weight',
#     'blocks.27.cross_attn.proj.bias',
# ]

# for name in params_to_check:
#     if name in checkpoint["state_dict"]:
#         print(name, checkpoint["state_dict"][name].shape)
#     else:
#         print(name, "not found in checkpoint")
#         print(name, "not found in checkpoint")
#         print(name, "not found in checkpoint")







# import json

# # Load the original JSON
# with open("/home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion.json", "r") as f:
#     old_data = json.load(f)

# new_data = {}

# for i, (k, v) in enumerate(old_data.items(), start=1):
#     # Each value becomes a list with one string
#     new_key = f"prompt{i}"
#     new_data[new_key] = [v]

# # Save the transformed JSON
# with open("/home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion_new.json", "w") as f:
#     json.dump(new_data, f, indent=2)
#     json.dump(new_data, f, indent=2)
#     json.dump(new_data, f, indent=2)
#     json.dump(new_data, f, indent=2)
