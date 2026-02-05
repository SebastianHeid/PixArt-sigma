from diffusion.model.nets.place_holder_blocks import Identity
from diffusion.model.nets.pruned_model_parts import (
    AttentionKVCompressPruned,
    MultiHeadCrossAttentionPruned,
    PrunedMLP,
)
from typing import Tuple
from diffusion.model.utils import decompose_linear_to_svd
import numpy as np
import torch
import torch.nn as nn


def compute_svd_rank_for_compression(n: int, m: int, ratio_x: float) -> int:
    """
    Berechnet den Ziel-Rang (r') für eine SVD-basierte Kompression, 
    um die Parameterzahl um ratio_x zu reduzieren.
    
    Args:
        n (int): Eingabedimension (Zeilen der Matrix W).
        m (int): Ausgabedimension (Spalten der Matrix W).
        ratio_x (float): Gewünschtes Reduktionsverhältnis der Parameter (z.B. 0.6 für 60% Reduktion).

    Returns:
        int: Der Ziel-Rang r' (aufgerundet).
    """
    P_orig = n * m
    P_target = P_orig * (1.0 - ratio_x)
    r_float = P_target / (n + m)
    r_prime = max(1, int(np.ceil(r_float)))
    if ratio_x >= 1.0:
        return 0
    r_prime = min(r_prime, min(n, m))
    return int(r_prime)

def iterative_svd_pruning(
    A: np.ndarray, 
    B: np.ndarray, 
    rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    W = A @ B
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]  # shape: [in_features, r]
    S_r = S[:rank]  # shape: [r]
    Vh_r = Vh[:rank, :]  # shape: [r, out_features]
    A = U_r @ torch.diag(torch.sqrt(S_r))  # shape: [in_features, r]
    B = torch.diag(torch.sqrt(S_r)) @ Vh_r  # shape: [r, out_features]
    return A, B


def modify_model_training(model, config):
    # Create the base model
    rank_attn = []
    rank_mlp = []
    for r in config.compression_ratio_mlp_new:
        rank_mlp.append(compute_svd_rank_for_compression(1152, 1152*4, r))
    for r in config.compression_ratio_attn_new:
        rank_attn.append(compute_svd_rank_for_compression(1152, 1152, r))
    
    for idx_rank, block in enumerate(config.transformer_blocks_mlp_new):
        print("MLP")
        if  block in config.transformer_blocks_mlp:
            replace_low_rank_matrices_mlp(model.blocks[block], rank_mlp[idx_rank])
        else:
            mlp_helper = PrunedMLP(model.blocks[block], rank_mlp[idx_rank])
            model.blocks[block].mlp = mlp_helper
            
    for idx_rank, block in enumerate(config.transformer_blocks_attn_new):
        print("ATTN")
        if  block in config.transformer_blocks_attn:
            replace_low_rank_matrices_attn(model.blocks[block], rank_attn[idx_rank])
        else:
            attn_helper = AttentionKVCompressPruned(model.blocks[block], rank_attn[idx_rank])
            model.blocks[block].attn = attn_helper
            
    
    for idx_rank, block in enumerate(config.transformer_blocks_cross_attn_new):
        print("CROSS ATTN")
        if  block in config.transformer_blocks_cross_attn:
            replace_low_rank_matrices_cross_attn(model.blocks[block], rank_attn[idx_rank])
        else:
            cross_attn_helper = MultiHeadCrossAttentionPruned(model.blocks[block], rank_attn[idx_rank])
            model.blocks[block].cross_attn = cross_attn_helper
    
    return model

def modify_model_base(model, config):
    # Create the base model
    for idx_rank, idx in enumerate(config.transformer_blocks_mlp):
        print("Reduced MLP size")
        rank_mlp = compute_svd_rank_for_compression(1152, 1152*4, config.compression_ratio_mlp[idx_rank])
        mlp_helper = PrunedMLP(model.blocks[idx], rank_mlp)
        model.blocks[idx].mlp = mlp_helper
    
    for idx_rank, idx in enumerate(config.transformer_blocks_attn):
        print("Reduced Attn size")
        rank_attn = compute_svd_rank_for_compression(1152, 1152, config.compression_ratio_attn[idx_rank])  
        attn_helper = AttentionKVCompressPruned(model.blocks[idx], rank_attn)
        model.blocks[idx].attn = attn_helper
        
    for idx_rank, idx in enumerate(config.transformer_blocks_cross_attn):
        print("Reduced Cross Attn size")
        rank_cross_attn = compute_svd_rank_for_compression(1152, 1152, config.compression_ratio_cross_attn[idx_rank])  
        cross_attn_helper = MultiHeadCrossAttentionPruned(model.blocks[idx], rank_cross_attn)
        model.blocks[idx].cross_attn = cross_attn_helper
        
    return model 
    
    
def modify_model_current(config, model, blocks, compression_ratios):
    # Create the base model
    rank_attns = []
    rank_mlp = []
    for r in compression_ratios:
        rank_attns.append(compute_svd_rank_for_compression(1152, 1152, r))
        rank_mlp.append(compute_svd_rank_for_compression(1152, 1152*4, r))
    
    for idx in range(len(blocks)):
        if blocks[idx] in config.transformer_blocks_attn and blocks[idx] in config.transformer_blocks_cross_attn and blocks[idx] in config.transformer_blocks_mlp:
            replace_low_rank_matrices_mlp(model.blocks[blocks[idx]], rank_mlp[idx])
            replace_low_rank_matrices_attn(model.blocks[blocks[idx]], rank_attns[idx])
            replace_low_rank_matrices_cross_attn(model.blocks[blocks[idx]], rank_attns[idx])
        else:
            mlp_helper = PrunedMLP(model.blocks[blocks[idx]], rank_mlp[idx])
            model.blocks[blocks[idx]].mlp = mlp_helper
            
            attn_helper = AttentionKVCompressPruned(model.blocks[blocks[idx]], rank_attns[idx])
            model.blocks[blocks[idx]].attn = attn_helper
                
            cross_attn_helper = MultiHeadCrossAttentionPruned(model.blocks[blocks[idx]],rank_attns[idx])
            model.blocks[blocks[idx]].cross_attn = cross_attn_helper
            
        
    return model 
    
    
def modify_model_new(model, config, compression_ratio_idx):   
    # prune the new block
    idx_block = config.new_block[0]
    if idx_block in config.transformer_blocks_attn and idx_block in config.transformer_blocks_cross_attn and idx_block in config.transformer_blocks_mlp:
        idx_ = config.transformer_blocks_mlp.index(idx_block)
        old_compression_ratio = config.compression_ratio_mlp[idx_]
        new_compression_ratio = 1 - (1-old_compression_ratio) * (1-config.compression_ratios[compression_ratio_idx])
    else: 
        new_compression_ratio = config.compression_ratios[compression_ratio_idx]
    

    rank_attns = compute_svd_rank_for_compression(1152, 1152, new_compression_ratio)  
    rank_mlp = compute_svd_rank_for_compression(1152, 1152*4, new_compression_ratio)
    

    if idx_block in config.transformer_blocks_attn and idx_block in config.transformer_blocks_cross_attn and idx_block in config.transformer_blocks_mlp:
        replace_low_rank_matrices_mlp(model.blocks[idx_block], rank_mlp)
        replace_low_rank_matrices_attn(model.blocks[idx_block], rank_attns)
        replace_low_rank_matrices_cross_attn(model.blocks[idx_block], rank_attns)
    else:
        mlp_helper = PrunedMLP(model.blocks[idx_block], rank_mlp)
        model.blocks[idx_block].mlp = mlp_helper
        
        attn_helper = AttentionKVCompressPruned(model.blocks[idx_block], rank_attns)
        model.blocks[idx_block].attn = attn_helper
            
        cross_attn_helper = MultiHeadCrossAttentionPruned(model.blocks[idx_block],rank_attns)
        model.blocks[idx_block].cross_attn = cross_attn_helper
    return model


def replace_low_rank_matrices_mlp(block, new_rank):
    new_A, new_B = iterative_svd_pruning(block.mlp.linear1_A, block.mlp.linear1_B, new_rank)
    block.mlp.linear1_A, block.mlp.linear1_B = nn.Parameter(new_A), nn.Parameter(new_B)
    
    new2_A, new2_B = iterative_svd_pruning(block.mlp.linear2_A, block.mlp.linear2_B, new_rank)
    block.mlp.linear2_A, block.mlp.linear2_B = nn.Parameter(new2_A), nn.Parameter(new2_B)
    
    
def replace_low_rank_matrices_attn(block, new_rank):
    new_linear_q_A, new_linear_q_B = iterative_svd_pruning(block.attn.linear_q_A, block.attn.linear_q_B, new_rank)
    block.attn.linear_q_A, block.attn.linear_q_B = nn.Parameter(new_linear_q_A), nn.Parameter(new_linear_q_B)
    
    new_linear_k_A, new_linear_k_B = iterative_svd_pruning(block.attn.linear_k_A, block.attn.linear_k_B, new_rank)
    block.attn.linear_k_A, block.attn.linear_k_B = nn.Parameter(new_linear_k_A), nn.Parameter(new_linear_k_B)
    
    new_linear_v_A, new_linear_v_B = iterative_svd_pruning(block.attn.linear_v_A, block.attn.linear_v_B, new_rank)
    block.attn.linear_v_A, block.attn.linear_v_B = nn.Parameter(new_linear_v_A), nn.Parameter(new_linear_v_B)
    
def replace_low_rank_matrices_cross_attn(block, new_rank):
    new_linear_q_A, new_linear_q_B = iterative_svd_pruning(block.cross_attn.linear_q_A, block.cross_attn.linear_q_B, new_rank)
    block.cross_attn.linear_q_A, block.cross_attn.linear_q_B = nn.Parameter(new_linear_q_A), nn.Parameter(new_linear_q_B)
    
    new_linear_k_A, new_linear_k_B = iterative_svd_pruning(block.cross_attn.linear_k_A, block.cross_attn.linear_k_B, new_rank)
    block.cross_attn.linear_k_A, block.cross_attn.linear_k_B = nn.Parameter(new_linear_k_A), nn.Parameter(new_linear_k_B)
    
    new_linear_v_A, new_linear_v_B = iterative_svd_pruning(block.cross_attn.linear_v_A, block.cross_attn.linear_v_B, new_rank)
    block.cross_attn.linear_v_A, block.cross_attn.linear_v_B = nn.Parameter(new_linear_v_A), nn.Parameter(new_linear_v_B)
    