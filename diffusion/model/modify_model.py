from diffusion.model.nets.place_holder_blocks import Identity
from diffusion.model.nets.pruned_model_parts import (
    AttentionKVCompressPruned,
    GraspAttentionKVCompress,
    GraspCompressedAttentionKV,
    GraspCompressedMLP,
    GraspCompressedMultiHeadCrossAttention,
    GraspdMLP,
    GraspMultiHeadCrossAttention,
    MultiHeadCrossAttentionPruned,
    PrunedMLP,
)


def modify_model(model, config):
    # Example modification: replace a specific layer with an Identity layer
    
    # -------------------- GRASP Method to determine which Singular Values to keep -------------------------
    for idx in config.invSVD_blocks:
        mlp_helper = GraspdMLP(model.blocks[idx], config.rank_mlp)
        model.blocks[idx].mlp = mlp_helper
        
        attn_helper = GraspAttentionKVCompress(model.blocks[idx], config.rank_attn)
        model.blocks[idx].attn = attn_helper
        
        cross_attn_helper = GraspMultiHeadCrossAttention(model.blocks[idx], config.rank_cross_attn)
        model.blocks[idx].cross_attn = cross_attn_helper
    
    # ------------------------------------------------------------------------------------------------------
    for idx in config.grasp_compressed_blocks:
        mlp_helper = GraspCompressedMLP(model.blocks[idx], config.rank_mlp)
        model.blocks[idx].mlp = mlp_helper
        
        attn_helper = GraspCompressedAttentionKV(model.blocks[idx], config.rank_attn)
        model.blocks[idx].attn = attn_helper
        
        cross_attn_helper = GraspCompressedMultiHeadCrossAttention(model.blocks[idx], config.rank_cross_attn)
        model.blocks[idx].cross_attn = cross_attn_helper
    
    
    # --------------------- Replace MLP, Attn, Cross Attn via SVD keeping the r biggest singular valuer ----------------------------
    for idx in config.transformer_blocks_mlp:
        print("Reduced MLP size")
        mlp_helper = PrunedMLP(model.blocks[idx], config.rank_mlp)
        model.blocks[idx].mlp = mlp_helper
    
    for idx in config.transformer_blocks_attn:
        print("Reduced Attn size")
        attn_helper = AttentionKVCompressPruned(model.blocks[idx], config.rank_attn)
        model.blocks[idx].attn = attn_helper
        
    for idx in config.transformer_blocks_cross_attn:
        print("Reduced Cross Attn size")
        cross_attn_helper = MultiHeadCrossAttentionPruned(model.blocks[idx], config.rank_cross_attn)
        model.blocks[idx].cross_attn = cross_attn_helper
    for idx in config.transformer_blocks:
        model.blocks[idx] = Identity()
    
    # -------------------------------------------------------------------------------------------------------------------------------
    
    
    
    return model