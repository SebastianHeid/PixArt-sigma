from diffusion.model.nets.place_holder_blocks import Identity
from diffusion.model.nets.pruned_model_parts import (
    AttentionKVCompressPruned,
    MultiHeadCrossAttentionPruned,
    PrunedMLP,
)


def modify_model(model, config):
    # Example modification: replace a specific layer with an Identity layer
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
    return model