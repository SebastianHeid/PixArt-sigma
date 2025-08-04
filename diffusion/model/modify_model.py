from diffusion.model.nets.place_holder_blocks import Identity


def modify_model(model, transformer_blocks):
    # Example modification: replace a specific layer with an Identity layer
    for idx in transformer_blocks:
        model.blocks[idx] = Identity()
    return model