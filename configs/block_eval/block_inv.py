# https://huggingface.co/PixArt-alpha/PixArt-Sigma
#Compress mlp, attn, cross_attn via SVD
transformer_blocks = []
transformer_blocks_mlp = [0, 1, 2, 3, 4, 5, 6, 16, 21,23, 27]

rank_mlp = 368

transformer_blocks_attn = [0, 1, 2, 3, 4, 5, 6, 16, 21,23, 27]

rank_attn = 230

transformer_blocks_cross_attn =  [0, 1, 2, 3, 4, 5, 6, 16, 21,23, 27]
rank_cross_attn = 230