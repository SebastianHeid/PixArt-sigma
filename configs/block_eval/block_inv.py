# https://huggingface.co/PixArt-alpha/PixArt-Sigma
#Compress mlp, attn, cross_attn via SVD
transformer_blocks = []
transformer_blocks_mlp =  []
rank_mlp = 460

transformer_blocks_attn =  []
rank_attn = 288

transformer_blocks_cross_attn =   []
rank_cross_attn = 288

invSVD_blocks = []

grasp_compressed_blocks = [4, 0, 3, 1, 5, 7, 25, 27, 2, 6, 18]