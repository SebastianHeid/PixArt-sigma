# https://huggingface.co/PixArt-alpha/PixArt-Sigma
#Compress mlp, attn, cross_attn via SVD
transformer_blocks = [8, 17, 16]
transformer_blocks_mlp = [4, 0, 3, 1, 5, 7, 25, 27, 2, 6, 18, 20, 8, 16, 21, 15, 11, 9, 17, 10, 23]
rank_mlp = 512

transformer_blocks_attn = [4, 0, 3, 1, 5, 7, 25, 27, 2, 6, 18, 20, 8, 16, 21, 15, 11, 9, 17, 10, 23]
rank_attn = 128

transformer_blocks_cross_attn =  [4, 0, 3, 1, 5, 7, 25, 27, 2, 6, 18, 20, 8, 16, 21, 15, 11, 9, 17, 10, 23]
rank_cross_attn = 128