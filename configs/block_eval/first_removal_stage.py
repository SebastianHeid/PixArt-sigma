model_path = "/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/PixArt-Sigma-XL-2-512-MS.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
#Compress mlp, attn, cross_attn via SVD
blocks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

transformer_blocks_mlp = []
rank_mlp = 512

transformer_blocks_attn = []
rank_attn = 128

transformer_blocks_cross_attn =  []
rank_cross_attn = 128