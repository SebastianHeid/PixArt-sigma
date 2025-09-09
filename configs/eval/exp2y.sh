CUDA_VISIBLE_DEVICES=4 python /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/scripts/inference2.py \
  --model_path /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13/checkpoints/epoch_1_step_38000.pth \
  --add_red_hidd_size_factor 2 \
  --add_num_head 8 \
  --add_param_blocks 10 \
  --add_mlp_ratio 2.0 \
  --transformer_blocks 17 15 8 20 11 16 12 23 21 18 24 7 13 \
  --save_path /export/scratch/sheid/similarty_matrix/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13/epoch_1_step_38000/

CUDA_VISIBLE_DEVICES=4 python /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/scripts/inference2.py \
  --model_path /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13/checkpoints/epoch_1_step_76000.pth \
  --add_red_hidd_size_factor 2 \
  --add_num_head 8 \
  --add_param_blocks 10 \
  --add_mlp_ratio 2.0 \
  --transformer_blocks 17 15 8 20 11 16 12 23 21 18 24 7 13 \
  --save_path /export/scratch/sheid/similarty_matrix/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13/epoch_1_step_76000/

