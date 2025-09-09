CUDA_VISIBLE_DEVICES=5 python /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/scripts/inference2.py \
  --model_path /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/checkpoints/epoch_1_step_38000.pth \
  --add_red_hidd_size_factor 2 \
  --add_num_head 8 \
  --add_param_blocks 14 \
  --add_mlp_ratio 2.0 \
  --transformer_blocks 17 15 8 20 11 16  \
  --save_path /export/scratch/sheid/similarty_matrix/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/epoch_1_step_38000/

  CUDA_VISIBLE_DEVICES=5 python /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/scripts/inference2.py \
  --model_path /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/checkpoints/epoch_2_step_76001.pth \
  --add_red_hidd_size_factor 2 \
  --add_num_head 8 \
  --add_param_blocks 14 \
  --add_mlp_ratio 2.0 \
  --transformer_blocks 17 15 8 20 11 16  \
  --save_path /export/scratch/sheid/similarty_matrix/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/epoch_2_step_76001/


  CUDA_VISIBLE_DEVICES=5 python /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/scripts/inference2.py \
  --model_path /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/checkpoints/epoch_3_step_114001.pth \
  --add_red_hidd_size_factor 2 \
  --add_num_head 8 \
  --add_param_blocks 14 \
  --add_mlp_ratio 2.0 \
  --transformer_blocks 17 15 8 20 11 16  \
  --save_path /export/scratch/sheid/similarty_matrix/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/epoch_3_step_114001/


  CUDA_VISIBLE_DEVICES=5 python /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/scripts/inference2.py \
  --model_path /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/checkpoints/epoch_4_step_152001.pth \
  --add_red_hidd_size_factor 2 \
  --add_num_head 8 \
  --add_param_blocks 14 \
  --add_mlp_ratio 2.0 \
  --transformer_blocks 17 15 8 20 11 16  \
  --save_path /export/scratch/sheid/similarty_matrix/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/epoch_4_step_152001/


  CUDA_VISIBLE_DEVICES=5 python /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/scripts/inference2.py \
  --model_path /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/checkpoints/epoch_5_step_190001.pth \
  --add_red_hidd_size_factor 2 \
  --add_num_head 8 \
  --add_param_blocks 14 \
  --add_mlp_ratio 2.0 \
  --transformer_blocks 17 15 8 20 11 16  \
  --save_path /export/scratch/sheid/similarty_matrix/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16/epoch_5_step_190001/