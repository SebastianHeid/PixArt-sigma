
available_gpus=(4,5)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/mapillary/PixArt_sigma_xl2_img512_laion_dist_17_15_8_20_11_16_12_23_21_18_24_7_13_4_9.py \
          --work-dir /export/data/sheid/pixart/mapillary/PixArt_sigma_xl2_img512_laion_dist_17_15_8_20_11_16_12_23_21_18_24_7_13_4_9 \
      
done