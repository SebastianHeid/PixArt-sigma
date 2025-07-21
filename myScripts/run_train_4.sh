available_gpus=(3,7)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12337 \
          /export/home/sheid/skip_connection_pixart/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/skip_connection_pixart/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion_o_17_15_8_20_11_14_n_12.py \
          --work-dir /export/data/sheid/pixart/PixArt_sigma_xl2_img512_laion_o_17_15_8_20_11_14_n_12 \

done