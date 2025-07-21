
available_gpus=(3,4,5,6)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12331 \
          /export/home/sheid/skip_connection_pixart/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/skip_connection_pixart/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_image_net.py \
          --work-dir /export/data/sheid/pixart/PixArt_sigma_xl2_img512_image_net_2 \
        
done