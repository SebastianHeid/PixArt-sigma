available_gpus=(0,1,3)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=3 --master_port=12337 \
          /export/home/sheid/skip_connection_pixart/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/skip_connection_pixart/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion2M_skipConnection.py \
          --work-dir /export/data/sheid/pixart/PixArt_sigma_xl2_img512_laion2M_skipConnection_continued \

done