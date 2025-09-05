
available_gpus=(5,6)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12335 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/cityscapes/PixArt_sigma_xl2_img512_laion_dist_17_15_8_20_11_16.py \
          --work-dir /export/data/sheid/pixart/cityscapes/PixArt_sigma_xl2_img512_laion_dist_17_15_8_20_11_16/ \

done