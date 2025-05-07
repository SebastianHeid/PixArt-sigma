
available_gpus=(1)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12340 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion_all_int_losses.py \
          --load-from /export/scratch/sheid/pixart/PixArt-Sigma-XL-2-512-MS.pth \
          --work-dir /export/data/sheid/pixart/epx_lr6_all_int_losses \
          --debug
done