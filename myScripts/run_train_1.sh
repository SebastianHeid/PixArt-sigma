
available_gpus=(0)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12317 \
          /home/hd/hd_hd/hd_om233/PixArt-sigma/train_scripts/train.py \
          /home/hd/hd_hd/hd_om233/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_image_net.py \
          --work-dir /gpfs/bwfor/work/ws/hd_om233-flux/pixart \


done