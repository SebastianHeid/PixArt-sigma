available_gpus=(7)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12337 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/mapillary/PixArt_sigma_xl2_img512_laion_org_model.py \
          --work-dir /export/data/sheid/pixart/mapillary/PixArt_sigma_xl2_img512_laion_org_model \

done