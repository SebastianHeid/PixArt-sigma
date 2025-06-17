
available_gpus=(0,7)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12337 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_pixart_generated_images_o_17_15_8_n_20_11.py \
          --work-dir /export/data/sheid/pixart/test \

done