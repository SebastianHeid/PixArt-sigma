available_gpus=(3)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12337 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_14_manual_finetuning.py \
          --work-dir /export/data/sheid/pixart/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_14_manual_finetuning \

done