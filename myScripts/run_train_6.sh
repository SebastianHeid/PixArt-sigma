
available_gpus=(3)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning.py\
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning \

done

