available_gpus=(5,6)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12339 \
          /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/configs/pixart_sigma_config/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning.py \
          --work-dir /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning \

done


for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12339 \
          /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/configs/pixart_sigma_config/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning_on_pixart.py \
          --work-dir /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning_on_pixart \

done