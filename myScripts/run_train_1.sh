available_gpus=(7)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
  export TORCH_DISTRIBUTED_DEBUG=INFO
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12381 \
          /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/pixart_addNewBlocks/PixArt-sigma/configs/pixart_sigma_config/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13.py \
          --work-dir /export/data/sheid/pixart/add_blocks/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_full_block_t0 \

done