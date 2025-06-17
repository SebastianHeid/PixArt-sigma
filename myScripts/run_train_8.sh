
available_gpus=(5)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12331 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning_train_on_pixart_generated_images_feat_based_on_feat.py \
          --work-dir /export/data/sheid/pixart/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning_train_on_pixart_generated_images_feat_based_on_feat \
        
done