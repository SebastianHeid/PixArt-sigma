available_gpus=(0)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12337 \
          /home/hd/hd_hd/hd_om233/partially_removal/PixArt-sigma/train_scripts/train.py \
          /home/hd/hd_hd/hd_om233/partially_removal/PixArt-sigma/configs/pixart_sigma_config/partially_block_removal/PixArt_sigma_xl2_img512_laion_17_15_8.py \
           --work-dir /gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/unsplash/test \

done