
available_gpus=(1,3)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12337 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_9.py \
          --work-dir /export/data/sheid/pixart/second_pruning_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_9 \

done


for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12336 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_9_finetuning.py \
          --work-dir /export/data/sheid/pixart/second_pruning_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_9_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12335 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_9_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/second_pruning_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_9_finetuning_trained_on_pixart_generated_images \

done


for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12337 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24.py \
          --work-dir /export/data/sheid/pixart/second_pruning_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24 \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12337 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24.py \
          --work-dir /export/data/sheid/pixart/second_pruning_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24 \

done