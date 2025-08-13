
available_gpus=(4,5)
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning_trained_on_pixart_generated_images.py\
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_finetuning.pyy \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_finetuning_trained_on_pixart_generated_images \

done









for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12331 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/third_distillation_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_12_finetuning.py \
          --work-dir /export/data/sheid/pixart/third_pruning_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_12_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12331 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/third_distillation_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_12_finetuning.py \
          --work-dir /export/data/sheid/pixart/third_pruning_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_12_finetuning \

done