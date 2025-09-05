
# ----- error logging -----
LOGDIR=/export/home/sheid/PixArt-sigma/distillation/error_shortcut_training/
mkdir -p "$LOGDIR"
ERRFILE="$LOGDIR/distill_errors_$(date +%Y%m%d_%H%M%S).txt"

# capture all stderr from here on; also echo to terminal
exec 2> >(tee -a "$ERRFILE" >&2)
# if you prefer silent logging (no terminal echo), use:
# exec 2>>"$ERRFILE"
# --------------------------------

available_gpus=(5)
# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning_trained_on_pixart_generated_images.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_finetuning_trained_on_pixart_generated_images \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_finetuning.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_finetuning \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_finetuning_trained_on_pixart_generated_images.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_finetuning_trained_on_pixart_generated_images \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           //export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16 \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning_old_test.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning_old_test \

# done







# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning_trained_on_pixart_generated_images.py\
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_finetuning_trained_on_pixart_generated_images \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_finetuning.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_finetuning \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_finetuning_trained_on_pixart_generated_images.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_finetuning_trained_on_pixart_generated_images \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_finetuning.py \
#           --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_finetuning \

# done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_finetuning_trained_on_pixart_generated_images \

done


for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_finetuning_trained_on_pixart_generated_images \

done


for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_finetuning.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_finetuning_trained_on_pixart_generated_images \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_finetuning.py\
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12387 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_finetuning_trained_on_pixart_generated_images.py \
          --work-dir /export/data/sheid/pixart/shortcut_training/PixArt_sigma_xl2_img512_laion_17_15_8_20_11_16_12_23_21_18_24_7_13_4_finetuning_trained_on_pixart_generated_images \

done





for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12331 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/third_distillation_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_12_finetuning.py \
          --work-dir /export/data/sheid/pixart/third_pruning_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_12_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12331 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/third_distillation_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_12_finetuning.py \
          --work-dir /export/data/sheid/pixart/third_pruning_attempt/PixArt_sigma_xl2_img512_laion_17_15_8_20_12_finetuning \

done