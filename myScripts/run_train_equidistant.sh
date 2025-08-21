# ----- error logging -----
LOGDIR=/export/home/sheid/PixArt-sigma/distillation/error_equidistant_attempt/
mkdir -p "$LOGDIR"
ERRFILE="$LOGDIR/distill_errors_$(date +%Y%m%d_%H%M%S).txt"

# capture all stderr from here on; also echo to terminal
exec 2> >(tee -a "$ERRFILE" >&2)
# if you prefer silent logging (no terminal echo), use:
# exec 2>>"$ERRFILE"
# --------------------------------


available_gpus=(5,7)
# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8.py \
#           --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8 \

# done

# for gpu in "${available_gpus[@]}"; do
#   export CUDA_VISIBLE_DEVICES=$gpu
#  python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
#           /export/home/sheid/PixArt-sigma/train_scripts/train.py \
#           /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_finetuning.py \
#           --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_finetuning \

# done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_finetuning_on_Pixart.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_finetuning_on_Pixart \

done



for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12 \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_finetuning.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_finetuning_on_Pixart.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_finetuning_on_Pixart \

done



for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14 \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_finetuning.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_finetuning_on_Pixart.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_finetuning_on_Pixart \

done



for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16 \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_finetuning.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_finetuning_on_Pixart.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_finetuning_on_Pixart \

done



for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18 \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_finetuning.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_finetuning_on_Pixart.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_finetuning_on_Pixart \

done



for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20 \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_finetuning.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_finetuning_on_Pixart.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_finetuning_on_Pixart \

done



for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22 \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_finetuning.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_finetuning_on_Pixart.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_finetuning_on_Pixart \

done



for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_24.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_24 \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2- --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_24_finetuning.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_24_finetuning \

done

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12332 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          /export/home/sheid/PixArt-sigma/configs/pixart_sigma_config/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_24_finetuning_on_Pixart.py \
          --work-dir /export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8_10_12_14_16_18_20_22_24_finetuning_on_Pixart \

done