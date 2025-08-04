# ----- error logging -----
LOGDIR=/export/home/sheid/PixArt-sigma/distillation/error/
mkdir -p "$LOGDIR"
ERRFILE="$LOGDIR/distill_errors_$(date +%Y%m%d_%H%M%S).txt"

# capture all stderr from here on; also echo to terminal
exec 2> >(tee -a "$ERRFILE" >&2)
# if you prefer silent logging (no terminal echo), use:
# exec 2>>"$ERRFILE"
# --------------------------------


available_gpus=(1,2)


for idx in {0..5}; do
# Load YAML values into shell variables
eval $(python3 /export/home/sheid/PixArt-sigma/distillation/parse_yaml.py /export/home/sheid/MasterThesis_Evaluation/configs/PixArt/config_distillation_third_distillation_attempt.yaml)
echo "Loaded configuration for idx $idx"
# # Training Stage

if [ "$idx" -ne 0 ]; then
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=12337 \
    /export/home/sheid/PixArt-sigma/train_scripts/train.py \
    $CONFIG_FILE_TRAINING \
    --work-dir $WORK_DIR_TRAINING
done
fi

# Finetuning Stage
if [ "$idx" -ne 0 ]; then
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12336 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
         $CONFIG_FILE_FINETUNING \
          --work-dir $WORK_DIR_FINETUNING \

done
fi


# Finetuning on Pixart Images
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12335 \
          /export/home/sheid/PixArt-sigma/train_scripts/train.py \
          $CONFIG_FILE_FINETUNING_PIXART \
          --work-dir $WORK_DIR_FINETUNING_PIXART \

done


#convert ckpt to diffusers
for gpu in "${available_gpus[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu} python /export/home/sheid/PixArt-sigma/tools/convert_pixart_to_diffusers_distillation.py --config_path /export/home/sheid/MasterThesis_Evaluation/configs/PixArt/config_distillation_third_distillation_attempt.yaml
done



# compute best blocks to remove
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
python  /export/home/sheid/MasterThesis_Evaluation/block_analysis_distillation_clip.py
done

#compute best blocks to remove
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
python  /export/home/sheid/PixArt-sigma/distillation/create_eval_yaml.py
done


# update config_distillation.yamle file
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
python /export/home/sheid/PixArt-sigma/distillation/update_config_distillation.py
done




done