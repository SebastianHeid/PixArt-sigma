#!/bin/bash
#SBATCH --job-name=block_analysis_partially_removal_individual_compression_linear_linear
#SBATCH --output=/home/hd/hd_hd/hd_om233/slurm/pixart/std_out/partially_removed/block_analysis_partially_removal_individual_compression_linear.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/slurm/pixart/std_error/partially_removed/block_analysis_partially_removal_individual_compression_linear.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G  
#SBATCH --time=24:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:1,gpumem_per_gpu:25GB
#SBATCH --ntasks=1    

#--------------------
# JOB EXECUTION
#--------------------

echo "========================================================="
echo "Starting job execution"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $SLURM_JOB_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================="


#--------------------
# CONDA SETUP
#--------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixart


python /home/hd/hd_hd/hd_om233/partially_removal_individual_compression/PixArt-sigma/scripts/block_evaluation.py \
    --model_path /gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/partially_removal_individual_compression/linear_compression/First_Iteration_finetuning_on_pixart/checkpoints/epoch_2_step_12500.pth \
    --config_path /home/hd/hd_hd/hd_om233/partially_removal_individual_compression/PixArt-sigma/block_analysis_eval/linear/first_removal_stage_large.yaml
