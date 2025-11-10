#!/bin/bash
#SBATCH --job-name=GRASPFast
#SBATCH --output=/home/hd/hd_hd/hd_om233/slurm/pixart/partially_removed/std_out/GRASPFast.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/slurm/pixart/partially_removed/std_error/GRASPFast.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=48:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:1,gpumem_per_gpu:30GB
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


export CUDA_VISIBLE_DEVICES=2

 python /export/home/sheid/GRASP/PixArt-sigma/train_scripts/grasp_diff_pruning.py \
        --config /export/home/sheid/GRASP/PixArt-sigma/configs/pixart_sigma_config/partially_block_removal/GRASP/Diff_pruning_ind_layers.py



