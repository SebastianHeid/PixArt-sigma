#!/bin/bash
#SBATCH --job-name=GRASP
#SBATCH --output=/home/hd/hd_hd/hd_om233/slurm/pixart/partially_removed/std_out/GRASP.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/slurm/pixart/partially_removed/std_error/GRASP.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=80G  
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




 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12317 \
           /home/hd/hd_hd/hd_om233/GRASP/PixArt-sigma/train_scripts/train.py \
         /home/hd/hd_hd/hd_om233/GRASP/PixArt-sigma/configs/pixart_sigma_config/partially_block_removal/GRASP/Testing.py \
          --work-dir /gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/partially_block_removal/test \


