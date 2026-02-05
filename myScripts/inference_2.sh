#!/bin/bash
#SBATCH --job-name=ranking_100
#SBATCH --output=/home/hd/hd_hd/hd_om233/slurm/pixart/std_out/ranking.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/slurm/pixart/std_error/ranking.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=60G  
#SBATCH --time=48:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1    


#--------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixart
#--------------------
# JOB EXECUTION
#--------------------
python /home/hd/hd_hd/hd_om233/partially_removal_individual_compression/PixArt-sigma/scripts/inference2.py \
  --save_path /gpfs/bwfor/work/ws/hd_om233-flux/pixart/block_evaluation/partially_removal_individual_compression/block_analysis/test_1k/original_100_2 \
  --config_path /home/hd/hd_hd/hd_om233/partially_removal_individual_compression/PixArt-sigma/block_analysis_eval/test_1k_images/org_100.yaml \
  --txt_file /home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion.json
