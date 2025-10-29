#!/bin/bash
#SBATCH --job-name=Eval
#SBATCH --output=/home/hd/hd_hd/hd_om233/partially_removal/PixArt-sigma/tools/std_out/feature_generation.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/partially_removal/PixArt-sigma/tools/std_error/feature_generation.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=24:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:1,gpumem_per_gpu:20GB
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
available_gpus=(0)

# python /home/hd/hd_hd/hd_om233/partially_removal/PixArt-sigma/tools/extract_features.py \
# --run_t5_feature_extract \
# --t5_json_path /home/hd/hd_hd/hd_om233/flux/dataset_config/json_laion2M_flux_10k_fixed.json \
# --t5_models_dir /gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/pixart_sigma_sdxlvae_T5_diffusers \
# --t5_save_root /gpfs/bwfor/work/ws/hd_om233-flux/Laion/feature_pixart_t5

python /home/hd/hd_hd/hd_om233/partially_removal/PixArt-sigma/tools/extract_features.py \
--run_vae_feature_extract \
--dataset_root /gpfs/bwfor/work/ws/hd_om233-flux/Laion/10k_hed_edges \
--vae_json_file /home/hd/hd_hd/hd_om233/flux/dataset_config/json_laion2M_flux_10k_fixed.json \
--vae_models_dir /gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/pixart_sigma_sdxlvae_T5_diffusers \
--vae_save_root /gpfs/bwfor/work/ws/hd_om233-flux/Laion/feature_pixart_hed_edges/ 

# python /home/hd/hd_hd/hd_om233/partially_removal/PixArt-sigma/tools/extract_features.py \
# --run_vae_feature_extract \
# --dataset_root /gpfs/lsdf02/sd23g007/datasets/laion2M_flux/images/ \
# --vae_json_file /home/hd/hd_hd/hd_om233/flux/dataset_config/json_laion2M_flux_10k_fixed.json \
# --vae_models_dir /gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/pixart_sigma_sdxlvae_T5_diffusers \
# --vae_save_root /gpfs/bwfor/work/ws/hd_om233-flux/Laion/feature_pixart_vae/ 