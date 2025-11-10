# python /home/hd/hd_hd/hd_om233/GRASP/PixArt-sigma/scripts/inference2.py \
#   --save_path /home/hd/hd_hd/hd_om233/GRASP/images/11_blocks/magnitudeSVD/ \
#   --model_path /gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/PixArt-Sigma-XL-2-512-MS.pth \
#   --org_model_path /gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/PixArt-Sigma-XL-2-512-MS.pth

export CUDA_VISIBLE_DEVICES=2

python /export/home/sheid/GRASP/PixArt-sigma/scripts/inference2.py \
  --save_path /export/home/sheid/GRASP/images/11_blocks/importanceSVD \
  --model_path /export/home/sheid/GRASP/output/11_blocksFast/compressed_model.safetensors \
