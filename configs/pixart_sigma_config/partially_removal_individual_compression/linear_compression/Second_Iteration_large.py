_base_ = ["../../../PixArt_xl2_internal.py"]
data_root = "pixart-sigma-toy-dataset"
image_list_json = ["/gpfs/lsdf02/sd23g007/datasets/laion2M_pixart/Laion/laion2M/feature_pixart/data_info.json"]

data = dict(
    type="InternalDataMSSigma",
    root="/gpfs/lsdf02/sd23g007/datasets/laion2M_pixart/Laion/laion2M/feature_pixart/",
    image_list_json=image_list_json,
    transform="default_train",
    load_vae_feat=True,
    load_t5_feat=True,
    load_img_vae_feat=False,
)
image_size = 512
# model setting
model = "PixArtMS_XL_2"
mixed_precision = "bf16"  # ['fp16', 'no', 'bf16']
fp32_attention = True
load_from = "/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/PixArt-Sigma-XL-2-512-MS.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
pruned_load_from =  "/gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/partially_removal_individual_compression/linear_compression/First_Iteration_finetuning_on_pixart/checkpoints/epoch_2_step_12500.pth"
ref_load_from = "/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/PixArt-Sigma-XL-2-512-MS.pth" 
resume_from = None
vae_pretrained = (
    "/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
)
aspect_ratio_type = "ASPECT_RATIO_512"
multi_scale = False  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 8
train_batch_size = 16  # 48 as default
num_epochs = 1  # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(
    type="CAMEWrapper",
    lr=2e-6,
    weight_decay=0.03,
    betas=(0.9, 0.999, 0.9999),
    eps=(1e-30, 1e-16),
)
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 500
visualize = True
log_interval = 20
save_model_epochs = 1
save_model_steps = 38000
work_dir = "output/debug"


# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 1.0
model_max_length = 300
class_dropout_prob = 0.1

# Intermediate loss
intermediate_loss_flag = True
intermediate_loss_blocks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
final_output_loss_flag = True
org_loss_flag = False

# Modfication of Model
transformer_blocks = []
trainable_blocks = [8, 1, 5, 27, 20, 4, 0, 7, 14, 3, 18, 25, 17, 21, 26]
# wenn ich hier eine Block hinzufüge, dann funktioniert es nicht mehr


#Compress mlp, attn, cross_attn via SVD
compression_ratio_mlp=  [0.5, 0.455, 0.41, 0.365, 0.32, 0.275, 0.23, 0.185, 0.14, 0.095, 0.05, 0.005]
compression_ratio_attn=  [0.5, 0.455, 0.41, 0.365, 0.32, 0.275, 0.23, 0.185, 0.14, 0.095, 0.05, 0.005]
compression_ratio_cross_attn= [0.5, 0.455, 0.41, 0.365, 0.32, 0.275, 0.23, 0.185, 0.14, 0.095, 0.05, 0.005]


transformer_blocks_mlp = [2, 1, 4, 0, 6, 5, 17, 27, 3, 16, 7, 21]
transformer_blocks_attn = [2, 1, 4, 0, 6, 5, 17, 27, 3, 16, 7, 21]
transformer_blocks_cross_attn =  [2, 1, 4, 0, 6, 5, 17, 27, 3, 16, 7, 21]


transformer_blocks_mlp_new = [8, 1, 5, 27, 20, 4, 0, 7, 14, 3, 18, 25, 17, 21, 26]
transformer_blocks_attn_new = [8, 1, 5, 27, 20, 4, 0, 7, 14, 3, 18, 25, 17, 21, 26]
transformer_blocks_cross_attn_new =  [8, 1, 5, 27, 20, 4, 0, 7, 14, 3, 18, 25, 17, 21, 26]
compression_ratio_mlp_new =  [0.5, 0.71115, 0.594, 0.51915, 0.38, 0.6165, 0.5682, 0.3255, 0.26, 0.3378, 0.2, 0.17, 0.3378, 0.11445, 0.08]
compression_ratio_attn_new =  [0.5, 0.71115, 0.594, 0.51915, 0.38, 0.6165, 0.5682, 0.3255, 0.26, 0.3378, 0.2, 0.17, 0.3378, 0.11445, 0.08]
