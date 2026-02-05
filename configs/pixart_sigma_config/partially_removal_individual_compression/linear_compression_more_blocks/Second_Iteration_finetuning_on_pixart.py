_base_ = ["../../../PixArt_xl2_internal.py"]
data_root = "pixart-sigma-toy-dataset"
image_list_json = ["data_info.json"]

data = dict(
    type="InternalDataMSSigma",
    root="/gpfs/bwfor/work/ws/hd_om233-flux/pixart/generatedImg/feature_pixart/",
    img_root="/export/data/vislearn/rother_subgroup/sheid/pixart/pixart_generated_images/images",
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
load_from = "/gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/partially_removal_individual_compression/linear_compression_more_blocks/Second_Iteration_finetuning/checkpoints/epoch_1_step_38000.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
current_pruned_load_from =  "/gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/partially_removal_individual_compression/linear_compression_more_blocks/Second_Iteration_finetuning/checkpoints/epoch_1_step_38000.pth"
ref_load_from = "/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/PixArt-Sigma-XL-2-512-MS.pth" 
pruned_load_from=None
resume_from = None
vae_pretrained = (
    "/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
)
aspect_ratio_type = "ASPECT_RATIO_512"
multi_scale = False  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 8
train_batch_size = 8  # 48 as default
num_epochs = 2  # 3
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
save_model_steps = 12500
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
trainable_blocks = []
# wenn ich hier eine Block hinzufüge, dann funktioniert es nicht mehr


#Compress mlp, attn, cross_attn via SVD
compression_ratio_mlp=  [0.3, 0.285, 0.27, 0.255, 0.24, 0.225, 0.21, 0.195, 0.18, 0.165, 0.15, 0.135, 0.12, 0.105, 0.09, 0.075, 0.06, 0.045, 0.03, 0.015]
compression_ratio_attn=  [0.3, 0.285, 0.27, 0.255, 0.24, 0.225, 0.21, 0.195, 0.18, 0.165, 0.15, 0.135, 0.12, 0.105, 0.09, 0.075, 0.06, 0.045, 0.03, 0.015]
compression_ratio_cross_attn= [0.3, 0.285, 0.27, 0.255, 0.24, 0.225, 0.21, 0.195, 0.18, 0.165, 0.15, 0.135, 0.12, 0.105, 0.09, 0.075, 0.06, 0.045, 0.03, 0.015]


transformer_blocks_mlp = [1, 0, 4, 2, 20, 3, 27, 5, 7, 6, 17, 25, 14, 22, 9, 13, 21, 11, 10, 8]
transformer_blocks_attn = [1, 0, 4, 2, 20, 3, 27, 5, 7, 6, 17, 25, 14, 22, 9, 13, 21, 11, 10, 8]
transformer_blocks_cross_attn =  [1, 0, 4, 2, 20, 3, 27, 5, 7, 6, 17, 25, 14, 22, 9, 13, 21, 11, 10, 8]



transformer_blocks_mlp_new = [8, 21, 1, 5, 25, 0, 7, 16, 6, 2, 27, 3, 10, 4, 17, 9, 26, 18, 11, 14]
transformer_blocks_attn_new = [8, 21, 1, 5, 25, 0, 7, 16, 6, 2, 27, 3, 10, 4, 17, 9, 26, 18, 11, 14]
transformer_blocks_cross_attn_new =  [8, 21, 1, 5, 25, 0, 7, 16, 6, 2, 27, 3, 10, 4, 17, 9, 26, 18, 11, 14]
compression_ratio_mlp_new =  [0.3105, 0.33072, 0.4932, 0.40752, 0.35298, 0.4566, 0.36696, 0.216, 0.33534, 0.39804, 0.3522, 0.3552, 0.18132, 0.37512, 0.2622, 0.1992, 0.108, 0.096, 0.12522, 0.18336]
compression_ratio_attn_new =  [0.3105, 0.33072, 0.4932, 0.40752, 0.35298, 0.4566, 0.36696, 0.216, 0.33534, 0.39804, 0.3522, 0.3552, 0.18132, 0.37512, 0.2622, 0.1992, 0.108, 0.096, 0.12522, 0.18336]
