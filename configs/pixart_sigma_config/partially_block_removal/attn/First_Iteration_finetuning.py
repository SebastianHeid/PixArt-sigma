_base_ = ["../../PixArt_xl2_internal.py"]
data_root = "pixart-sigma-toy-dataset"
image_list_json = ["/gpfs/bwfor/work/ws/hd_om233-flux/unsplash/data_info_1Mio.json"]

data = dict(
    type="InternalDataMSSigma",
    root="/gpfs/bwfor/work/ws/hd_om233-flux/unsplash/feature_pixart",
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
load_from = "/gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/partially_block_removal/attn/First_Iteration/checkpoints/epoch_1_step_38000.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
pruned_load_from =  "/gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/partially_block_removal/attn/First_Iteration/checkpoints/epoch_1_step_38000.pth"
ref_load_from = "/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/PixArt-Sigma-XL-2-512-MS.pth" 
resume_from = None
vae_pretrained = (
    "/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
)
aspect_ratio_type = "ASPECT_RATIO_512"
multi_scale = False  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 16
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
trainable_blocks = []
# wenn ich hier eine Block hinzufüge, dann funktioniert es nicht mehr


#Compress mlp, attn, cross_attn via SVD


transformer_blocks_mlp = []
rank_mlp = 512

transformer_blocks_attn = [20, 1, 0, 3, 18, 5, 4, 6, 17, 9, 7, 27, 10, 8, 25, 22, 2]
rank_attn = 128

transformer_blocks_cross_attn =  []
rank_cross_attn = 128