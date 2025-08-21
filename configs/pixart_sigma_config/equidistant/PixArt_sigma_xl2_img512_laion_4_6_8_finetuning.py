_base_ = ["../PixArt_xl2_internal.py"]
data_root = "pixart-sigma-toy-dataset"
image_list_json = ["data_info.json"]
data = dict(
    type="InternalDataMSSigma",
    root="/export/data/vislearn/rother_subgroup/sheid/pixart/laion2M/feature_pixart",
    img_root="/export/data/vislearn/rother_subgroup/rother_datasets/LaionAE/laion2B-en-art_512/",
    image_list_json=image_list_json,
    transform="default_train",
    load_vae_feat=True,
    load_t5_feat=True,
    load_img_vae_feat=False,
)
image_size = 512
# model setting
model = "PixArtMS_XL_2"
mixed_precision = "fp16"  # ['fp16', 'no', 'bf16']
fp32_attention = False
  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
load_from = '/export/data/sheid/pixart/equidistant/PixArt_sigma_xl2_img512_laion_4_6_8/checkpoints/epoch_1_step_37990.pth'
ref_load_from = "/export/scratch/sheid/pixart/PixArt-Sigma-XL-2-512-MS.pth" 
resume_from = None
vae_pretrained = (
    "/export/scratch/sheid/pixart/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
)
aspect_ratio_type = "ASPECT_RATIO_512"
multi_scale = False  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 4
train_batch_size = 8  # 48 as default
num_epochs = 1  # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(
    type="CAMEWrapper",
    lr=2e-5,
    weight_decay=0.03,
    betas=(0.9, 0.999, 0.9999),
    eps=(1e-30, 1e-16),
)
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 500
visualize = True
log_interval = 20
save_model_epochs = 1
save_model_steps = 37990
work_dir = "output/debug"


# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 1.0
model_max_length = 300
class_dropout_prob = 0.1

# Intermediate loss
intermediate_loss_flag = True
intermediate_loss_blocks = [4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  21, 22, 23,  24, 25, 26, 27]
final_output_loss_flag = True
org_loss_flag = False

# Modfication of Model
transformer_blocks = [4, 6, 8]
trainable_blocks = []
# wenn ich hier eine Block hinzufüge, dann funktioniert es nicht mehr

reserve_memory=True