_base_ = ["../../PixArt_xl2_internal.py"]
data_root = "pixart-sigma-toy-dataset"
image_list_json = ["data_info_train.json"]
data = dict(
    type="InternalDataMSSigma",
    root="/export/data/vislearn/rother_subgroup/sheid/pixart/mapillary",
    img_root="/export/data/vislearn/rother_subgroup/dzavadsk/datasets/mapillary/training/",
    image_list_json=image_list_json,
    transform="default_train",
    load_vae_feat=False,
    load_t5_feat=False,
    load_img_vae_feat=False,
)

image_size = 512

# model setting
model = "PixArtMS_XL_2"
mixed_precision = "bf16"  # ['fp16', 'no', 'bf16']
fp32_attention = False
load_from = "/export/scratch/sheid/pixart/PixArt-Sigma-XL-2-512-MS.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
resume_from = None
vae_pretrained = (
    "/export/scratch/sheid/pixart/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
)
aspect_ratio_type = "ASPECT_RATIO_512"
multi_scale = True  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 3
train_batch_size = 16  # 48 as default
num_epochs = 20  # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.5
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
log_interval_stable_loss = 1000
save_model_epochs = 1
work_dir = "output/debug"

# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 1.0
model_max_length = 300
class_dropout_prob = 0.1

