_base_ = ["../PixArt_xl2_internal.py"]
data_root = "pixart-sigma-toy-dataset"
image_list_json = ["data_info.json"]

data = dict(
    type="InternalDataMSSigma",
    root="/export/data/vislearn/rother_subgroup/sheid/pixart/laion",
    img_root="/export/data/vislearn/rother_subgroup/dzavadsk/datasets/laion/subset_250/images/",
    image_list_json=image_list_json,
    transform="default_train",
    load_vae_feat=False,
    load_t5_feat=False,
)
image_size = 512

# model setting
model = "PixArtMS_XL_2"
mixed_precision = "fp16"  # ['fp16', 'no', 'bf16']
fp32_attention = True
load_from = "/export/data/sheid/pixart/PixArt_sigma_xl2_img512_laion_o_4_7_16_19_22_n_11/checkpoints/epoch_1_step_304707.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
ref_load_from = "/export/scratch/sheid/pixart/PixArt-Sigma-XL-2-512-MS.pth" 
resume_from = None
vae_pretrained = (
    "/export/scratch/sheid/pixart/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
)
aspect_ratio_type = "ASPECT_RATIO_512"
multi_scale = True  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 0
train_batch_size = 32  # 48 as default
num_epochs = 1  # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(
    type="CAMEWrapper",
    lr=2e-6,
    weight_decay=0.0,
    betas=(0.9, 0.999, 0.9999),
    eps=(1e-30, 1e-16),
)
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 500
visualize = True
log_interval = 20
save_model_epochs = 1
save_model_steps = 50000
work_dir = "output/debug"

# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 1.0
model_max_length = 300
class_dropout_prob = 0.1

# Intermediate loss
intermediate_loss_flag = True
intermediate_loss_blocks = [4,5,7,8,9,11,12,13,14,16,17,19,20,22,23,24,25,26,27]
final_output_loss_flag = True
org_loss_flag = False

# Modfication of Model
transformer_blocks = [4,7,11,16,19,22]
trainable_blocks = []
# wenn ich hier eine Block hinzufüge, dann funktioniert es nicht mehr
