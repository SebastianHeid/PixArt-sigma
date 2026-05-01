_base_ = ["../../../PixArt_xl2_internal.py"]
data_root = "pixart-sigma-toy-dataset"
image_list_json = ["data_info.json"]

data = dict(
    type="InternalDataMSSigma",
    root="/export/data/vislearn/rother_subgroup/sheid/pixart/pixart_generated_images/feature_pixart",
    img_root="/export/data/vislearn/rother_subgroup/sheid/pixart/pixart_generated_images/feature_pixart",   
    image_list_json=image_list_json,
    transform="default_train",
    load_vae_feat=True,
    load_t5_feat=True,
)
image_size = 512
# model setting
model = "PixArtMS_XL_2"
mixed_precision = "bf16"  # ['fp16', 'no', 'bf16']
fp32_attention = True
current_pruned_load_from = "/export/data/sheid/pixart/partially_removal_individual_compression/compression_strength/7_blocks/Fifth_Iteration_finetuning/checkpoints/epoch_1_step_38000.pth"  
load_from =  "/export/data/sheid/pixart/partially_removal_individual_compression/compression_strength/7_blocks/Fifth_Iteration_finetuning/checkpoints/epoch_1_step_38000.pth"   # https://huggingface.co/PixArt-alpha/PixArt-Sigma
pruned_load_from=None
ref_load_from =  "/export/scratch/sheid/pixart/PixArt-Sigma-XL-2-512-MS.pth"  
resume_from = None
vae_pretrained = (
   "/export/scratch/sheid/pixart/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
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


compression_ratio_mlp=  [0.8175, 0.8953, 0.8497, 0.4714, 0.4286, 0.6754, 0.6879, 0.6, 0.8857, 0.8838, 0.8293, 0.8302, 0.8, 0.6177, 0.5813, 0.8, 0.603, 0.5373, 0.406]
compression_ratio_attn=  [0.8175, 0.8953, 0.8497, 0.4714, 0.4286, 0.6754, 0.6879, 0.6, 0.8857, 0.8838, 0.8293, 0.8302, 0.8, 0.6177, 0.5813, 0.8, 0.603, 0.5373, 0.406]
compression_ratio_cross_attn= [0.8175, 0.8953, 0.8497, 0.4714, 0.4286, 0.6754, 0.6879, 0.6, 0.8857, 0.8838, 0.8293, 0.8302, 0.8, 0.6177, 0.5813, 0.8, 0.603, 0.5373, 0.406]


transformer_blocks_mlp = [1, 0, 4, 2, 20, 3, 27, 25, 7, 6, 17, 9, 5, 19, 21, 16, 8, 23, 26]
transformer_blocks_attn = [1, 0, 4, 2, 20, 3, 27, 25, 7, 6, 17, 9, 5, 19, 21, 16, 8, 23, 26]
transformer_blocks_cross_attn =  [1, 0, 4, 2, 20, 3, 27, 25, 7, 6, 17, 9, 5, 19, 21, 16, 8, 23, 26]



transformer_blocks_mlp_new = [11, 18, 13, 8, 22, 14, 7]
transformer_blocks_attn_new = [11, 18, 13, 8, 22, 14, 7]
transformer_blocks_cross_attn_new =  [11, 18, 13, 8, 22, 14, 7]
compression_ratio_mlp_new =  [0.8, 0.7201, 0.6402, 0.8254, 0.4803, 0.4004, 0.9223]
compression_ratio_attn_new =  [0.8, 0.7201, 0.6402, 0.8254, 0.4803, 0.4004, 0.9223]