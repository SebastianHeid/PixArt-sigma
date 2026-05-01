_base_ = ["../../../PixArt_xl2_internal.py"]
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
mixed_precision = "bf16"  # ['fp16', 'no', 'bf16']
fp32_attention = True
current_pruned_load_from = "/export/data/sheid/pixart/partially_removal_individual_compression/compression_strength/21_blocks/Second_Iteration_large/checkpoints/epoch_1_step_38000.pth"  
load_from =  "/export/data/sheid/pixart/partially_removal_individual_compression/compression_strength/21_blocks/Second_Iteration_large/checkpoints/epoch_1_step_38000.pth"   # https://huggingface.co/PixArt-alpha/PixArt-Sigma
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
compression_ratio_mlp=  [0.2, 0.1957, 0.1914, 0.1871, 0.1829, 0.1786, 0.1743, 0.17, 0.1657, 0.1614, 0.1571, 0.1529, 0.1486, 0.1443, 0.14, 0.1357, 0.1314, 0.1271, 0.1229, 0.1186, 0.1143]
compression_ratio_attn=  [0.2, 0.1957, 0.1914, 0.1871, 0.1829, 0.1786, 0.1743, 0.17, 0.1657, 0.1614, 0.1571, 0.1529, 0.1486, 0.1443, 0.14, 0.1357, 0.1314, 0.1271, 0.1229, 0.1186, 0.1143]
compression_ratio_cross_attn= [0.2, 0.1957, 0.1914, 0.1871, 0.1829, 0.1786, 0.1743, 0.17, 0.1657, 0.1614, 0.1571, 0.1529, 0.1486, 0.1443, 0.14, 0.1357, 0.1314, 0.1271, 0.1229, 0.1186, 0.1143]


transformer_blocks_mlp = [1, 0, 4, 2, 20, 3, 27, 5, 7, 6, 17, 25, 14, 22, 9, 13, 21, 11, 10, 8, 26]
transformer_blocks_attn = [1, 0, 4, 2, 20, 3, 27, 5, 7, 6, 17, 25, 14, 22, 9, 13, 21, 11, 10, 8, 26]
transformer_blocks_cross_attn =  [1, 0, 4, 2, 20, 3, 27, 5, 7, 6, 17, 25, 14, 22, 9, 13, 21, 11, 10, 8, 26]


transformer_blocks_mlp_new = [16, 27, 7, 4, 6, 1, 0, 2, 9, 21, 5, 19, 17, 11, 10, 3, 26, 20, 13, 25, 22]
transformer_blocks_attn_new = [16, 27, 7, 4, 6, 1, 0, 2, 9, 21, 5, 19, 17, 11, 10, 3, 26, 20, 13, 25, 22]
transformer_blocks_cross_attn_new =  [16, 27, 7, 4, 6, 1, 0, 2, 9, 21, 5, 19, 17, 11, 10, 3, 26, 20, 13, 25, 22]
compression_ratio_mlp_new =  [0.2, 0.3381, 0.3298, 0.3492, 0.3236, 0.3535, 0.3487, 0.3404, 0.3008, 0.2924, 0.3224, 0.182, 0.3092, 0.2831, 0.2783, 0.3228, 0.2683, 0.3236, 0.2832, 0.296, 0.2875]
compression_ratio_attn_new =  [0.2, 0.3381, 0.3298, 0.3492, 0.3236, 0.3535, 0.3487, 0.3404, 0.3008, 0.2924, 0.3224, 0.182, 0.3092, 0.2831, 0.2783, 0.3228, 0.2683, 0.3236, 0.2832, 0.296, 0.2875]
