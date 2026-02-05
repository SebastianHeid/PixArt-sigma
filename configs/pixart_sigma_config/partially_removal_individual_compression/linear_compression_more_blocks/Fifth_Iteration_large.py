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
current_pruned_load_from=None
load_from = "/gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/partially_removal_individual_compression/linear_compression_more_blocks/Fourth_Iteration_finetuning_on_pixart/checkpoints/epoch_2_step_12500.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
pruned_load_from =  "/gpfs/bwfor/work/ws/hd_om233-flux/pixart/model_distillation/partially_removal_individual_compression/linear_compression_more_blocks/Fourth_Iteration_finetuning_on_pixart/checkpoints/epoch_2_step_12500.pth"
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
trainable_blocks = [18, 7, 10, 27, 1, 17, 19, 3, 8, 5, 4, 0, 22, 20, 11, 13, 14, 21, 9, 6]
# wenn ich hier eine Block hinzufüge, dann funktioniert es nicht mehr


#Compress mlp, attn, cross_attn via SVD
compression_ratio_mlp=  [0.76208, 0.7351, 0.69453, 0.66287, 0.5562, 0.51253, 0.64726, 0.67153, 0.69671, 0.5354, 0.57138, 0.35298, 0.38344, 0.28579, 0.59682, 0.26832, 0.59349, 0.44434, 0.3647, 0.60813, 0.60504, 0.47187, 0.3672, 0.35]
compression_ratio_attn=  [0.76208, 0.7351, 0.69453, 0.66287, 0.5562, 0.51253, 0.64726, 0.67153, 0.69671, 0.5354, 0.57138, 0.35298, 0.38344, 0.28579, 0.59682, 0.26832, 0.59349, 0.44434, 0.3647, 0.60813, 0.60504, 0.47187, 0.3672, 0.35]
compression_ratio_cross_attn= [0.76208, 0.7351, 0.69453, 0.66287, 0.5562, 0.51253, 0.64726, 0.67153, 0.69671, 0.5354, 0.57138, 0.35298, 0.38344, 0.28579, 0.59682, 0.26832, 0.59349, 0.44434, 0.3647, 0.60813, 0.60504, 0.47187, 0.3672, 0.35]


transformer_blocks_mlp = [1, 0, 4, 2, 20, 3, 27, 5, 7, 6, 17, 25, 14, 22, 9, 13, 21, 11, 10, 8, 16, 26, 18, 19]
transformer_blocks_attn = [1, 0, 4, 2, 20, 3, 27, 5, 7, 6, 17, 25, 14, 22, 9, 13, 21, 11, 10, 8, 16, 26, 18, 19]
transformer_blocks_cross_attn =  [1, 0, 4, 2, 20, 3, 27, 5, 7, 6, 17, 25, 14, 22, 9, 13, 21, 11, 10, 8, 16, 26, 18, 19]



transformer_blocks_mlp_new = [18, 7, 10, 27, 1, 17, 19, 3, 8, 5, 4, 0, 22, 20, 11, 13, 14, 21, 9, 6]
transformer_blocks_attn_new = [18, 7, 10, 27, 1, 17, 19, 3, 8, 5, 4, 0, 22, 20, 11, 13, 14, 21, 9, 6]
transformer_blocks_cross_attn_new =  [18, 7, 10, 27, 1, 17, 19, 3, 8, 5, 4, 0, 22, 20, 11, 13, 14, 21, 9, 6]
compression_ratio_mlp_new =  [0.62032, 0.81681, 0.61374, 0.78412, 0.85344, 0.73426, 0.5944, 0.69387, 0.75234, 0.79109, 0.8045, 0.8294, 0.53719, 0.71064, 0.63549, 0.51709, 0.5906, 0.72845, 0.72906, 0.68593]
compression_ratio_attn_new =  [0.62032, 0.81681, 0.61374, 0.78412, 0.85344, 0.73426, 0.5944, 0.69387, 0.75234, 0.79109, 0.8045, 0.8294, 0.53719, 0.71064, 0.63549, 0.51709, 0.5906, 0.72845, 0.72906, 0.68593]
