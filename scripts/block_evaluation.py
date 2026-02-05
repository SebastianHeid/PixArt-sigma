import os
import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings

warnings.filterwarnings("ignore")  # ignore warning
import argparse
import json
import re
import sys
from datetime import datetime

import diffusion.data.datasets.utils as ds_utils
import torch
from diffusers.models import AutoencoderKL
from diffusion import DPMS, IDDPM, SASolverSampler
from diffusion.data.datasets import get_chunks
from diffusion.model.modify_model import modify_model_base, modify_model_new, modify_model_current
from diffusion.model.nets import PixArt_XL_2, PixArtMS
from diffusion.model.utils import prepare_prompt_ar
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, read_config
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from tools.download import find_model

sys.path.append("/home/hd/hd_hd/hd_om233/partially_removal/MasterThesis_Evaluation")
sys.path.append("/home/hd/hd_hd/hd_om233/partially_removal/")
from MasterThesis_Evaluation.evaluation_CLIP_2 import compute_clip
from MasterThesis_Evaluation.evaluation_cmmd import compute_cmmd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument(
        "--pipeline_load_from", default="/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/pixart_sigma_sdxlvae_T5_diffusers",
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--txt_file', default='/home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion.json', type=str)
    parser.add_argument('--model_path', default="/gpfs/bwfor/work/ws/hd_om233-flux/model_pixart/PixArt-Sigma-XL-2-512-MS.pth", type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='mlp', type=str)
    parser.add_argument('--save_path', default='/home/hd/hd_hd/hd_om233/partially_removal/img_all_attn_r600', type=str,)
    parser.add_argument('--pe_interpolation', default=1.0, type=float)
    parser.add_argument('--config_path', default="/home/hd/hd_hd/hd_om233/partially_removal_individual_compression/PixArt-sigma/block_analysis_eval/first_removal_stage_large.yaml", type=str)
    parser.add_argument('--cross_attn', action='store_false', help='sd vae')
    parser.add_argument('--attn', action='store_false', help='sd vae')
    parser.add_argument('--mlp', action='store_false', help='sd vae')
    parser.add_argument('--total_blocks', action='store_true', help='sd vae')

    return parser.parse_args()

def best_network(idx, cmmd, cmmd_raw, block_list, config):
    
    
    sorted_cmmd  = sorted(cmmd.items(), key=lambda item: item[1])
    best_raw_cmmd = cmmd_raw[sorted_cmmd[0][0]]
    position_cmmd = {}
    total_position = {}

    
    for idx, (key, _) in enumerate(sorted_cmmd):
        position_cmmd[key] = idx

    
 
    
    for idx, key in enumerate(position_cmmd):
        total_position[key] = 0
        total_position[key] += position_cmmd[key]
  

    best_block = min(total_position, key=total_position.get)
    
    with open(config.log_path+"/results.txt", "a") as f: 
        f.write("Best block: " + str(best_block) + "\n")
        for i in range(len(block_list)):
            f.write("Block: " + str(block_list[i]) + " CMMD: " + str(cmmd[block_list[i]]) + " total position: " +str(total_position[block_list[i]]) + "\n")
    return best_block, best_raw_cmmd
    
    
def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

@torch.inference_mode()
def visualize( items,keys, bs, sample_steps, cfg_scale):

    for idx, chunk in enumerate(tqdm(list(get_chunks(items, bs)), unit='batch')):
        key = keys[idx]
        prompts = []
        if bs == 1:
            # save_path = os.path.join(save_root, f"{prompts[0][:100]}.jpg")
            # if os.path.exists(save_path):
            #     continue
            
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(chunk[0], base_ratios, device=device, show=False)  # ar for aspect ratio
     
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            for prompt in chunk:
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size
        print(f'prompts: {prompts[0]}')
        caption_token = tokenizer(prompts[0], max_length=max_sequence_length, padding="max_length", truncation=True,
                                  return_tensors="pt").to(device)
        caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
        emb_masks = caption_token.attention_mask

        caption_embs = caption_embs[:, None]
        null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]
        print(f'finish embedding')

        with torch.no_grad():

            if args.sampling_algo == 'iddpm':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            elif args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif args.sampling_algo == 'sa-solver':
                # Create sampling noise:
                n = len(prompts)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(4, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]
        

        samples = samples.to(weight_dtype)
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        #print(torch.min(samples), torch.max(samples))
        torch.cuda.empty_cache()
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        print(samples.shape
              )
        for i, sample in enumerate(samples):
            save_path = os.path.join(save_root, f"{key}.jpg")
            print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))
        del samples
        del caption_embs, caption_token,
        torch.cuda.empty_cache()
        


if __name__ == '__main__':
    args = get_args()
    config = read_config(args.config_path)
    args.model_path = config.model_path
    args.save_path = config.save_path
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']
    block_list = config.block_list
    removed_block_list = config.removed_blocks
    # only support fixed latent size currently
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    pe_interpolation = args.image_size / 512
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    # model setting
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    pe_interpolation = args.pe_interpolation if args.pe_interpolation > 0 else args.image_size / 512
    
    _save_path = args.save_path
    lpips_list = []
    cmmd_list = []
    clip_list = []
    
    base_ratios = getattr(ds_utils, f'ASPECT_RATIO_{args.image_size}', ds_utils.ASPECT_RATIO_1024)

    if args.sdvae:
        # pixart-alpha vae link: https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema
        vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device).to(weight_dtype)
    else:
        # pixart-Sigma vae link: https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae
        vae = AutoencoderKL.from_pretrained(f"{args.pipeline_load_from}/vae").to(device).to(weight_dtype)

    tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pipeline_load_from, subfolder="text_encoder").to(device)

    null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    current_removed_blocks = []
    current_compression_ratios = []
    for idx_removed in range(len(config.compression_ratios)):
        score = {}
        raw_cmmd_values = {}
        for idx in block_list:
            seed = args.seed
            set_env(seed)
            config.new_block = [idx]
            
            args.save_path = _save_path + "/it_" + str(idx_removed) +  "/" +  str(idx) + "/"
            
            model = PixArtMS(
                input_size=latent_size,
                pe_interpolation=pe_interpolation,
                micro_condition=micro_condition,
                model_max_length=max_sequence_length,
                skip_connections=True
            ).to(device)

        
            print("Generating sample from ckpt: %s" % args.model_path)
            model = modify_model_base(model, config)
            state_dict = find_model(args.model_path)
            missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
            print("missing: ", missing)
            print("unexpected: ", unexpected)
            model = modify_model_current(config, model, current_removed_blocks, current_compression_ratios)
            model = modify_model_new(model, config, idx_removed)
            model.eval()
            model = model.to(device)
            model.to(weight_dtype)
            
            work_dir = os.path.join(*args.model_path.split('/')[:-2])
            work_dir = '/'+work_dir if args.model_path[0] == '/' else work_dir

            # data setting
            # with open(args.txt_file, 'r') as f:
            #     items = [item.strip() for item in f.readlines()]
            print(config.txt_file)
            with open(config.txt_file, "r") as f:
                data = json.load(f)

            # Get string values, assuming each dict has one key-value pair
            items = [d for d in data.values()]
            keys = [k for k in data.keys()]
            print("Len Items", len(items))
            # img save setting
            try:
                epoch_name = re.search(r'.*epoch_(\d+).*', args.model_path).group(1)
                step_name = re.search(r'.*step_(\d+).*', args.model_path).group(1)
            except:
                epoch_name = 'unknown'
                step_name = 'unknown'
            img_save_dir = os.path.join(work_dir, 'vis')
            os.umask(0o000)  # file permission: 666; dir permission: 777
            os.makedirs(img_save_dir, exist_ok=True)

            #save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_{args.dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}_seed{seed}")
            save_root = args.save_path
            os.makedirs(save_root, exist_ok=True)
            
            visualize( items,keys, args.bs, sample_steps, args.cfg_scale)
            
            cmmd = compute_cmmd(config.ref_path, args.save_path).item()
            print(cmmd)
            delta_cmmd = cmmd - config.base_cmmd
            if idx in config.transformer_blocks_mlp:
                idx_ = config.transformer_blocks_mlp.index(idx)
                old_total_compression_ratio = config.compression_ratio_mlp[idx_]
                current_compression_ratio = config.compression_ratios[idx_removed]
                new_total_compression_ratio = 1 - (1-old_total_compression_ratio) * (1-current_compression_ratio)
                effective_comp_ratio = new_total_compression_ratio - old_total_compression_ratio
            else: 
                effective_comp_ratio = config.compression_ratios[idx_removed]
            
            print(effective_comp_ratio)
            delta_cmmd /= effective_comp_ratio
        
            score[idx] = delta_cmmd
            raw_cmmd_values[idx] = cmmd
            del model
            torch.cuda.empty_cache()
            print(f"GPU Memory after block {idx}: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            
         
            

        torch.cuda.empty_cache()
        os.makedirs(config.log_path, exist_ok=True)
        best_block, best_raw_cmmd = best_network(idx_removed,score,raw_cmmd_values , block_list, config)
        print(best_raw_cmmd)
        config.base_cmmd = best_raw_cmmd
        if best_block in config.transformer_blocks_mlp:
            idx_ = config.transformer_blocks_mlp.index(best_block)
            old_total_compression_ratio = config.compression_ratio_mlp[idx_]
            current_compression_ratio = config.compression_ratios[idx_removed]
            new_total_compression_ratio = 1 - (1-old_total_compression_ratio) * (1-current_compression_ratio)
            print("Best Block in Transformer Block")
            idx_block = config.transformer_blocks_mlp.index(best_block)
            print(idx_block)
            current_removed_blocks.append(best_block)
            current_compression_ratios.append(new_total_compression_ratio)
        
        else: 
            new_total_compression_ratio = config.compression_ratios[idx_removed]
            current_removed_blocks.append(best_block)
            current_compression_ratios.append(new_total_compression_ratio)
            
        block_list.remove(best_block)
        


       
            
        with open(config.log_path+"/results.txt", "a") as f: 
            f.write("Removed Block List: " + str(config.transformer_blocks_mlp))
            f.write("\n")
            f.write("Compression Ratios: " + str(config.compression_ratio_mlp))
            f.write("\n")
            f.write("New Removed Block List: " + str(current_removed_blocks))
            f.write("\n")
            f.write("New Compression Ratios: " + str(current_compression_ratios))
            f.write("\n")
            f.write("New Block List: " + str(block_list))
            f.write("\n")  
            f.write("\n")

        with open(config.log_path+"/cmmd_dict.json", "w") as f:
            json.dump(score, f)
            f.write("\n") 
        
     
        with open(config.log_path+"/_temp_new_removed_blocks.json", "w") as f:
            json.dump(config.transformer_blocks_mlp, f)
            f.write("\n")   