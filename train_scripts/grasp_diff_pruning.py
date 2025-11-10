import argparse
import datetime
import os
import sys
import time
import json
import warnings
from pathlib import Path

from safetensors.torch import save_file
from tqdm import tqdm

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs

from diffusers.models import AutoencoderKL
from diffusion import  IDDPM
from diffusion.data.builder import build_dataloader, build_dataset, set_data_root
from diffusion.model.builder import build_model

from diffusion.utils.checkpoint import load_checkpoint


from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.misc import (
    init_random_seed,
    read_config,
    set_random_seed,
)
from transformers import T5EncoderModel, T5Tokenizer
from diffusion.model.nets.modeling_grasp_diff_pruning import GRASPBaseModel

def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "PixArtBlock"


def main(model, 
         tokenizer, 
         encoder,
         vae,
         dataloader,
         config,
         train_diffusion, 
         device
         ):
    
    grasp_model = GRASPBaseModel(model=model, vae=vae, tokenizer=tokenizer, encoder=encoder, config=config)

   
    
    layers_id = config.invSVD_blocks
    if isinstance(layers_id, int):
        layers_id = [layers_id]
        
    layers_id.sort(reverse=True)
    grasp_model.to(device=device)
    
    if config.prune_all_layers_together:
        for layer_id in tqdm(layers_id, desc="GRASP Compressing", total=len(layers_id), leave=True):
            grasp_model.compress_block(layer_id)
            grasp_model.to(device=device)
            
            # KORRIGIERTE LOGIK in main()
        for name, param in grasp_model.model.named_parameters():
            if name.endswith(".S"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        grasp_layer_grads = grasp_model.get_svdlayer_gradients(dataloader, device=device, train_diffusion=train_diffusion, save_model_steps=config.save_model_steps)
        indices_dict = grasp_model.dynamic_svd_selection(
                grasp_layer_grads,
                compression_ratio=config.compression_ratio
            )
        grasp_model.compile_grasp_model(indices_dict)

    else:
        for layer_id in tqdm(layers_id, desc="GRASP Compressing", total=len(layers_id), leave=True):
            grasp_model.compress_block(layer_id)
            grasp_model.to(device=device)
            
            # KORRIGIERTE LOGIK in main()
            for name, param in grasp_model.model.named_parameters():
                # Prüft, ob der Parameter zur aktuellen Schicht gehört UND ein .S-Vektor ist
                if f"blocks.{layer_id}." in name and name.endswith(".S"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
            grasp_layer_grads = grasp_model.get_svdlayer_gradients(dataloader, device=device, train_diffusion=train_diffusion, save_model_steps=config.save_model_steps)
            indices_dict = grasp_model.dynamic_svd_selection(
                    grasp_layer_grads,
                    compression_ratio=config.compression_ratio
                )
            grasp_model.compile_grasp_model(indices_dict)
    
    # 4. Den state_dict vom entpackten Modell holen
    compressed_state_dict = grasp_model.model.state_dict()

    # 5. Speicherpfad definieren (Beispiel)
    save_path = os.path.join(config.output_dir, "compressed_model.safetensors")
    os.makedirs(config.output_dir, exist_ok=True)
    # 6. Mit safetensors speichern (bevorzugte Methode)
    # (Eventuell müssen Sie 'pip install safetensors' ausführen)
    save_file(compressed_state_dict, save_path)



def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config", default="/export/home/sheid/GRASP/PixArt-sigma/configs/pixart_sigma_config/partially_block_removal/GRASP/Diff_pruning_all_layers.py", type=str, help="config")
    
    parser.add_argument("--loss_report_name", type=str, default="loss")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)
    device="cuda"
    pred_sigma = getattr(config, "pred_sigma", True)
    learn_sigma = getattr(config, "learn_sigma", True) and pred_sigma
    tokenizer = text_encoder = None
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    log_name = "train_log.log"
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(os.path.join(config.work_dir, "logs"), exist_ok=True)
    image_size = config.image_size  # @param [256, 512]
    latent_size = int(image_size) // 8
   
    if os.path.exists(os.path.join(config.work_dir, log_name)):
        rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    config.seed = init_random_seed(config.get("seed", None))
    set_random_seed(config.seed)


    config.dump(os.path.join(config.work_dir, "config.py"))
        
    vae = None
    tokenizer = None
    text_encoder = None
    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(
            config.vae_pretrained, torch_dtype=torch.float16
        ).to(device)
        config.scale_factor = vae.config.scaling_factor
    tokenizer = text_encoder = None
    if not config.data.load_t5_feat:
        tokenizer = T5Tokenizer.from_pretrained(
            args.pipeline_load_from, subfolder="tokenizer"
        )
        text_encoder = T5EncoderModel.from_pretrained(
            args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(device)

    logger.info(f"vae scale factor: {config.scale_factor}")
    
    model_kwargs = {
        "pe_interpolation": config.pe_interpolation,
        "config": config,
        "model_max_length": max_length,
        "qk_norm": config.qk_norm,
        "kv_compress_config": kv_compress_config,
        "micro_condition": config.micro_condition,
    }

    # build models
    train_diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.snr_loss,
    )
    model = build_model(
        config.model,
        config.grad_checkpointing,
        config.get("fp32_attention", False),
        input_size=latent_size,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        **model_kwargs,
    ).train().to(device)
    
    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from,
            model,
            load_ema=config.get("load_ema", False),
            max_length=max_length,
        )
        
    set_data_root(config.data_root)
    dataset = build_dataset(
        config.data,
        resolution=image_size,
        aspect_ratio_type=config.aspect_ratio_type,
        real_prompt_ratio=config.real_prompt_ratio,
        max_length=max_length,
        config=config,
    )
    
    train_dataloader = build_dataloader(
            dataset,
            num_workers=config.num_workers,
            batch_size=config.train_batch_size,
            shuffle=True,
        )
    

    main(model, 
         tokenizer, 
         text_encoder,
         vae,
         train_dataloader,
         config, 
         train_diffusion,
         device)