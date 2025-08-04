import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import yaml
from box import Box
from diffusers import PixArtSigmaPipeline, Transformer2DModel
from tqdm import tqdm


def prepare_json_file(json_laion2M_path,json_pixart_generated_dataset_path):
    with open(json_laion2M_path, "r") as file:
        laion2M_data = json.load(file)
    with open(json_pixart_generated_dataset_path, "w") as file2:
        for idx, item in tqdm(enumerate(laion2M_data)):
            if idx ==1000000:
                break
            item["width"] = 512
            item["height"] = 512
            item["ratio"] = 1.0
            json.dump(item, file2)
            file2.write(",\n")
            
def create_images(json_pixart_generated_dataset_path, output_dir, start_idx, end_idx=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    weight_dtype = torch.float16
    transformer = Transformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", 
        subfolder='transformer', 
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    
    pipeline = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
            transformer=transformer,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
    pipeline.to("cuda")
    
    
    with open(json_pixart_generated_dataset_path, "r") as file:
        laion_prompts = json.load(file)
        if end_idx is None:
            end_idx = len(laion_prompts)
        for idx in tqdm(range(start_idx, end_idx)):
            item = laion_prompts[idx]
            prompt = item["prompt"]
            image = pipeline(prompt).images[0]
            image.save(os.path.join(output_dir+item["path"]))
            
            
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--start_idx",
        type=int,
        const=True,
        nargs="?",
        default=0,
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        const=True,
        nargs="?",
        default=1000000,
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    json_laion2M_path = "/export/data/vislearn/rother_subgroup/sheid/pixart/laion2M/data_info.json"
    json_pixart_generated_dataset_path = "/export/data/vislearn/rother_subgroup/sheid/pixart/1M_pixart_generated_images/json/data_info.json"
    output_dir = "/export/data/vislearn/rother_subgroup/sheid/pixart/1M_pixart_generated_images/images/"
    #prepare_json_file(json_laion2M_path,json_pixart_generated_dataset_path)
    create_images(json_pixart_generated_dataset_path, output_dir, args.start_idx, args.end_idx)
    