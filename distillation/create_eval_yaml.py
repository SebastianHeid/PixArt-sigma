import argparse
import json
import os
import shutil

import numpy as np
import yaml
from box import Box


def copy_and_edit_script(src_path, dst_path, edits):
    # Copy the file
    shutil.copy(src_path, dst_path)

    # Read and modify lines
    with open(dst_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        for var, new_value in edits.items():
            if line.strip().startswith(f"{var} ="):
                line = f"{var} = {repr(new_value)}\n"
        new_lines.append(line)

    # Write back
    with open(dst_path, 'w') as f:
        f.writelines(new_lines)

def compute_trainable_blocks(all_blocks, old_blocks):
    trainable_blocks = [int(block) for block in all_blocks if block not in old_blocks]
    trainable_blocks = [block - 1 for block in trainable_blocks if block != 0]

    for idx, block in enumerate(trainable_blocks):
        while (block >=0 and block in all_blocks):
            trainable_blocks[idx] = block - 1
            block -= 1 
    trainable_blocks = list(set(trainable_blocks))
    return trainable_blocks
 
def compute_intermediate_loss_blocks(all_blocks):
    intermediate_blocks = np.linspace(4,25, 22, dtype=int).tolist()  
    min_all_blocks = min(all_blocks) 
    intermediate_loss_blocks = [block for block in intermediate_blocks if block+1 not in all_blocks and block >= min_all_blocks]
    return intermediate_loss_blocks
    
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--config_path",
        type=str,
        const=True,
        default="/export/home/sheid/MasterThesis_Evaluation/configs/PixArt/config_distillation.yaml",
        nargs="?",
        help="Path to config.yaml file",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        config = Box(yaml.safe_load(file))
    
    with open(config.log_path + "/_temp_new_removed_blocks.json", "r") as new_blocks_file:
        all_blocks = json.load(new_blocks_file)
    new_str = ""
    for idx in all_blocks:
        new_str += "_"+str(idx)

    
    with open(config.log_path + "/_temp_old_removed_blocks.json", "r") as old_blocks_file:
        old_blocks = json.load(old_blocks_file)
    old_str = ""
    for idx in old_blocks:
        old_str += "_"+str(idx)
    
    
    # determine which blocks to train
    trainable_blocks = compute_trainable_blocks(all_blocks, old_blocks)
    intermediate_loss_blocks = compute_intermediate_loss_blocks(all_blocks)

    copy_and_edit_script(
        src_path=config.config_file+old_str+".py",
        dst_path=config.config_file+new_str+".py",
        edits={
        "intermediate_loss_blocks" : intermediate_loss_blocks,
        "transformer_blocks" : all_blocks,
        "trainable_blocks" : trainable_blocks,
        "load_from": config.checkpoint_path + old_str + "_finetuning_trained_on_pixart_generated_images/checkpoints/epoch_"+str(config.epochs_pixart_training)+"_step_"+str(config.training_steps)+".pth"
        }
    )
    
    copy_and_edit_script(
        src_path=config.config_file+old_str+"_finetuning.py",
        dst_path=config.config_file+new_str+"_finetuning.py",
        edits={
        "intermediate_loss_blocks" : intermediate_loss_blocks,
        "transformer_blocks" : all_blocks,
        "trainable_blocks" : [],
        "load_from": config.checkpoint_path + new_str+ "/checkpoints/epoch_"+str(config.epochs_pixart_finetuning)+"_step_"+str(config.finetuning_steps)+".pth"
        }
    )
    
    copy_and_edit_script(
        src_path=config.config_file+old_str+"_finetuning_trained_on_pixart_generated_images.py",
        dst_path=config.config_file+new_str+"_finetuning_trained_on_pixart_generated_images.py",
        edits={
        "intermediate_loss_blocks" : intermediate_loss_blocks,
        "transformer_blocks" : all_blocks,
        "trainable_blocks" : [],
        "load_from":  config.checkpoint_path + new_str+ "_finetuning/checkpoints/epoch_"+str(config.epochs_pixart_finetuning)+"_step_"+str(config.finetuning_steps)+".pth"
        }
    )