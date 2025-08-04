import argparse
import json
import os

import numpy as np
import yaml
from box import Box


def update_config( config_path,  updates):
    # Load the current config
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Apply updates
    config.update(updates)

    # Write the updated config back to the file
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Updated config written to: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/export/home/sheid/MasterThesis_Evaluation/configs/PixArt/config_distillation.yaml", help="Path to the YAML config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = Box(yaml.safe_load(f))

    with open(config.log_path + "/_temp_new_removed_blocks.json", "r") as new_blocks_file:
        all_removed_blocks = json.load(new_blocks_file)
        
    
    all_blocks = np.linspace(4, 25, 22, dtype=int).tolist()
    block_list = [block for block in  all_blocks if block not in all_removed_blocks]
    
    old_log_path = config.log_path
    old_log_path_split = old_log_path.split("/")
    old_log_path_split[-2] = str(int(old_log_path_split[-2]) + 1)
    new_log_path = "/".join(old_log_path_split)
    
    old_save_path = config.save_path
    old_save_path_split = old_save_path.split("/")
    old_save_path_split[-2] = str(int(old_save_path_split[-2]) + 1)
    new_save_path = "/".join(old_save_path_split)
        
    block_str = ""
    for idx in all_removed_blocks:
        block_str += "_" + str(idx)
    orig_ckpt_path = config.work_dir + block_str + "_finetuning_trained_on_pixart_generated_images/checkpoints/epoch_"+str(config.epochs_pixart_training)+"_step_"+str(config.training_steps)+".pth" 
    dump_path = config.work_dir + block_str + "_finetuning_trained_on_pixart_generated_images/checkpoints/epoch_"+str(config.epochs_pixart_training)+"_step_"+str(config.training_steps)
    # Example dictionary with updated values
    updates = {
        "block_list": block_list,
        "removed_blocks": all_removed_blocks,
        "save_path": new_save_path,
        "log_path": new_log_path,
        "orig_ckpt_path": orig_ckpt_path,
        "dump_path": dump_path
    }

    update_config( args.config, updates)
