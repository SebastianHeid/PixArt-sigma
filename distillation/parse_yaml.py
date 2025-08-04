import sys

import yaml

# Load path to YAML file from command-line argument
with open(sys.argv[1], "r") as f:
    data = yaml.safe_load(f)

block_str = "" 
for idx in data["removed_blocks"]:
        block_str += "_"+str(idx)
    

# Print shell export statements
print(f"CONFIG_FILE_TRAINING='{data['config_file'] +block_str+ '.py'}'")
print(f"WORK_DIR_TRAINING='{data['work_dir']+block_str}'")

print(f"CONFIG_FILE_FINETUNING='{data['config_file'] + block_str+'_finetuning.py'}'")
print(f"WORK_DIR_FINETUNING='{data['work_dir'] + block_str+'_finetuning'}'")

print(f"CONFIG_FILE_FINETUNING_PIXART='{data['config_file'] +block_str+ '_finetuning_trained_on_pixart_generated_images.py'}'")
print(f"WORK_DIR_FINETUNING_PIXART='{data['work_dir'] + block_str+'_finetuning_trained_on_pixart_generated_images'}'")
