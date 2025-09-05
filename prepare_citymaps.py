""" Create data_info.json for laion2M dataset"""

import json
import os

from PIL import Image
from tqdm import tqdm

path_json_input = "/export/data/vislearn/rother_subgroup/dzavadsk/datasets/cityscapes/generated_captions/cityscapes_trainANDtrainextra_joy_prompts_midjourney_long.json"
path_images = "/export/data/vislearn/rother_subgroup/rother_datasets/cityscapes_denis/cityscapes_full/leftImg8bit_trainextra/"
path_new_json = "/export/data/vislearn/rother_subgroup/sheid/pixart/cityscapes/data_info_train.json"


with open(path_json_input, "r") as input_file:
    captions = json.load(input_file)
    
    output_data = []
   
    for key in captions.keys():
        try:
            img = Image.open(path_images+key+".png")
            width, height = img.size 
            ratio = round(width/height, 3)
            new_entry = {"path": path_images+key+".png",
                        "prompt": captions[key],
                        "width": width,
                        "height": height,
                        "ratio": ratio,
                        "sharegpt4v": ""}
            output_data.append(new_entry)
        except: 
            continue
   

with open(path_new_json, "w") as file:
    json.dump(output_data, file, indent=2)
