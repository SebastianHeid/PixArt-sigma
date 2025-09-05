""" Create data_info.json for laion2M dataset"""

import json
import os

from PIL import Image
from tqdm import tqdm

path_json_input = "/export/data/vislearn/rother_subgroup/dzavadsk/datasets/mapillary/generated_captions/mapillary_train_joycaptions.json"
path_images = "/export/data/vislearn/rother_subgroup/dzavadsk/datasets/mapillary/training/images/"
path_new_json = "/export/data/vislearn/rother_subgroup/sheid/pixart/mapillary/data_info_train.json"


with open(path_json_input, "r") as input_file:
    captions = json.load(input_file)
    
    output_data = []
    list_img_names = sorted([f for f in os.listdir(path_images) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    for img_n in list_img_names:
        img = Image.open(path_images+img_n)
        width, height = img.size 
        ratio = round(width/height, 3)
        key=img_n[:-4]
        new_entry = {"path": path_images+img_n,
                    "prompt": captions[key],
                    "width": width,
                    "height": height,
                    "ratio": ratio,
                    "sharegpt4v": ""}
        output_data.append(new_entry)
        
   

with open(path_new_json, "w") as file:
    json.dump(output_data, file, indent=2)
