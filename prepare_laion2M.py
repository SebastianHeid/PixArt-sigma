""" Create data_info.json for laion2M dataset"""

import os 
import json 
from PIL import Image
from tqdm import tqdm 

path_json_input = "/export/data/vislearn/rother_subgroup/rother_datasets/LaionAE/joyPrompts_laion2B_en_aesthetic_train_split_captions.json"
path_images = "/export/data/vislearn/rother_subgroup/rother_datasets/LaionAE/laion2B-en-art_512/"
path_new_json = "/export/data/vislearn/rother_subgroup/sheid/pixart/laion2M/data_info.json"

path_new_json_fixed = "/export/data/vislearn/rother_subgroup/sheid/pixart/laion2M/data_info_fixed.json"

with open(path_json_input, "r") as input_file:
    captions = json.load(input_file)
    
    output_data = []
    for key in tqdm(captions.keys()):
        img = Image.open(path_images+key+".webp")
        width, height = img.size 
        ratio = round(width/height, 3)
        new_entry = {"path": key+".webp",
                    "prompt": captions[key],
                    "width": width,
                    "height": height,
                    "ration": ratio,
                    "sharegpt4v": ""}
        output_data.append(new_entry)
        break
   
    

with open(path_new_json, "w") as file:
    json.dump(output_data, file, indent=2)

with open(path_new_json, "r") as file:
    captions = json.load(file)
    for entry in tqdm(captions):
        entry["ratio"] = entry.pop("ration")

with open(path_new_json_fixed, "w") as file2:
    json.dump(captions, file2, indent=2)