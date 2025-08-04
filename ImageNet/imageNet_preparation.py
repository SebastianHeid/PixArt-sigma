import io
import json
import os
import tarfile

import numpy as np
from PIL import Image
from tqdm import tqdm

# directory_path = "/export/data/ntatsch/imageNet/train/"
# class_labels_path = "/export/data/vislearn/rother_subgroup/sheid/ImageNet/imagenet_class_index.json"
# json_path = "/export/data/vislearn/rother_subgroup/sheid/ImageNet/data_info.json"

# with open(class_labels_path, 'r') as f:
#     class_labels = json.load(f)
#     wnid_to_labels = {v[0]: v[1] for v in class_labels.values()}
    
#     folder_names = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

#     for folder_name in tqdm(folder_names):
#         image_names = [name for name in os.listdir(os.path.join(directory_path, folder_name)) if name.endswith('.JPEG')]
#         for image_name in image_names:
#             image_path = os.path.join(directory_path, folder_name, image_name)
#             with open(image_path, 'rb') as f:
#                 image = Image.open(f)
#                 width, height = image.size
#                 ratio = np.round(width / height,3)
                
#                 image_dict = {"height": height, "width": width, "ratio": ratio, "path": image_path, "prompt": wnid_to_labels[folder_name], "sharegpt4v": ""}
#                 with open(json_path, 'a') as json_file:
#                     json.dump(image_dict, json_file)
#                     json_file.write(',\n')


correct_path = "/gpfs/lsdf02/sd23g007/datasets/imageNet/train/"
json_path = "/home/hd/hd_hd/hd_om233/PixArt-sigma/ImageNet/data_info.json"
new_json_path = "/home/hd/hd_hd/hd_om233/PixArt-sigma/ImageNet/data_info_feat.json"
new_dict = []
with open(json_path, 'r') as file:
    data_info = json.load(file)
    for item in data_info:
        path = "/".join(item["path"].split("/")[-2:])
        item["path"] = path 
        new_dict.append(item)

with open(new_json_path, "w") as file2:
    json.dump(new_dict, file2, indent=4)
        
    
#     for idx in tqdm(range(len(data_info))):
#         path = data_info[idx]["path"]
#         ending = "/".join(path.split("/")[-2:])
#         data_info[idx]["path"] = correct_path + ending

# with open(json_path, 'w') as file:
#     json.dump(data_info, file, indent=4)