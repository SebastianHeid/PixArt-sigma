import io
import json
import os
import tarfile

import numpy as np
from PIL import Image
from tqdm import tqdm

directory_path = "/export/data/ntatsch/imageNet/train/"
class_labels_path = "/export/data/vislearn/rother_subgroup/sheid/ImageNet/imagenet_class_index.json"
json_path = "/export/data/vislearn/rother_subgroup/sheid/ImageNet/data_info.json"

with open(class_labels_path, 'r') as f:
    class_labels = json.load(f)
    wnid_to_labels = {v[0]: v[1] for v in class_labels.values()}
    
    folder_names = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

    for folder_name in tqdm(folder_names):
        image_names = [name for name in os.listdir(os.path.join(directory_path, folder_name)) if name.endswith('.JPEG')]
        for image_name in image_names:
            image_path = os.path.join(directory_path, folder_name, image_name)
            with open(image_path, 'rb') as f:
                image = Image.open(f)
                width, height = image.size
                ratio = np.round(width / height,3)
                
                image_dict = {"height": height, "width": width, "ratio": ratio, "path": image_path, "prompt": wnid_to_labels[folder_name], "sharegpt4v": ""}
                with open(json_path, 'a') as json_file:
                    json.dump(image_dict, json_file)
                    json_file.write(',\n')