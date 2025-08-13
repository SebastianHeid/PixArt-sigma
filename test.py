import os
from tqdm import tqdm 
import json 

json_org_path = "/export/data/vislearn/rother_subgroup/sheid/pixart/laion2M/helper.json"
json_new_path = "/export/data/vislearn/rother_subgroup/sheid/pixart/laion2M/helper.json"
img_path = "/export/data/vislearn/rother_subgroup/sheid/pixart/laion2M/feature_pixart/caption_features_new/"
vae_path = "/export/data/vislearn/rother_subgroup/sheid/pixart/laion2M/feature_pixart/img_sdxl_vae_features_512resolution_new/"
no_img_to_keep = 608000
file_names = os.listdir(img_path)
file_names = [f[:-4] for f in file_names]
file_names = [f+".webp" for f in file_names]


vae_names = os.listdir(vae_path)
vae_names = [f[:-4] for f in vae_names]
vae_names = [f+".webp" for f in vae_names]

set_vae = set(vae_names)
only_files = [item for item in file_names if item not in set_vae]


#print(file_names)
new_dic = []
# with open(json_new_path, "r") as org_file:
#     data = json.load(org_file)
#     file_set = set(file_names)
#     keys_not_in_list = [k for k in data if k not in file_set]
        

        
print(only_files)