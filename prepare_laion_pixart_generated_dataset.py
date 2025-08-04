import json 
from tqdm import tqdm 
import os 
from diffusers import PixArtSigmaPipeline, Transformer2DModel
import torch

def prepare_json_file(json_laion2M_path,json_pixart_generated_dataset_path):
    with open(json_laion2M_path, "r") as file:
        laion2M_data = json.load(file)
    with open(json_pixart_generated_dataset_path, "w") as file2:
        for idx, item in tqdm(enumerate(laion2M_data)):
            if idx ==100000:
                break
            item["width"] = 512
            item["height"] = 512
            item["ratio"] = 1.0
            json.dump(item, file2)
            file2.write(",\n")
            
def create_images(json_pixart_generated_dataset_path, output_dir):
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
        for idx, item in tqdm(enumerate(laion_prompts)):
            prompt = item["prompt"]
            image = pipeline(prompt).images[0]
            image.save(os.path.join(output_dir+item["path"]))
            if idx <= 50000:
                continue
            
            

if __name__ == "__main__":
    json_laion2M_path = "/export/data/vislearn/rother_subgroup/sheid/pixart/laion2M/data_info.json"
    json_pixart_generated_dataset_path = "/export/data/vislearn/rother_subgroup/sheid/pixart/pixart_generated_images/json/data_info.json"
    output_dir = "/export/data/vislearn/rother_subgroup/sheid/pixart/pixart_generated_images/images/"
    #prepare_json_file(json_laion2M_path,json_pixart_generated_dataset_path)
    create_images(json_pixart_generated_dataset_path, output_dir)
    