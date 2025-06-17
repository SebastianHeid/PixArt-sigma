from diffusion.data.datasets.InternalData_ms import InternalDataMSSigma

img_path = "/export/data/vislearn/rother_subgroup/dzavadsk/datasets/laion/subset_250/images/"
path = "/export/data/vislearn/rother_subgroup/sheid/pixart/laion"
json = "data_info.json"

dataset = InternalDataMSSigma(root=path,img_root=img_path,aspect_ratio_type = "ASPECT_RATIO_512")
print(dataset.__len__())