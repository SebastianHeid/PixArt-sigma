from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusion.model.nets.pruned_model_parts import GRASPLayer, SVDLinear


def dynamic_svd_selection(
            model,
            grasp_layer_grads: dict,
            metric: Literal["gradient", "taylor"] = "taylor",
            compression_ratio: Optional[float] = None,
            threshold_ratio: Optional[float] = None,
            verbose: Optional[bool] = False,
            log_file: Optional[str] = None
        ):
        
        if not grasp_layer_grads:
            raise ValueError("gradients of grasp_layer should be given, but got None")

        indices_dict = {}

        for grasp_layer_name, grasp_layer_grad in grasp_layer_grads.items():
            grasp_layer: GRASPLayer = model.get_submodule(grasp_layer_name)
            S = grasp_layer.S

            if metric == "gradient":
                svd_importance: torch.Tensor = torch.abs(grasp_layer_grad)
            elif metric == "taylor":
                svd_importance: torch.Tensor = torch.abs(grasp_layer_grad * S)
            else:
                raise RuntimeError(f"{metric} not support")

            if grasp_layer.compression_ratio is not None:
                compression_ratio = grasp_layer.compression_ratio

            if compression_ratio is not None:            
                k = compute_preserve_rank(grasp_layer, compression_ratio=compression_ratio)
                _, indices = torch.topk(svd_importance, k=k)
            else:
                assert threshold_ratio, "Please provide Taylor threshold to select rank adaptively"
                indices = adaptive_rank_selection(svd_importance_list=svd_importance, target_ratio=threshold_ratio)
            indices_dict[grasp_layer_name] = indices
            grasp_values_dict = {}
            grasp_values_dict[grasp_layer_name] = {}
            grasp_values_dict[grasp_layer_name]["svd_importance"] = torch.round(svd_importance.cpu(), decimals=3).tolist()
            grasp_values_dict[grasp_layer_name]["svd_value"] = torch.round(S.data.cpu(), decimals=3).tolist()

        return indices_dict, grasp_values_dict    
    

def adaptive_rank_selection(svd_importance_list, target_ratio):
    total_sum = sum(svd_importance_list)
    target_sum = total_sum * target_ratio

    sorted_list = sorted(enumerate(svd_importance_list), key=lambda x: -x[1])

    cumulative_sum = 0
    indices = []
    for index, value in sorted_list:
        cumulative_sum += value
        indices.append(index)
        if cumulative_sum >= target_sum:
            break
    return indices


def compute_preserve_rank(grasp_layer: GRASPLayer, compression_ratio: float):
        if compression_ratio is None:
            raise ValueError("Compression ratio should not be None")
        in_features = grasp_layer.in_features
        out_features = grasp_layer.out_features
        k = int(in_features * out_features * (1 - compression_ratio) / (in_features + out_features))
        return k
    
    
    
def compile_grasp_model(
        model,
        indices_dict: Optional[dict] = None,
        merge: Optional[bool] = False,
        sigma_fuse: Literal["UV", "U", "V"] = "UV",
        device: Literal["cpu", "cuda"] = "cuda",
        log_file: Optional[str] = None
    ):
       

        rank_dict = {}

        for grasp_layer_name, indices in indices_dict.items():
            grasp_layer: GRASPLayer = model.get_submodule(grasp_layer_name)

            S = grasp_layer.S[indices]
            U = grasp_layer.U[:, indices]
            Vh = grasp_layer.Vh[indices, :]
            bias = grasp_layer.bias

            rank_dict[grasp_layer_name] = S.shape[0]

            if merge:
                in_features = Vh.shape[1]
                out_features = U.shape[0]
                _set_module(model, grasp_layer_name, nn.Linear(in_features=in_features, out_features=out_features, bias=True if bias is not None else False))
                linear_layer: nn.Linear = model.get_submodule(grasp_layer_name)

                # re-initialize linear weight and bias
                W_compressed = torch.mm(U, torch.mm(torch.diag(S), Vh))
                linear_layer.weight.data = W_compressed

                if bias is not None:
                    linear_layer.bias = bias
                
                linear_layer.requires_grad_(False)
            else:
                _set_module(model, grasp_layer_name, SVDLinear(U=U, S=S, Vh=Vh, bias=bias, sigma_fuse=sigma_fuse))
                svd_linear_layer: SVDLinear = model.get_submodule(grasp_layer_name)
                svd_linear_layer.requires_grad_(False)
            
            del grasp_layer
            if "cuda" in device:
                torch.cuda.empty_cache()
        return model
    
    
def _set_module( model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_model = model
    for token in tokens[:-1]:
        sub_model = getattr(sub_model, token)
    setattr(sub_model, tokens[-1], module)