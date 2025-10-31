import logging
from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusion.model.nets.pruned_model_parts import (
    GraspAttentionKVCompress,
    GraspdMLP,
    GraspMultiHeadCrossAttention,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    
    
class SVDLinear(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], sigma_fuse: Literal["UV", "U", "V"] = "UV"):
        '''
        **__Args__:**
            U: Left Singular Vectors after rank truncation, which is shape of [rank, out_features]
            S: Diagonal Matrix of singular values, which is shape of [rank, rank]
            Vh: Right Singular Vectors after rank truncation, which is shape of [in_features, rank]
            bias: bias
        '''
        super(SVDLinear, self).__init__()
        
        in_features = Vh.shape[1]
        out_features = U.shape[0]
        hidden_size = S.shape[0]

        self.InLinear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)
        self.OutLinear = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True if bias is not None else False)

        if bias is not None:
            self.OutLinear.bias.data = bias
        
        if sigma_fuse == "UV":
            self.InLinear.weight.data = Vh.mul(S.sqrt().view(-1, 1)).contiguous()
            self.OutLinear.weight.data = U.mul(S.sqrt()).contiguous()
        elif sigma_fuse == "U":
            self.InLinear.weight.data = Vh.contiguous()
            self.OutLinear.weight.data = U.mul(S).contiguous()
        elif sigma_fuse == "V":
            self.InLinear.weight.data = Vh.mul(S.view(-1, 1)).contiguous()
        else:
            raise ValueError(f"value of sigma_fuse {sigma_fuse} not support")
    
    def forward(self, x: torch.Tensor):
        output = self.OutLinear(self.InLinear(x))
        return output



class GRASPLayer(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], compression_ratio: Optional[float]):
        super(GRASPLayer, self).__init__()
        self.U = nn.Parameter(U.clone().detach().requires_grad_(False))
        self.S = nn.Parameter(S.clone().detach().requires_grad_(True))
        self.Vh = nn.Parameter(Vh.clone().detach().requires_grad_(False))

        self.in_features = self.Vh.shape[1]
        self.out_features = self.U.shape[0]

        self.bias = bias
        self.compression_ratio = compression_ratio

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        sigma = torch.diag(self.S)
        W_reconstructed =  torch.mm(self.U, torch.mm(sigma, self.Vh))
        return torch.mm(x.view(b*s, -1), W_reconstructed.t()).view(b, s, -1)
    

class GRASPBaseModel(nn.Module):
    def __init__(self, model: nn.Module, vae: nn.Module, encoder: nn.Module, tokenizer, config: dict,  *args, **kwargs) -> None:
        super(GRASPBaseModel, self).__init__(*args, **kwargs)
        self.model = model
        self.vae = vae
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = config
        for params in self.model.parameters():
            params.requires_grad = False

        self.grasp_values_dict = {}
          
    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_model = model
        for token in tokens[:-1]:
            sub_model = getattr(sub_model, token)
        setattr(sub_model, tokens[-1], module)
        
    def replace_with_GRASPLayer(self, target_layer: int, device: Literal["cuda", "cpu"] = "cuda", log_file: Optional[str] = None):
        print("REPLACEMENT")
        print(target_layer)
        mlp_helper = GraspdMLP(self.model.blocks[target_layer])
        self.model.blocks[target_layer].mlp = mlp_helper
        
        attn_helper = GraspAttentionKVCompress(self.model.blocks[target_layer])
        self.model.blocks[target_layer].attn = attn_helper
        
        cross_attn_helper = GraspMultiHeadCrossAttention(self.model.blocks[target_layer])
        self.model.blocks[target_layer].cross_attn = cross_attn_helper
        
    def compress_block(
            self,
            layer_id: int,
        ):
        self.replace_with_GRASPLayer(target_layer=layer_id)
        
    def compute_preserve_rank(self, grasp_layer: GRASPLayer, compression_ratio: float):
        if compression_ratio is None:
            raise ValueError("Compression ratio should not be None")
        in_features = grasp_layer.in_features
        out_features = grasp_layer.out_features
        k = int(in_features * out_features * (1 - compression_ratio) / (in_features + out_features))
        return k
    
    def check_exists_grasp_layer(self, log_file: Optional[str] = None):
        setup_logger(log_file=log_file)
        grasp_layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, GRASPLayer):
                grasp_layer_names.append(name)
                continue
        if not grasp_layer_names:
            logger.info("GRASPLayer not found in current model, please use GRASPBaseModel.replace_with_GRASPLayer first")
    
    def get_svdlayer_gradients(self, calibration_dataloader: DataLoader, accelerator, device: Literal["cuda:0", "cpu"] = "cuda:0", log_file: Optional[str] = None, *args, **kwargs):
        setup_logger(log_file=log_file)
        load_vae_feat = getattr(calibration_dataloader.dataset, "load_vae_feat", False)
        load_t5_feat = getattr(calibration_dataloader.dataset, "load_t5_feat", False)
        grasp_layer_names = self.check_exists_grasp_layer()
        if grasp_layer_names is None:
            raise NotImplementedError("GRASPLayer not found, can not compute gradients, please use GRASPBaseModel.replace_with_GRASPLayer first")

        iterator = tqdm(calibration_dataloader, desc="Gradients Collection", total=len(calibration_dataloader), leave=True)
        grasp_layer_grads = {}
        self.model.to(device=device)
        for batch_idx, batch in enumerate(iterator):
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=(
                            self.config.mixed_precision == "fp16"
                            or self.config.mixed_precision == "bf16"
                        )
                    ):
                        posterior = self.vae.encode(batch[0]).latent_dist
                        if self.config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()

            clean_images = z * self.config.scale_factor
            data_info = batch[3]

            if load_t5_feat:
                y = batch[1]
                y_mask = batch[2]
            else:
                with torch.no_grad():
                    txt_tokens = self.tokenizer(
                        batch[1],
                        max_length=300,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    y = self.text_encoder(
                        txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask
                    )[0][:, None]
                    y_mask = txt_tokens.attention_mask[:, None, None]

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, self.config.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
               
                loss_term = train_diffusion.training_losses(
                    self.model,
                    clean_images,
                    timesteps,
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info),
                )
                loss = loss_term["loss"].mean()
            
                self.model.zero_grad()
                accelerator.backward(loss)
                
                for grasp_layer_name in grasp_layer_names:
                    module: GRASPLayer = self.model.get_submodule(grasp_layer_name)
                    if not module:
                        raise ValueError("module can not found")
                    if grasp_layer_name not in grasp_layer_grads:
                        grasp_layer_grads[grasp_layer_name] = module.S.grad
                    else:
                        grasp_layer_grads[grasp_layer_name] += module.S.grad

            if "cuda" in device:
                torch.cuda.empty_cache()
            if batch_idx == 10:
                break

        self.grasp_layer_grads = grasp_layer_grads

        return grasp_layer_grads
    
    
    def dynamic_svd_selection(
            self,
            grasp_layer_grads: dict,
            metric: Literal["gradient", "taylor"] = "taylor",
            compression_ratio: Optional[float] = None,
            threshold_ratio: Optional[float] = None,
            verbose: Optional[bool] = False,
            log_file: Optional[str] = None
        ):
        setup_logger(log_file=log_file)
        if not grasp_layer_grads:
            grasp_layer_grads = self.grasp_layer_grads
            raise ValueError("gradients of grasp_layer should be given, but got None")

        indices_dict = {}

        for grasp_layer_name, grasp_layer_grad in grasp_layer_grads.items():
            grasp_layer: GRASPLayer = self.model.get_submodule(grasp_layer_name)
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
                k = self.compute_preserve_rank(grasp_layer, compression_ratio=compression_ratio)
                _, indices = torch.topk(svd_importance, k=k)
            else:
                assert threshold_ratio, "Please provide Taylor threshold to select rank adaptively"
                indices = adaptive_rank_selection(svd_importance_list=svd_importance, target_ratio=threshold_ratio)
            indices_dict[grasp_layer_name] = indices
            self.grasp_values_dict[grasp_layer_name] = {}
            self.grasp_values_dict[grasp_layer_name]["svd_importance"] = torch.round(svd_importance.cpu(), decimals=3).tolist()
            self.grasp_values_dict[grasp_layer_name]["svd_value"] = torch.round(S.data.cpu(), decimals=3).tolist()

        if verbose:
            logger.info("+" * 100)
            for grasp_layer_name, indices in indices_dict.items():
                logger.info(f"{grasp_layer_name}")
                logger.info(indices.detach().cpu().numpy().tolist()[:128])
            logger.info("+" * 100)

        self.indices_dict = indices_dict
        return indices_dict
    
    
    def compile_grasp_model(
        self,
        indices_dict: Optional[dict] = None,
        merge: Optional[bool] = False,
        sigma_fuse: Literal["UV", "U", "V"] = "UV",
        device: Literal["cpu", "cuda"] = "cuda",
        log_file: Optional[str] = None
    ):
        setup_logger(log_file=log_file)
        if indices_dict is None:
            indices_dict = self.indices_dict

        rank_dict = {}

        for grasp_layer_name, indices in indices_dict.items():
            grasp_layer: GRASPLayer = self.model.get_submodule(grasp_layer_name)

            S = grasp_layer.S[indices]
            U = grasp_layer.U[:, indices]
            Vh = grasp_layer.Vh[indices, :]
            bias = grasp_layer.bias

            rank_dict[grasp_layer_name] = S.shape[0]

            if merge:
                in_features = Vh.shape[1]
                out_features = U.shape[0]
                self._set_module(self.model, grasp_layer_name, nn.Linear(in_features=in_features, out_features=out_features, bias=True if bias is not None else False))
                linear_layer: nn.Linear = self.model.get_submodule(grasp_layer_name)

                # re-initialize linear weight and bias
                W_compressed = torch.mm(U, torch.mm(torch.diag(S), Vh))
                linear_layer.weight.data = W_compressed

                if bias is not None:
                    linear_layer.bias = bias
                
                linear_layer.requires_grad_(False)
            else:
                self._set_module(self.model, grasp_layer_name, SVDLinear(U=U, S=S, Vh=Vh, bias=bias, sigma_fuse=sigma_fuse))
                svd_linear_layer: SVDLinear = self.model.get_submodule(grasp_layer_name)
                svd_linear_layer.requires_grad_(False)
            
            del grasp_layer
            if "cuda" in device:
                torch.cuda.empty_cache()
        return