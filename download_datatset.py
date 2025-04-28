from datasets import load_dataset

from diffusion.model.builder import build_model
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, read_config

config = read_config(
    "configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py"
)
model = build_model(
    config.model,
    config.grad_checkpointing,
    config.get("fp32_attention", False),
    input_size=512 // 8,
    learn_sigma=1,
    pred_sigma=1,
)

print(model)
