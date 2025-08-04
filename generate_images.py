import os

import torch
from diffusers import AutoencoderKL, PixArtSigmaPipeline

from diffusion.model.nets.PixArtMS import PixArtMS
from PIL import Image
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer
from diffusion.utils.checkpoint import load_checkpoint
# Configs
prompt = "A futuristic city in the clouds, digital painting"
output_path = "output.png"
image_size = 512  # or 512 for smaller models

guidance_scale = 4.0
num_inference_steps = 50
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pipeline to get VAE + tokenizer + T5

weight_dtype = torch.float16
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to(device)

# Extract pretrained VAE and tokenizer/text encoder
vae: AutoencoderKL = pipe.vae
vae.eval()

text_encoder: T5EncoderModel = pipe.text_encoder
tokenizer: T5Tokenizer = pipe.tokenizer
text_encoder.eval()

# Load only the PixArtSigma model separately

ckpt_path = "/export/data/sheid/pixart/PixArt_sigma_xl2_img512_laion2M_skipConnection/checkpoints/epoch_1_step_127226.pth"
model = PixArtMS(skip_connections=True, model_max_length=300)
missing, unexpected  = load_checkpoint(checkpoint=ckpt_path, model=model, max_length=300)
model.to(device)

model.eval()

# Prompt encoding
with torch.no_grad():

    txt_tokens = tokenizer(
        [prompt],
        max_length=300,

        padding="max_length",
        return_tensors="pt",
        truncation=True
    ).to(device)


    prompt_embeds = text_encoder(
                        txt_tokens.input_ids
                    )[0][:, None]

# Latent noise
batch_size = 1
latent_shape = (4, image_size // 8, image_size // 8)
latents = torch.randn(
    (batch_size, *latent_shape),

    device=device,
    dtype=torch.float16
)

# Scheduler (DDIM)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

scheduler = DDIMScheduler.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="scheduler")
scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = scheduler.timesteps

# Denoising loop
with torch.no_grad():
    for i, t in enumerate(timesteps):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Time embedding
        timestep = torch.tensor([t], dtype=torch.float16, device=device)

        # Duplicate text encoder hidden states for guidance

        prompt_embeds_input = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds])


        noise_pred = model(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds_input

        ).sample

        # Apply classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Update latents
        latents = scheduler.step(noise_pred, t, latents).prev_sample

# Decode latents to image
with torch.no_grad():
    latents = 1 / 0.13025 * latents  # scale factor from config
    image = vae.decode(latents).sample
    image = (image.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    save_image(image, output_path)
    print(f"Image saved to {output_path}")
