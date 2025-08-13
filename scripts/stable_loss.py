import torch as th 
import random 
import numpy as np
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
@contextmanager
def temprngstate(new_seed: Optional[int] = None):
    """
    Context manager that saves and restores the RNG state of PyTorch, NumPy and Python.
    If new_seed is not None, the RNG state is set to this value before the context is entered.
    """

    # Save RNG state
    old_torch_rng_state = th.get_rng_state()
    old_torch_cuda_rng_state = th.cuda.get_rng_state()
    old_numpy_rng_state = np.random.get_state()
    old_python_rng_state = random.getstate()

    # Set new seed
    if new_seed is not None:
        th.manual_seed(new_seed)
        th.cuda.manual_seed(new_seed)
        np.random.seed(new_seed)
        random.seed(new_seed)

    yield

    # Restore RNG state
    th.set_rng_state(old_torch_rng_state)
    th.cuda.set_rng_state(old_torch_cuda_rng_state)
    np.random.set_state(old_numpy_rng_state)
    random.setstate(old_python_rng_state)
    
    
    
def stable_loss():
    for step, batch in enumerate(tqdm(train_dataloader)):
            if step < skip_step:
                global_step += 1
                continue  # skip data in the resumed ckpt
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=(
                            config.mixed_precision == "fp16"
                            or config.mixed_precision == "bf16"
                        )
                    ):
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()

            clean_images = z * config.scale_factor
            data_info = batch[3]

            if load_t5_feat:
                y = batch[1]
                y_mask = batch[2]
            else:
                with torch.no_grad():
                    txt_tokens = tokenizer(
                        batch[1],
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    y = text_encoder(
                        txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask
                    )[0][:, None]
                    y_mask = txt_tokens.attention_mask[:, None, None]

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                # Predict the noise residual
                with th.no_grad():
                    if config.intermediate_loss_flag:
                        loss_term = train_diffusion.training_losses(
                            model,
                            clean_images,
                            timesteps,
                            ref_model=ref_model,
                            intermediate_loss_blocks=config.intermediate_loss_blocks,
                            final_output_loss_flag=config.final_output_loss_flag,
                            org_loss_flag = config.org_loss_flag,
                            model_kwargs=dict(y=y, mask=y_mask, data_info=data_info),
                        )
                    else:
                        loss_term = train_diffusion.training_losses(
                            model,
                            clean_images,
                            timesteps,
                            model_kwargs=dict(y=y, mask=y_mask, data_info=data_info),
                        )
                loss = loss_term["loss"].mean()