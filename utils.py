import torch
import logging as log
import os
import yaml
from box import Box
from diffusers.optimization import get_scheduler
import torch.nn.functional as F


def get_logging(result_path, log_level=log.INFO):
    logger = log.getLogger(result_path)
    logger.setLevel(log_level)

    fh = log.FileHandler(os.path.join(result_path, "logs.txt"))
    formatter = log.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = log.StreamHandler()
    logger.addHandler(sh)

    return logger


def prepare_optimizer(config, accelerator, parameters):
    if config.optimizer.scale_lr:
        config.optimizer.learning_rate = (
                config.optimizer.learning_rate * config.train_settings.gradient_accumulation_steps * config.train_settings.batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if config.optimizer.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        parameters,
        lr=float(config.optimizer.learning_rate),
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        weight_decay=float(config.optimizer.adam_weight_decay),
        eps=float(config.optimizer.adam_epsilon),
    )
    lr_scheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps,
        num_training_steps=config.train_settings.max_train_steps,
    )
    return optimizer, lr_scheduler


def snr_loss(timesteps, noise_scheduler, model_pred, config, target):
    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
    # This is discussed in Section 4.2 of the same paper.
    snr = compute_snr(timesteps, noise_scheduler)
    mse_loss_weights = (
            torch.stack([snr, config.train_settings.min_snr_loss.snr_gamma * torch.ones_like(timesteps)],
                        dim=1).min(dim=1)[0] / snr
    )

    # We first calculate the original loss. Then we mean over the non-batch dimensions and
    # rebalance the sample-wise losses with their respective loss weights.
    # Finally, we take the mean of the rebalanced loss.
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    loss = loss.mean()
    return loss


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


if __name__ == '__main__':
    yaml_config_path = 'config.yaml'
    with open(yaml_config_path) as file:
        dict_config = yaml.safe_load(file)

    config = Box(dict_config)
    print('for test')


