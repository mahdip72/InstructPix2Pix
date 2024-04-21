import os

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionInstructPix2PixPipeline
from diffusers import ControlNetModel
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer


def adapt_unet_for_pix2pix(unet):
    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    return unet


def prepare_model(config, logger):
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")

    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer", revision=config.revision
    )
    logger.info("load tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision
    )
    logger.info("load text encoder")

    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision)
    logger.info("load vae")

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", revision=config.non_ema_revision
    )
    logger.info("load unet")

    if config.adapt_unet:
        unet = adapt_unet_for_pix2pix(unet)
        logger.info("prepare unet for instruct pix2pix training")

    if config.freeze_text_encoder:
        # Freeze text_encoder
        text_encoder.requires_grad_(False)
        logger.info("freeze text encoder weights")
    else:
        text_encoder.train()

    # Freeze vae
    vae.requires_grad_(False)
    logger.info("freeze vae weights")

    # Create EMA for the unet.
    if config.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        logger.info("build ema unet model")

    else:
        ema_unet = False

    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training,"
                    " please update xFormers to at least 0.0.17. "
                    "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        logger.info("enable xformer memory efficient attention")

    return vae, text_encoder, unet, noise_scheduler, tokenizer, ema_unet


def save_diffuser_checkpoint(accelerator, config, unet, ema_unet, text_encoder, vae, logging_dir, logger,
                             controlnet=False):
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if config.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            safety_checker=None,
            requires_safety_checker=False,
            revision=config.revision,
        )
        # pipeline.requires_safety_checker = False

        save_path = os.path.join(logging_dir, 'diffusers_checkpoint')
        pipeline.save_pretrained(save_path)
        logger.info("save diffusion format checkpoint")


def resume_from_checkpoint(accelerator, num_update_steps_per_epoch, config):
    accelerator.print(f"Resuming from checkpoint {config.resume_from_checkpoint.accelerator_path}")
    accelerator.load_state(os.path.abspath(config.resume_from_checkpoint.accelerator_path))
    global_step = int(config.resume_from_checkpoint.accelerator_path.split("-")[1])

    resume_global_step = global_step * config.train_settings.gradient_accumulation_steps
    first_epoch = global_step // num_update_steps_per_epoch
    resume_step = resume_global_step % (num_update_steps_per_epoch * config.train_settings.gradient_accumulation_steps)
    return first_epoch, resume_step


if __name__ == '__main__':
    print('test model modules')
