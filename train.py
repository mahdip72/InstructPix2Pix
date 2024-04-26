import argparse
import datetime
import logging
import math
import os
import shutil

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from box import Box
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from instruct_pix2pix.data import SDDataset, ZoomDataset, tokenize_captions
from instruct_pix2pix.model import prepare_model, save_diffuser_checkpoint
from instruct_pix2pix.utils import prepare_optimizer, snr_loss, get_logging


def main(yaml_config_path):
    with open(yaml_config_path) as file:
        dict_config = yaml.safe_load(file)

    config = Box(dict_config)

    # making saving directories
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    logging_dir = os.path.abspath(os.path.join(config.logging_dir, timestamp))

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    shutil.copyfile(yaml_config_path, os.path.join(logging_dir, 'config.yaml'))

    accelerator_project_config = ProjectConfiguration(total_limit=config.checkpoints_total_limit,
                                                      logging_dir=os.path.abspath(os.path.join(logging_dir, 'logs')))
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train_settings.gradient_accumulation_steps,
        mixed_precision=config.train_settings.mixed_precision,
        log_with="tensorboard" if config.tensorboard else None,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

    # create logger
    logger = get_logging(logging_dir, log_level=logging.INFO)
    logger.info(f"prepare a logger to log in {logging_dir}")

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(int(config.seed))
        logger.info(f"set random seed to {config.seed}")

    vae, text_encoder, unet, noise_scheduler, tokenizer, ema_unet = prepare_model(
        config, logger
    )

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("enable gradient checkpointing for low memory training")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.train_settings.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("enable Tensor Float 32 training")

    train_dataset = ZoomDataset(tokenizer=tokenizer, config=config)

    logger.info("prepare training dataset")
    
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=config.train_settings.shuffle,
        batch_size=config.train_settings.batch_size,
        num_workers=config.train_settings.num_workers,
        pin_memory=True
    )
    logger.info("prepare dataloader from dataset")

    # Prepare everything with our `accelerator`.
    unet, train_dataloader = accelerator.prepare(unet, train_dataloader)
    logger.info("prepare unet and train dataloader for accelerator")

    if config.freeze_text_encoder:
        text_encoder = accelerator.prepare(text_encoder)
        logger.info("freeze text encoder weights")

    parameters = unet.parameters()

    optimizer, lr_scheduler = prepare_optimizer(config, accelerator, parameters)
    logger.info("create a new optimizer and lr scheduler")

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    logger.info("prepare optimizer and lr scheduler for accelerator")

    if config.use_ema:
        ema_unet.to(accelerator.device)
        logger.info("use ema to train unet module")

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # compile model to train faster and efficiently
    if config.model_compile:
        vae = torch.compile(vae)
        text_encoder = torch.compile(text_encoder)
        unet = torch.compile(unet)
        if accelerator.is_main_process:
            logging.info('compile model is done')

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.train_settings.gradient_accumulation_steps)

    # Afterward we recalculate our number of training epochs
    num_train_epochs = math.ceil(config.train_settings.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    
    if accelerator.is_main_process:
        accelerator.init_trackers("accelerator_tracker", config=None)

    total_batch_size = config.train_settings.batch_size * accelerator.num_processes * config.train_settings.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_settings.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train_settings.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.train_settings.max_train_steps}")

    epoch = 0
    global_step = 0
    first_epoch = 0
    resume_step = 0
    if config.resume_from_checkpoint.enable:
        accelerator.print(f"Resuming from checkpoint {config.resume_from_checkpoint.accelerator_path}")
        accelerator.load_state(config.resume_from_checkpoint.accelerator_path)
        global_step = int(config.resume_from_checkpoint.accelerator_path.split("-")[-1])

        resume_global_step = global_step * config.train_settings.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * config.train_settings.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config.train_settings.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        unet.train()
        target_model = unet

        for step, batch in enumerate(train_dataloader):

            # Skip steps until we reach the resumed step
            if config.resume_from_checkpoint.enable and epoch == first_epoch and step < resume_step:
                if step % config.train_settings.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(target_model):
                # We want to learn the de-noising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning.
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                
                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if config.train_settings.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * config.train_settings.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""], tokenizer).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                            (random_p >= config.train_settings.conditioning_dropout_prob).to(image_mask_dtype)
                            * (random_p < 3 * config.train_settings.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample

                # use min-snr loss for faster convergence
                if config.train_settings.min_snr_loss.enable:
                    loss = snr_loss(timesteps, noise_scheduler, model_pred, config, target)
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.train_settings.batch_size)).mean()
                train_loss += avg_loss.item() / config.train_settings.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    parameters = unet.parameters()

                    accelerator.clip_grad_norm_(parameters, config.optimizer.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, 'lr': lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(logging_dir, 'accelerator_checkpoints', f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.train_settings.max_train_steps:
                break

    save_diffuser_checkpoint(accelerator, config, unet, ema_unet, text_encoder, vae, logging_dir, logger)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an Instruct Pix2Pix model")

    parser.add_argument(
        "--config_path", "-c", help="The location of config file", default='./configs/config.yaml')

    args = parser.parse_args()
    config_path = args.config_path

    main(config_path)
