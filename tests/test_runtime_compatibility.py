"""Small CPU-only compatibility checks; no training, sampling, or downloads."""
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image
import torch
from accelerate import Accelerator
from box import Box
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from instruct_pix2pix.data import load_parquet_dataset
from instruct_pix2pix.model import adapt_unet_for_pix2pix, prepare_model

torch.set_num_threads(1)


class RuntimeCompatibilityTests(unittest.TestCase):
    def tiny_unet(self):
        return UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            layers_per_block=1, block_out_channels=(8,), norm_num_groups=4,
            down_block_types=("CrossAttnDownBlock2D",), up_block_types=("CrossAttnUpBlock2D",),
            cross_attention_dim=8, attention_head_dim=2)

    def test_unet_adaptation_preserves_pretrained_weights(self):
        unet = self.tiny_unet()
        before = unet.conv_in.weight.detach().clone()
        adapt_unet_for_pix2pix(unet)
        self.assertEqual(unet.config.in_channels, 8)
        torch.testing.assert_close(unet.conv_in.weight[:, :4], before)
        self.assertEqual(torch.count_nonzero(unet.conv_in.weight[:, 4:]).item(), 0)
        ema = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        self.assertEqual(len(ema.shadow_params), len(list(unet.parameters())))

    def test_accelerator_checkpoint_round_trip(self):
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        model = accelerator.prepare(torch.nn.Linear(2, 2))
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as scratch:
            accelerator.save_state(scratch, safe_serialization=True)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            accelerator.load_state(scratch)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key])

    def test_legacy_clip_files_and_local_components_load(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            tokenizer_path = root / "tokenizer"
            tokenizer_path.mkdir()
            vocabulary = {"<|startoftext|>": 0, "<|endoftext|>": 1, "z</w>": 2,
                          "o</w>": 3, "m</w>": 4, "z": 5, "o": 6, "m": 7}
            (tokenizer_path / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
            (tokenizer_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            ids = tokenizer("zoom", add_special_tokens=True)["input_ids"]
            self.assertEqual(ids, [0, 5, 6, 6, 4, 1])
            tokenizer.save_pretrained(tokenizer_path)
            self.assertEqual(CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)("zoom")["input_ids"], ids)
            text_encoder = CLIPTextModel(CLIPTextConfig(vocab_size=len(vocabulary), hidden_size=8,
                intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                max_position_embeddings=77, bos_token_id=0, eos_token_id=1, pad_token_id=1))
            vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                block_out_channels=(8,), layers_per_block=1, norm_num_groups=4, sample_size=8)
            scheduler = DDPMScheduler(num_train_timesteps=10)
            pipe = StableDiffusionInstructPix2PixPipeline(vae=vae, text_encoder=text_encoder,
                tokenizer=tokenizer, unet=self.tiny_unet(), scheduler=scheduler,
                safety_checker=None, feature_extractor=None, requires_safety_checker=False)
            pipe.save_pretrained(root)
            config = Box(dict(pretrained_model_name_or_path=str(root), revision=None,
                non_ema_revision=None, adapt_unet=True, freeze_text_encoder=True,
                use_ema=True, enable_xformers_memory_efficient_attention=False))
            parts = prepare_model(config, logging.getLogger("offline-test"))
            self.assertEqual(parts[2].config.in_channels, 8)
            self.assertFalse(any(parameter.requires_grad for parameter in parts[0].parameters()))
            self.assertFalse(any(parameter.requires_grad for parameter in parts[1].parameters()))

    def test_pillow_parquet_loader_round_trip(self):
        image = Image.new("RGB", (4, 4), "blue")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "pairs.parquet"
            pd.DataFrame([dict(input_image={"bytes": buffer.getvalue()}, edited_image={"bytes": buffer.getvalue()},
                               edit_prompt="zoom")]).to_parquet(path)
            rows = load_parquet_dataset(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0].size, (4, 4))
        self.assertEqual(rows[0][2], "zoom")


if __name__ == "__main__":
    unittest.main()
