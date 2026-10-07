# InstructPix2Pix
Unofficial repository for instruct pix2pix paper.


Instruct Pix2Pix is a [image-to-image diffusion model](https://arxiv.org/abs/2211.09800)
that can change images based on written instructions. It uses a method called Stable Diffusion, 
which is trained using a special [dataset](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered). This model is the foundation of various projects.

For a better understanding of what Instruct Pix2Pix can do, you can read a blog post from
[Hugging Face](https://huggingface.co/blog/instruction-tuning-sd).

This repository is built on top of the Diffusers library to customize the InstructPix2Pix model by fine-tune it.
This repo supports training on multiple GPUs with the help of accelerator library. The model can be fine-tuned on
a custom dataset and used for inference. The model can be used
for various tasks including image-to-image translation and image editing. 

For now, this repository supports Stable Diffusion 1.5 and 2.0 and 2.1 as the starting point of InstructPixPix model.

## Installation
To use this project, install the corresponding requirement.txt file in your environment. Or you can follow 
the install.sh file to install the dependencies in your conda environment.

### Install using requirement file
Follow these steps to install the required packages using the requirements.txt file:
1. Create a new python env.
2. Activate the environment you have just created.
3. Install the requirements using the following command:

```commandline
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

### Install using SH file
Create a fresh Python 3.10-3.12 environment, for example `conda create -n myenv python=3.12`, and activate it. Run all requirement commands from the repository root so the local wheel path resolves. Then use the following commands to
install the required packages inside the conda environment.

First, make the install.sh file executable by running the following command:
```commandline
chmod +x install.sh
```

Then, run the following command to install the required packages inside the conda environment:
```commandline
bash install.sh
```


The security dependency set uses PyTorch 2.13.0, torchvision 0.28.0, Diffusers 0.41.0,
Transformers 5.19.0 and xFormers 0.0.35. xFormers now uses the PyTorch stable ABI for
2.10 and later. CUDA 11.8 builds are no longer supported by this dependency set;
select an [official PyTorch 2.13 CUDA channel](https://pytorch.org/get-started/previous-versions/)
and a compatible NVIDIA driver. `install.sh` accepts `PYTORCH_CHANNEL=cpu`, `cu126`
(default), or `cu130` and installs the same pinned requirements. GPU installation
also selects xFormers from the same channel: the PyPI wheel targets CUDA 12.8
and should not be mixed with the CUDA 12.6 or 13.0 builds. It no
longer mixes independently selected conda and pip framework versions.

Accelerate includes a narrow local checkpoint-loading security patch because
upstream 1.15.0 still has an unfixed traversal issue. Its [source patch, hashes and
rebuild instructions](vendor/README.md) are committed beside the required wheel.
Do not replace it with an unpatched upstream installation. Unused Datasets,
torchaudio, torchmetrics, cosine-warmup and Datasets helper requirements were
removed; Parquet loading continues through pandas/PyArrow. The model structure,
UNet adaptation, default training settings and optimizer choices are unchanged.

Offline checks can run with a CPU PyTorch installation:

```sh
PYTORCH_CHANNEL=cpu bash install.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

These tests import project modules, load tiny local CLIP/VAE/UNet components,
verify checkpoint and Parquet/image round trips, and reject unsafe shard indexes.
They perform no training or diffusion sampling and download no pretrained models
or datasets. The CPU environment does not load the xFormers CUDA extension;
the local component test disables CUDA attention. POSIX named-pipe tests skip on Windows. Full GPU training,
multi-GPU resume, xFormers CUDA kernels and pretrained model output equivalence
have not been evaluated; use a fresh environment and retain original checkpoints
when migrating from the old framework stack.


### Prepare Dataset
Download a test dataset ([link](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples)) and
add its directory in the config file.


You can create a custom dataset by yourself.

### Train

After data preparation and carefully setting the config.yaml file, run the following command in the terminal inside
the train folder.

```bash
accelerate launch train.py --config_path './config.yaml'
 ```

#### Note:
Once the training is finished the model will be saved to a folder called "save_results". You could also monitor the performance of the model in the images_log folder.

After a training is finished, the following directories will be available in the `save_results/timestamp` directory:
1) **diffusers_checkpoint**: This directory can be used for getting inference with diffusers library.
2) **accelerator_checkpoints**: This directory can be used for resume training. The numbers in the directories' names show the global step. 
In order to use it for inference you have to convert it using
3) **lora_checkpoints**: To do (this directory would be used for getting inference with low rank adaptation).

We can simply use our fine-tuned model using the following code:

```python
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch

model_path = "path_to_saved_model" # including modules' folder
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt='make the sky blue', image=img, image_guidance_scale=1.5).images[0]

image.save("result.png")
```

### Custom Training

To show the InstructPix2Pix model's capabilities, we did a cool and unique training on approximately 
1 million images from the Open Image V5 dataset. The objective was to fine-tune the model to perform zooming on images
based on textual instructions specifying the zoom percentage. This involved collecting images, generating target images
with random center zooms, and creating corresponding text prompts. The whole process is based on self-supervised 
learning. The training utilized the Adam optimizer with a learning rate schedule from 5e-5 to 1e-6, employing 
cosine annealing and gradient accumulation over nearly 4 epochs (10,000 steps). The fine-tuned model effectively
zooms into images per the given instructions, showing promising capabilities up to 200% zoom, with minor
artifacts at higher levels. Below are two GIFs demonstrating the model's performance in zooming tasks.

**Example Prompt**: “Zoom 150 percent into the center of the image.”

![Zoom Example 1](files/example_1.gif)
![Zoom Example 2](files/example_2.gif)

This project is a great example of how InstructPix2Pix can be fine-tuned for various tasks, and we are excited to see
what other creative applications the community will come up with!
