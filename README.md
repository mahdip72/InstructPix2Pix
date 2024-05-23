# InstructPix2Pix
Unofficial repository for instruct pix2pix paper.


Instruct Pix2Pix is a [image-to-image diffusion model](https://arxiv.org/abs/2211.09800)
that can change images based on written instructions. It uses a method called Stable Diffusion, 
which is trained using a special [dataset](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered). This model is the foundation of various projects.

For a better understanding of what Instruct Pix2Pix can do, you can read a blog post from
[Hugging Face](https://huggingface.co/blog/instruction-tuning-sd).

## Installation
To use this project, install the corresponding requirement.txt file in your environment. Or you can follow 
the install.sh file to install the dependencies in your conda environment.

### Install using requirement file
Follow these steps to install the required packages using the requirements.txt file:
1. Create a new python env.
2. Activate the environment you have just created.
3. Install the requirements using the following command:

```commandline
pip install -r requirements.txt
```

### Install using SH file
Create a conda environment using: `conda create -n myenv python=3.10`. Then, do as the following commands to
install the required packages inside the conda environment.

First, make the install.sh file executable by running the following command:
```commandline
chmod +x install.sh
```

Then, run the following command to install the required packages inside the conda environment:
```commandline
bash install.sh
```


__Note__: It's crucial to pay attention to the dependencies to make sure all necessary components are installed.
The versions of accelerator, diffusers, transformers, PyTorch and xformers you use can affect what features
are available during training and inference. Sometimes you need to change the version of libraries to make it work 
flawlessly For example, PyTorch 2.0 switches from standard attention to 
flash attention. Also, newer versions of accelerator have a different setup for accelerator_checkpoint compared 
to older versions. In addition, xformers library needs a specific version of PyTorch to work properly. We added a verified
requirements file that works well on ubuntu os and therefore, you can start with that. Sticking to the guidelines in the
requirements.txt file is the simplest approach. 


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
