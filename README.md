# InstructPix2Pix
Unofficial repository for instruct pix2pix paper.


Instruct Pix2Pix is a [image-to-image diffusion model](https://arxiv.org/abs/2211.09800)
that can change images based on written instructions. It uses a method called Stable Diffusion, 
which is trained using a special [dataset](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered). This model is the foundation of various projects.

For a better understanding of what Instruct Pix2Pix can do, you can read a blog post from
[Hugging Face](https://huggingface.co/blog/instruction-tuning-sd).

## Install

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install one of the requirements.txt files 
as the following. Please refer to the Requirements section for more detail.

```bash
pip install -r requirements.txt
```


## Requirements
+ Python 3.9
+ PyTorch: 2.0.1


It's crucial to pay attention to the requirements.txt file to make sure all necessary components are installed.
The versions of accelerator, diffusers, transformers (Huggingface), and PyTorch you use can affect what features
are available during training and inference. For example, PyTorch 2.0 switches from standard attention to 
flash attention. Also, newer versions of accelerator have a different setup for accelerator_checkpoint compared 
to older versions. Sticking to the guidelines in the requirements.txt file is the simplest approach.


## Use
Clone the repository to your local machine
Install the required dependencies (listed in the 'Install' section above)


__Note__: You could set your custom checkpoints or huggingface checkpoint, e.g., `timbrooks/instruct-pix2pix`.


### Prepare Dataset
Will be completed.

### Train

After data preparation and carefully setting the config.yaml file, run the following command in the terminal inside
the train folder.

```bash
python ./train.py --config_path './config.yaml'
 ```

#### Note:
Once the training is finished the model will be saved to a folder called "save_results". You could also monitor the performance of the model in the images_log folder.

After a training is finished, the following directories will be available in the `save_results/timestamp` directory:
1) **diffusers_checkpoint**: This directory can be used for getting inference with diffusers library.
2) **accelerator_checkpoints**: This directory can be used for resume training. The numbers in the directories' names show the global step. 
In order to use it for inference you have to convert it using  
3) **Images_log**: A folder including images on the evaluation images in different evaluation steps (or epochs).
4) **lora_checkpoints**: To do (this directory would be used for getting inference with low rank adaptation).

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

