import random
import torchvision.transforms.functional as function
import io
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import yaml
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import rotate
from transformers import CLIPTokenizer
from PIL import ImageEnhance
from box import Box


def load_parquet_dataset(data_path):

    samples = []
    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(data_path, engine='pyarrow')

    # Assuming the image data is in columns 'input_image' and 'edited_image'
    # We'll use the first row in the DataFrame as an example
    for i in range(len(df)):
        input_image_data = df.loc[i, 'input_image']['bytes']
        edited_image_data = df.loc[i, 'edited_image']['bytes']
        edit_prompt = df.loc[i, 'edit_prompt']

        # Convert the binary data to an image object using Pillow
        input_image = Image.open(io.BytesIO(input_image_data))
        edited_image = Image.open(io.BytesIO(edited_image_data))

        # Now you can work with these images as PIL Image objects
        # input_image.show()
        # edited_image.show()

        samples.append([input_image, edited_image, edit_prompt])
    return samples


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return function.pad(image, padding, 0, 'constant')


class SDDataset(Dataset):
    def __init__(self, tokenizer, config):
        self.datadir = config.train_settings.data_dir

        if type(config.train_settings.fixed_prompt) == str:
            self.fixed_prompt = config.train_settings.fixed_prompt
        else:
            self.fixed_prompt = False

        self.prompt_suffix = config.train_settings.prompt_suffix
        self.tokenizer = tokenizer
        self.samples = load_parquet_dataset(self.datadir)

        self.swap_input_output = config.train_settings.swap_input_output

        self.rand_horizontal_flip = config.augmentations.rand_horizontal_flip
        self.rotation = config.augmentations.rotation

        self.base_size = config.augmentations.crop_resolution
        self.final_size = config.train_settings.final_resolution

        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @staticmethod
    def tensor_to_pil(image_tensor):
        # Convert the PyTorch tensor to a NumPy array
        image_np = image_tensor.squeeze().numpy()

        # Rescale the values of the NumPy array to the range [0, 255]
        image_np = (image_np * 255).astype('uint8')

        # Create a PIL image from the NumPy array
        pil_image = Image.fromarray(image_np.transpose((1, 2, 0)))

        return pil_image

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def pil_to_cv2(pil_image):
        # Convert PIL Image to NumPy array
        np_array = np.array(pil_image)
        return np_array

    @staticmethod
    def cv2_to_pil(cv2_image):
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(cv2_image.astype(np.uint8))

        return pil_image

    @staticmethod
    def pad_to_square(image_tensor):
        _, channels, height, width = image_tensor.size()
        max_size = max(height, width)

        # Calculate the amount of padding needed for the image
        pad_height = max_size - height
        pad_width = max_size - width

        # Pad the image using torch.nn.functional.pad
        padded_image = torch.nn.functional.pad(image_tensor, (0, pad_width, 0, pad_height))

        return padded_image

    def __getitem__(self, i):
        input_image, target_image, captions = self.samples[i]

        if (input_image.size[0] > self.base_size) or (input_image.size[1] > self.base_size):
            if input_image.size[0] >= input_image.size[1]:
                new_h = self.base_size
                new_w = int(input_image.size[0] * self.base_size / input_image.size[1])
            else:
                new_h = int(input_image.size[1] * self.base_size / input_image.size[0])
                new_w = self.base_size
            input_image = input_image.resize((new_w, new_h))
            target_image = target_image.resize((new_w, new_h))
        else:
            input_image = input_image.resize((self.base_size, self.base_size))
            target_image = target_image.resize((self.base_size, self.base_size))

        input_image = self.train_transforms(input_image)
        target_image = self.train_transforms(target_image)

        images = torch.stack([input_image, target_image])

        if self.rotation:
            r = random.random()
            if r < 0.5:
                degree = random.randint(-8, 8)
                images = rotate(images, degree, fill=[-1], interpolation=transforms.InterpolationMode.BILINEAR)

        # unpack the cropped images from the batch
        input_image, target_image = images[0], images[1]

        if self.rand_horizontal_flip:
            r = random.random()
            if r < 0.5:
                input_image = hflip(input_image)
                target_image = hflip(target_image)

        input_ids = self.tokenize_captions(captions, self.tokenizer)
        result = dict(original_pixel_values=input_image, edited_pixel_values=target_image, input_ids=input_ids,
                      caption=captions)

        return result

    @staticmethod
    def tokenize_captions(captions, tokenizer):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids.squeeze()


def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def matplotlib_show(img, mask, masked_img):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.imshow(img)
    ax1.title.set_text('images')
    ax2.imshow(mask)
    ax2.title.set_text('masks')
    ax3.imshow(masked_img)
    ax3.title.set_text('masked_images')
    fig.tight_layout(pad=2.0)
    plt.show()


def cv2_show(input_img, target_img):
    input_img = (input_img + 1) * 0.5
    target_img = (target_img + 1) * 0.5
    cv2.imshow(f'input_img', input_img[..., ::-1])
    cv2.imshow(f'target_img', target_img[..., ::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
    yaml_config_path = 'config.yaml'
    with open(yaml_config_path) as file:
        dict_config = yaml.safe_load(file)

    config = Box(dict_config)

    tokenizer = CLIPTokenizer.from_pretrained(
        "timbrooks/instruct-pix2pix", subfolder="tokenizer"
    )
    train_dataset = SDDataset(tokenizer, config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=0,
                                                   batch_size=1)

    for i in range(10):
        for step, batch in enumerate(train_dataloader):
            input_img, target_img = batch['original_pixel_values'], batch['edited_pixel_values']
            prompt = batch['caption'][0]
            input_img = torchvision.utils.make_grid(torch.squeeze(input_img)).permute(1, 2, 0).numpy()
            target_img = torchvision.utils.make_grid(torch.squeeze(target_img)).permute(1, 2, 0).numpy()
            print(prompt)
            cv2_show(input_img, target_img)
