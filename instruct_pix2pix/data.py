import io
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as function
import yaml
from PIL import Image
from box import Box
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import rotate
from transformers import CLIPTokenizer


def load_parquet_dataset(data_path):
    """
    Load a dataset of image pairs and their edit prompts from a Parquet file.

    This function reads a Parquet file specified by the 'data_path'. It expects
    the Parquet file to contain a DataFrame with at least three columns: 'input_image',
    'edited_image', and 'edit_prompt'. The 'input_image' and 'edited_image' columns
    should contain binary image data, while 'edit_prompt' contains the associated text prompts.

    The function iterates over each row in the DataFrame, converts the binary image data
    into PIL Image objects for both 'input_image' and 'edited_image', and stores these
    images along with their corresponding 'edit_prompt' in a list.

    Parameters:
    data_path (str): The file path to the Parquet file containing the dataset.

    Returns:
    list: A list of tuples, each containing a pair of PIL Image objects (input and edited images)
          and the associated edit prompt string.
    """
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
    """
    A callable class that pads an image to make it square.

    This class is designed to transform an input image into a square-shaped image
    by adding padding. It calculates the necessary padding to be added to the
    sides of the image (both horizontally and vertically) to ensure the final
    image is square. The padding is applied equally to all sides of the image.

    The padding is filled with a constant value (defaulting to zero, representing
    black in most image formats). The class uses a specified padding function,
    typically from a library like PIL or NumPy, to apply the padding.

    Methods:
    __call__(self, image): Takes a PIL Image or similar object, computes the required padding
                           to make the image square, and returns the padded image.

    Example usage:
    square_padder = SquarePad()
    square_image = square_padder(original_image)

    Note:
    The class assumes that the 'pad' function is available under the 'function' namespace
    and follows the signature function.pad(image, padding, fill, padding_mode).
    """
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return function.pad(image, padding, 0, 'constant')


class SDDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing data for the Instruct Pix2Pix model,
    which is based on the Stable Diffusion model.

    This dataset loader is designed to handle images and corresponding captions from a Parquet file.
    It includes data augmentation methods like random horizontal flipping and rotation,
    and preprocesses images by resizing and normalizing them.

    Attributes:
    - datadir (str): Directory path where the Parquet dataset is stored.
    - fixed_prompt (str or bool): A fixed prompt if specified; otherwise False.
    - prompt_suffix (str): A suffix to append to prompts.
    - tokenizer (Tokenizer): A tokenizer for processing captions.
    - samples (list): Loaded dataset samples.
    - swap_input_output (bool): Whether to swap input and output images.
    - rand_horizontal_flip (bool): Whether to randomly flip images horizontally.
    - rotation (bool): Whether to randomly rotate images.
    - base_size (int): Base resolution for initial image cropping.
    - final_size (int): Final resolution of the images.
    - train_transforms (Compose): Composed PyTorch transforms for training data preprocessing.

    Methods:
    - __init__(self, tokenizer, config): Initializes the SDDataset instance.
                                      - tokenizer (Tokenizer): The tokenizer used for processing captions.
                                                               This should be an instance of a tokenizer compatible
                                                               with the captions in the dataset, typically provided
                                                               by libraries like Hugging Face's Transformers.
                                      - config (object): A configuration object containing various settings for
                                                         data loading and preprocessing. This should include
                                                         attributes for training settings, augmentations, and
                                                         other necessary configurations specific to the Instruct
                                                         Pix2Pix model.
    - tensor_to_pil(image_tensor): Converts a PyTorch tensor to a PIL Image.
    - pil_to_cv2(pil_image): Converts a PIL Image to a NumPy array (OpenCV format).
    - cv2_to_pil(cv2_image): Converts a NumPy array (OpenCV format) to a PIL Image.
    - pad_to_square(image_tensor): Pads a tensor image to make it square.
    - __len__(): Returns the length of the dataset.
    - __getitem__(i): Retrieves the i-th item from the dataset.
    - tokenize_captions(captions, tokenizer): Tokenizes captions using the provided tokenizer.

    Example usage:
    dataset = SDDataset(tokenizer, config)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    Note:
    This class is specifically tailored for the Instruct Pix2Pix model and assumes the presence of specific
    fields in the configuration and dataset structure.
    """
    def __init__(self, tokenizer, config):
        self.datadir = config.train_settings.data_dir

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
        """
        Converts a PyTorch tensor to a PIL Image.

        This method takes an image in the form of a PyTorch tensor, converts it to a NumPy array,
        rescales its values to the range [0, 255], and then converts it to a PIL Image. This is
        useful for visualizing PyTorch tensors or for further processing using PIL library functions.

        Parameters:
        image_tensor (Tensor): A PyTorch tensor representing an image. The tensor is expected to have
                               the shape (C, H, W), where C is the number of channels, H is the height,
                               and W is the width of the image.

        Returns:
        Image: A PIL Image object corresponding to the input tensor.
        """
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

        input_ids = tokenize_captions(captions, self.tokenizer)
        result = dict(original_pixel_values=input_image, edited_pixel_values=target_image, input_ids=input_ids,
                      caption=captions)

        return result


class ZoomDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing data for the Instruct Pix2Pix model,
    which is based on the Stable Diffusion model.
    """

    def __init__(self, tokenizer, config):
        self.datadir = config.train_settings.data_dir

        self.tokenizer = tokenizer
        self.samples = self.load_zoom_dataset(self.datadir)

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
    def load_zoom_dataset(data_path):
        samples = []
        # Find all images in the directory based on the desired format and add their path into the samples list
        for root, _, files in os.walk(data_path):
            for img_path in files:
                if img_path.endswith(".png", ".jpg", ".jpeg"):
                    samples.append(os.path.join(root, img_path))
        return samples

    @staticmethod
    def tensor_to_pil(image_tensor):
        """
        Converts a PyTorch tensor to a PIL Image.

        This method takes an image in the form of a PyTorch tensor, converts it to a NumPy array,
        rescales its values to the range [0, 255], and then converts it to a PIL Image. This is
        useful for visualizing PyTorch tensors or for further processing using PIL library functions.

        Parameters:
        image_tensor (Tensor): A PyTorch tensor representing an image. The tensor is expected to have
                               the shape (C, H, W), where C is the number of channels, H is the height,
                               and W is the width of the image.

        Returns:
        Image: A PIL Image object corresponding to the input tensor.
        """
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
    def random_crop_square(image, min_side):
        width, height = image.size

        # Choose a random start point for the crop that guarantees at least 1024x1024 crop
        max_left = width - min_side
        max_top = height - min_side
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)

        # Crop the image
        crop_rectangle = (left, top, left + min_side, top + min_side)
        cropped_image = image.crop(crop_rectangle)

        return cropped_image

    @staticmethod
    def center_zoom(image, min_zoom_size):
        # Make sure the minimum zoom size is less than the current image size
        if min_zoom_size >= min(image.size):
            raise ValueError("Minimum zoom size must be smaller than the image size.")

        # Calculate the scale range and select a random value
        zoom_size = random.randint(min_zoom_size, min(image.size))

        # Calculate the box coordinates for the new cropped (zoomed) area
        left = (image.width - zoom_size) // 2
        top = (image.height - zoom_size) // 2
        right = left + zoom_size
        bottom = top + zoom_size

        # Perform the crop
        zoomed_image = image.crop((left, top, right, bottom))

        # Calculate the zoom scale percentage relative to the original image size
        zoom_scale = (zoom_size / min(image.size)) * 100

        return zoomed_image, zoom_scale

    def __getitem__(self, i):
        input_image_path = self.samples[i]
        input_image = Image.open(input_image_path)
        input_image = self.random_crop_square(input_image, self.base_size)
        target_image, zoom_scale = self.center_zoom(input_image, self.final_size)

        captions = f"Zoom {zoom_scale} percent into the center of the image."

        input_image = input_image.resize((self.final_size, self.final_size))
        target_image = target_image.resize((self.final_size, self.final_size))

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

        input_ids = tokenize_captions(captions, self.tokenizer)
        result = dict(original_pixel_values=input_image, edited_pixel_values=target_image, input_ids=input_ids,
                      caption=captions)

        return result


def tokenize_captions(captions, tokenizer):
    """
    Tokenizes captions using a specified tokenizer.

    This function processes a given caption (or a list of captions) using the provided tokenizer.
    It sets the maximum length to the tokenizer's model maximum length, applies padding and
    truncation to ensure uniform length, and converts the tokens to PyTorch tensors. The method
    is particularly useful for preparing textual data for input to machine learning models,
    especially those in NLP.

    Parameters:
    captions (str or list of str): The caption(s) to be tokenized. Can be a single string or a list of strings.
    tokenizer (Tokenizer): The tokenizer to be used. This should be an instance of a pre-trained tokenizer
                           compatible with the model being used, typically provided by NLP libraries like
                           Hugging Face's Transformers.

    Returns:
    Tensor: A PyTorch tensor of token IDs corresponding to the input captions. If multiple captions are
            provided, the tensor will have a leading dimension equal to the number of captions.
    """
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def cv2_show(input_img, target_img):
    input_img = (input_img + 1) * 0.5
    target_img = (target_img + 1) * 0.5
    cv2.imshow('input_img', input_img[..., ::-1])
    cv2.imshow('target_img', target_img[..., ::-1])
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
