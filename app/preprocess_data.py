import io

import numpy as np
import requests
import torch
from PIL import Image
from keras.utils import image_utils
from torchvision import transforms

a = 0


def get_img_from_bytes(byte_array: bytes):
    image = Image.open(io.BytesIO(byte_array))
    image = image.convert("RGB")
    return image


def get_img_from_url(url: str):
    image_data = requests.get(url).content
    return get_img_from_bytes(image_data)


def get_img_from_path(path: str):
    image = Image.open(path)
    image = image.convert("RGB")
    return image


def process_image(images, shape: tuple, standardization=False) -> torch.Tensor:
    """

    :param images:
    :param shape:
    :param standardization:
    :return: batch with on tensored images in
    """
    transform = transforms.Compose(
        [
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    Ret = torch.empty(tuple([len(images)]+[3]+list(shape)))
    
    for i,img in enumerate(images):
        Ret[i] = transform(img)
        if standardization:
            Ret[i] = 2. * (Ret[i] - 0) - 1.
    
    return Ret


if __name__ == "__main__":
    pass
