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


def process_image(img, shape: tuple, standardization=False) -> torch.Tensor:
    """

    :param img:
    :param shape:
    :param standardization:
    :return: batch with on tensored image in
    """
    transform = transforms.Compose(
        [
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    ret = transform(img)
    if standardization:
        ret = 2. * (ret - 0) - 1.
    ret = ret.unsqueeze(0)
    return ret


if __name__ == "__main__":
    pass
