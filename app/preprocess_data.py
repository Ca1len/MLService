import torch
from torch.types import _TensorOrTensors
from torchvision import transforms
from PIL import Image
import io
import requests
import numpy as np
from numpy.typing import NDArray


def get_img_from_bytes(byte_array: bytes):
    image = Image.open(io.BytesIO(byte_array))
    return image


def get_img_from_url(url: str):
    image_data = requests.get(url).content
    return get_img_from_bytes(image_data)


def get_img_from_path(path: str):
    image = Image.open(path)
    return image


def process_image(img) -> _TensorOrTensors:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    ret = transform(img)
    ret = ret.unsqueeze(0)
    print(img, ret.size())
    return ret


if __name__ == "__main__":
    pass
