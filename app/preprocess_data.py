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


def get_img_from_url(urls: Union[list[str],str]):
    if type(urls) == list:
        image_datas = [requests.get(url).content for url in urls]
    else:
        image_datas = [requests.get(urls).content]
    return [get_img_from_bytes(image_data) for image_data in image_datas]


def get_img_from_path(pathes: Union[list[str],str]):
    if type(pathes) == list:
        images = [Image.open(path) for path in pathes]
    else:
        images = [Image.open(pathes)]
    return [image.convert("RGB") for img in images]


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
