from torchvision import transforms
from PIL import Image
import io
import requests
import numpy as np
from numpy.typing import NDArray
from keras.utils import image_utils

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


def process_image(img, shape: tuple, framework='torch') -> NDArray:
    """

    :param img:
    :param shape:
    :param framework: should be "torch", "keras" or "tf"
    :return: if framework - "torch" returns torch.tensor batch, for "keras" returns np.array batch
    """
    print(img)
    if framework == 'torch':
        transform = transforms.Compose(
            [
                transforms.Resize(shape),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        ret = transform(img)
        ret = ret.unsqueeze(0)
        return ret
    elif framework == "tf-torch":
        transform = transforms.Compose(
            [
                transforms.Resize(shape),
                transforms.ToTensor()
            ]
        )
        ret = transform(img)
        ret = 2. * ((ret - 0)) - 1.
        ret = ret.unsqueeze(0)
        return ret
    elif framework == 'keras':
        img = image_utils.img_to_array(img)
        img = image_utils.smart_resize(img, size=(299, 299))
        img = np.expand_dims(img, axis=0)
        return img


if __name__ == "__main__":
    pass
