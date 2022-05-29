from fastapi import FastAPI
import torch
import torch.nn as nn
from torchvision import models
import preprocess_data as prep
from typing import Callable, Union
from pydantic import BaseModel


ANIMAL_CLASSES = {
    0: "Cat",
    1: "Dog"
}


class Image(BaseModel):
    img_path: str


device = "cpu"

animal_m = models.AnimalTypeModel()
dog_breed = models.DogsBreedModel()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/{item:path}")
async def test(item: str):
    return {"message": item}


@app.get("/predict/animal_type/from_local_path/")
async def pred_from_path(img: Image):
    return {"asdf": "asdf"}
    # return {"asdfsdf": process_and_predict(img.img_path, prep.get_img_from_path)}


@app.post("/predict/animal_type/from_url/")
async def pred_type_from_url(img: Image):
    image = prep.process_image(prep.get_img_from_url(img.img_path), (224, 224))
    a = animal_m.predict(img=image)
    return {"class_name": a}


@app.post("/predict/dog_breed/from_url/")
async def pred_dog_from_url(img: Image):
    image = prep.process_image(prep.get_img_from_url(img.img_path), (299, 299), framework="tf-torch")
    print(image)
    b = dog_breed.predict(img=image)
    return {"breed": b}
