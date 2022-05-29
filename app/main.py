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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model():
    def __init__(self):
        self.pre_model = torch.nn.Sequential(*(list(timm.create_model('vgg16', 
                pretrained=True, num_classes=1000).children())[:-1]))
        self.path = "Log_Reg_on_Resnet50_features"
    def pre_f(img):
        return self.pre_model(img)

    def predict(img):
        Img = pre_f(img)
        model_after = pickle.load(open( "Models/Log_Reg_on_Resnet50_features" , 'rb' ) )
        return model_after.predict(Img)



def gen_model():
    model_ft = models.resnet18(pretrained=True)
    for param in model_ft.parameters():
      param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    return model_ft


def model_prediction(model, img, classes_map: dict) -> str:
    model_pred = model(img)
    ans = classes_map[int((torch.max(model_pred, 1)[1]))]
    return ans


def process_and_predict(data: Union[str, bytes], img_getter: Callable):
    img = prep.process_image(img_getter(data))
    return model_prediction(animal_m, img, ANIMAL_CLASSES)


animal_m = gen_model()
animal_m.load_state_dict(torch.load("./Models/ResNet18TL.pt", map_location=torch.device(device)))
animal_m.eval()


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict/animal_type/from_local_path/")
async def pred_from_path(img: Image):
    return {"asdfsdf": process_and_predict(img.img_path, prep.get_img_from_path)}


@app.post("/predict/animal_type/from_url/")
async def pred_from_url(img: Image):
    a = process_and_predict(img.img_path, prep.get_img_from_url)
    return {"CLASS_NAME": a}
