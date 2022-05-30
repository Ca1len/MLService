from typing import Tuple, Any

import torch.nn as nn
from torchvision import models
import torch
import numpy as np
import pickle
import timm
import json

device = "cpu"

with open("./animal_types.json") as jf:
    animal_types = json.load(jf)

with open("./dog_breeds.json") as jf:
    dog_breeds = json.load(jf)


with open("./cat_breeds.json") as jf:
    cat_breeds = json.load(jf)


def change_device(name: str):
    device = name
    return device


def model_prep(model):
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    return model


class DogsBreedModel(object):
    def __init__(self):
        super().__init__()
        POOLING = "avg"
        shape = (3, 299, 299)
        # self.xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
        # self.inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=POOLING)
        self.xception_bottleneck = model_prep(torch.nn.Sequential(
            *(list(timm.create_model('xception', pretrained=True, num_classes=1000).children())[:-1])))
        self.inception_bottleneck = model_prep(torch.nn.Sequential(
            *(list(timm.create_model('inception_v3', pretrained=True, num_classes=1000).children())[:-1])))
        self.logreg = pickle.load(open("Models/IXnceptionLogReg_on_dog_features.pkl", 'rb'))

    def predict(self, img: torch.Tensor) -> Tuple[str, float]:
        with torch.no_grad():
            x_bf = self.xception_bottleneck(img)
            i_bf = self.inception_bottleneck(img)
            # x_bf = self.xception_bottleneck.predict(img, batch_size=1, verbose=1)
            # i_bf = self.inception_bottleneck.predict(img, batch_size=1, verbose=1)
        X = np.hstack([x_bf, i_bf])
        return dog_breeds[str(int(self.logreg.predict(X)))], np.sort(self.logreg.predict_proba(X)[0])[-1]


class CatBreedsModel(object):
    def __init__(self):
        super().__init__()
        self.pre_model = torch.nn.Sequential(*(list(timm.create_model('resnet50',
                pretrained=True, num_classes=1000).children())[:-1]))
        self.log_reg = pickle.load(open("Models/Log_Reg_on_Resnet50_cats_features.sav", 'rb'))

    def predict(self, img: torch.Tensor) -> Tuple[str, float]:
        with torch.no_grad():
            X = self.pre_model(img)
        return cat_breeds[str(int(self.log_reg.predict(X))-1)], np.sort(self.log_reg.predict_proba(X)[0])[-1]


class AnimalTypeModel(object):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model.to(device)
        self.model.load_state_dict(torch.load("./Models/ResNet18TL.pt", map_location=torch.device(device)))
        self.model.eval()

    def predict(self, img: np.ndarray) -> str:
        img = torch.tensor(img)
        with torch.no_grad():
            model_pred = self.model(img)
            ans = animal_types[str(int((torch.max(model_pred, 1)[1])))]
        return ans
