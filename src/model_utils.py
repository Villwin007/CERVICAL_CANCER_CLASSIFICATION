import torch.nn as nn
from torchvision.models import resnet50
from config import NUM_CLASSES

def get_model(pretrained=True):
    model = resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
