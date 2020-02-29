import torch
import torchvision.models as models
import pandas as pd

from src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from src.ModelClasses.Net2.Net2 import Net2

model = Net2()
model.load_state_dict(torch.load("./ModelClasses/Net2/net2model.pt"))
model.eval()

X = pd.read_csv('./ModelClasses/Net2/X_to_train.csv')

# resnet18
# model = models.vgg19_bn(pretrained=True)
feature_extractor = FeatureExtractor(model, X._values)
feature_extractor.extract_features()