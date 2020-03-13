import pandas as pd
import torch

from src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from src.ModelClasses.Net2.Net2 import Net2

import os

from src.ModelClasses.NetX.netX import NetX


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model

for root, dirs, files in os.walk("../Fully Connected Training/Regression/"):
    if ('X_to_train.csv' not in files):
        continue

    X = pd.read_csv(root + '/X_to_train.csv')

    model_files = filter(lambda file_name: file_name.endswith('.pt'), files)

    for file in model_files:

        model = load_checkpoint(os.path.join(root, file))

        feature_extractor = FeatureExtractor(model, X._values)
        feature_extractor.extract_features()

# model = NetX()
#
# model.load_state_dict(torch.load("./ModelClasses/Net2/net2model.pt"))
# model.eval()
#
# X = pd.read_csv('./ModelClasses/Net2/X_to_train.csv')
#
# # model.load_state_dict(torch.load("./ModelClasses/Net2/net2model.pt"))
# # model.eval()
# #
# # X = pd.read_csv('./ModelClasses/Net2/X_to_train.csv')
#
# # resnet18
# # model = models.vgg19_bn(pretrained=True)
# feature_extractor = FeatureExtractor(model, X._values)
# feature_extractor.extract_features()
