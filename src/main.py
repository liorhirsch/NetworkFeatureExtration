import os
import time
import numpy as np
import pandas as pd
import torch


from src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from src.ModelClasses.NetX.netX import NetX


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model

for root, dirs, files in os.walk("../Fully Connected Training/"):
    load_times = []
    fe_times = []

    if ('X_to_train.csv' not in files):
        continue

    X = pd.read_csv(root + '/X_to_train.csv')

    model_files = filter(lambda file_name: file_name.endswith('.pt'), files)

    for file in model_files:

        start = time.time()
        model = load_checkpoint(os.path.join(root, file))
        end = time.time()
        load_times.append(end - start)

        start = time.time()
        feature_extractor = FeatureExtractor(model, X._values)
        feature_extractor.extract_features(layer_index=1)
        end = time.time()
        fe_times.append(end-start)

    print("Len dataset : ", len(X))
    print("Load Times : ", np.mean(load_times), " | Min : ", min(load_times), " | Max : ", max(load_times))
    print("FE Times : ", np.mean(fe_times), " | Min : ", min(fe_times), " | Max : ", max(fe_times))
    print("Done ", root)

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
