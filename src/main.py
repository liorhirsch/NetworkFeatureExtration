import os
import time
import numpy as np
import pandas as pd
import torch


from .FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from .ModelClasses.NetX.netX import NetX


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model

def load_model_and_data(model_path, data_path, layer_index):
    """
    :param model_path:
    :param data_path:
    :param layer_index:
    :return:
    """
    X = pd.read_csv(data_path)
    model = load_checkpoint(model_path)

    return model, X

def get_fm_for_model_and_layer(model, data, layer_index):
    feature_extractor = FeatureExtractor(model, data._values)
    return feature_extractor.extract_features(layer_index)

def main():
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

            end = time.time()
            fe_times.append(end-start)

        print("Len dataset : ", len(X))
        print("Load Times : ", np.mean(load_times), " | Min : ", min(load_times), " | Max : ", max(load_times))
        print("FE Times : ", np.mean(fe_times), " | Min : ", min(fe_times), " | Max : ", max(fe_times))
        print("Done ", root)

