from enum import Enum

import numpy as np
import torch

from ..FeatureExtractors.BaseFE import BaseFE


class Activations(Enum):
    ReLU = 1
    Softmax = 2
    Tanh = 3
    Sigmoid = 4


class BatchNorm(Enum):
    BatchNorm1d = 1


class TopologyFE(BaseFE):
    BatchNorm = 1
    Activation = 2
    Dropout = 3
    InFeatures = 4
    OutFeatures = 5

    def __init__(self, model_with_rows):
        super(TopologyFE, self).__init__(model_with_rows)
        self.all_layers = []
        self.MAX_LAYERS = 10

        self.layer_type_to_function = {
            torch.nn.modules.linear.Linear: self.handle_linear,

            torch.nn.modules.batchnorm.BatchNorm1d: self.handle_batchnorm(BatchNorm.BatchNorm1d),

            torch.nn.modules.activation.ReLU: self.handle_activation(Activations.ReLU),
            torch.nn.modules.activation.Softmax: self.handle_activation(Activations.Softmax),
            torch.nn.modules.activation.Tanh: self.handle_activation(Activations.Tanh),
            torch.nn.modules.activation.Sigmoid: self.handle_activation(Activations.Sigmoid),

            torch.nn.modules.dropout.Dropout: self.handle_dropout,
        }

    def extract_feature_map(self, layer_index):
        topology_map = np.zeros((self.MAX_LAYERS, 5))

        for i, curr_row in enumerate(self.model_with_rows.all_rows):
            for curr_layer in curr_row:
                self.layer_type_to_function[type(curr_layer)](curr_layer, topology_map[i])

        return (topology_map, topology_map[layer_index])

    def handle_linear(self, curr_layer, row_to_fill):
        row_to_fill[TopologyFE.InFeatures] = curr_layer.in_features
        row_to_fill[TopologyFE.OutFeatures] = curr_layer.out_features

    def handle_dropout(self, curr_layer, row_to_fill):
        row_to_fill[TopologyFE.Dropout] = 1

    def handle_batchnorm(self, batchnorm_type):
        def handler(curr_layer, row_to_fill):
            row_to_fill[TopologyFE.BatchNorm] = batchnorm_type.value

        return handler

    def handle_activation(self, activation_type):
        def handler(curr_layer, row_to_fill):
            row_to_fill[TopologyFE.Activation] = activation_type.value
            pass;

        return handler