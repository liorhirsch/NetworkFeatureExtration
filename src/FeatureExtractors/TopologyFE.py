from enum import Enum

import numpy as np
import torch

from src.FeatureExtractors.BaseFE import BaseFE


class MainLayers(Enum):
    Conv2D = 1
    Linear = 2


class Activations(Enum):
    ReLU = 1


class BatchNorm(Enum):
    BatchNorm2d = 1


class Pooling(Enum):
    MaxPool2d = 1
    AdaptiveAvgPool2d = 2
    AvgPool2d = 3


class TopologyFE(BaseFE):
    LayerMainType = 0
    KernelSize = 1
    StrideSize = 2
    PaddingSize = 3
    BatchNorm = 4
    Activation = 5
    Polling = 6
    InFeatures = 7
    OutFeatures = 8
    Dropout = 9

    def __init__(self, model_with_rows):
        super(TopologyFE, self).__init__(model_with_rows)
        self.all_layers = []

        self.layer_type_to_function = {
            torch.nn.modules.conv.Conv2d: self.handle_conv2d,
            torch.nn.modules.linear.Linear: self.handle_linear,

            torch.nn.modules.batchnorm.BatchNorm2d: self.handle_batchnorm(BatchNorm.BatchNorm2d),

            torch.nn.modules.activation.ReLU: self.handle_activation(Activations.ReLU),

            torch.nn.modules.dropout.Dropout: self.handle_dropout,

            torch.nn.modules.pooling.MaxPool2d: self.handle_pooling(Pooling.MaxPool2d),
            torch.nn.modules.pooling.AdaptiveAvgPool2d: self.handle_pooling(Pooling.AdaptiveAvgPool2d),
            torch.nn.modules.pooling.AvgPool2d: self.handle_pooling(Pooling.AvgPool2d)
        }

    def extract_feature_map(self):
        print(self.all_layers)

        topology_map = np.zeros((len(self.model_with_rows.all_rows), 10))

        for i, curr_row in enumerate(self.model_with_rows.all_rows):
            for curr_layer in curr_row:
                self.layer_type_to_function[type(curr_layer)](curr_layer, topology_map[i])

        return topology_map

    def handle_conv2d(self, curr_layer, row_to_fill):
        row_to_fill[TopologyFE.LayerMainType] = MainLayers.Conv2D.value
        row_to_fill[TopologyFE.KernelSize] = curr_layer.kernel_size[0]
        row_to_fill[TopologyFE.StrideSize] = curr_layer.stride[0]
        row_to_fill[TopologyFE.PaddingSize] = curr_layer.padding[0]

    def handle_linear(self, curr_layer, row_to_fill):
        row_to_fill[TopologyFE.LayerMainType] = MainLayers.Linear.value
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

    def handle_pooling(self, pooling_type):
        def handler(curr_layer, row_to_fill):
            row_to_fill[TopologyFE.Polling] = pooling_type.value

        return handler
