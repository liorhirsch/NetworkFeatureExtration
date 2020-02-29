from src.FeatureExtractors.BaseFE import BaseFE
import torch
import numpy as np
from scipy.stats import moment


class ActivationsStatisticsFE(BaseFE):
    def __init__(self, model_with_rows, dataset_x):
        super(ActivationsStatisticsFE, self).__init__(model_with_rows)
        self.MAX_LAYER_SIZE = 1000
        self.MAX_LAYERS = 10
        self.dataset_x = dataset_x

    def extract_feature_map(self):
        important_layer_in_each_row = []
        moment_map = [[], [], [], []]
        min_max_map = [[], []]

        self.register_forward_hooks_to_important_layers(important_layer_in_each_row)
        self.model_with_rows.model(torch.Tensor(self.dataset_x))
        self.calculate_moments_for_each_layer(moment_map, min_max_map)

        print(all_activations_in_important_layers)

    def pad_layer_outputs(self, layer_statistics):
        pad_size = self.MAX_LAYER_SIZE - len(layer_statistics)
        return np.pad(layer_statistics, (0, pad_size))

    def calculate_moments_for_each_layer(self, moment_map, min_max_map):
        for layer_activations in all_activations_in_important_layers:
            layer_activations_transposed = np.array(layer_activations).T

            mean = np.mean(layer_activations_transposed, axis=1)
            other_moments = moment(layer_activations_transposed, [2, 3, 4], axis=1)
            all_moments = [mean, *other_moments]

            min_per_neuron = np.min(layer_activations_transposed, axis=1)
            max_per_neuron = np.max(layer_activations_transposed, axis=1)

            min_max_map[0].append(self.pad_layer_outputs(min_per_neuron))
            min_max_map[1].append(self.pad_layer_outputs(max_per_neuron))

            for m in range(0, 4):
                pad_size = self.MAX_LAYER_SIZE - len(all_moments[m])
                padded = np.pad(all_moments[m], (0, pad_size))
                moment_map[m].append(padded)

    def register_forward_hooks_to_important_layers(self, important_layer_in_each_row):
        for curr_row in self.model_with_rows.all_rows[:-1]:
            most_imporatant_layer = curr_row[0]

            for curr_layer in curr_row[1:]:
                if ('activation' in str(type(curr_layer))):
                    most_imporatant_layer = curr_layer

            most_imporatant_layer.register_forward_hook(save_activations)
            important_layer_in_each_row.append(most_imporatant_layer)


all_activations_in_important_layers = []


def save_activations(self, input, output):
    # TODO - add support for CNN
    global all_activations_in_important_layers
    all_activations_in_important_layers.append(output.detach().numpy())
    return None
