import numpy as np
import torch
from scipy.stats import variation, skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from .BaseFE import BaseFE
from ..utils import pad_with_columns, pad_with_rows, get_scaler_exponent


class ActivationsStatisticsFE(BaseFE):
    def __init__(self, model_with_rows, dataset_x, device):
        super(ActivationsStatisticsFE, self).__init__(model_with_rows)
        self.MAX_LAYER_SIZE = 1000
        self.MAX_LAYERS = 10
        self.dataset_x = dataset_x
        self.device = device

    def extract_feature_map(self, layer_index):
        moment_map = [[], [], [], []]
        min_max_map = [[], []]
        norm_map = [[], []]

        unregister_hook_functions = self.build_register_forward_hooks_to_important_layers()
        all_activations_in_important_layers.clear()
        self.model_with_rows.model(torch.Tensor(self.dataset_x).to(self.device))
        [x.remove() for x in unregister_hook_functions]
        self.calculate_stats_for_each_layer(moment_map, min_max_map, norm_map)
        activations_map = np.array([*moment_map, *min_max_map, *norm_map])

        activations_map = np.array(list(map(lambda f_map : pad_with_rows(f_map, self.MAX_LAYERS),activations_map)))

        return (activations_map, activations_map[:,layer_index,:])

    def calculate_stats_for_each_layer(self, moment_map, min_max_map, norm_map):
        base_feature_map = np.zeros((self.MAX_LAYERS, self.MAX_LAYER_SIZE))

        for layer_activations in all_activations_in_important_layers:
            layer_activations_transposed = np.array(layer_activations).T
            # scaler = MinMaxScaler()
            # layer_activations_transposed_scaled = scaler.fit_transform(layer_activations).T
            scaler_exponent = get_scaler_exponent(layer_activations_transposed)
            layer_activations_transposed_scaled = layer_activations_transposed * (10 ** -scaler_exponent)
            layer_activations_transposed_scaled = np.array(layer_activations_transposed_scaled, dtype=float)


            mean = np.mean(layer_activations_transposed, axis=1)
            variation_val = np.std(layer_activations_transposed, axis=1)
            skew_val = skew(layer_activations_transposed_scaled, axis=1)
            kurtosis_val = kurtosis(layer_activations_transposed_scaled, axis=1)

            all_moments = [mean, variation_val, skew_val, kurtosis_val]

            min_per_neuron = np.min(layer_activations_transposed, axis=1)
            max_per_neuron = np.max(layer_activations_transposed, axis=1)

            min_max_map[0].append(pad_with_columns(min_per_neuron, self.MAX_LAYER_SIZE))
            min_max_map[1].append(pad_with_columns(max_per_neuron, self.MAX_LAYER_SIZE))

            l1_per_neuron = np.linalg.norm(layer_activations_transposed, axis=1, ord=1)
            l2_per_neuron = np.linalg.norm(layer_activations_transposed, axis=1, ord=2)

            norm_map[0].append(pad_with_columns(l1_per_neuron, self.MAX_LAYER_SIZE))
            norm_map[1].append(pad_with_columns(l2_per_neuron, self.MAX_LAYER_SIZE))

            for m in range(0, 4):
                moment_map[m].append(pad_with_columns(all_moments[m], self.MAX_LAYER_SIZE))



    def build_register_forward_hooks_to_important_layers(self):
        unregister_hook_function = []
        for curr_row in self.model_with_rows.all_rows[:-1]:
            most_imporatant_layer = curr_row[0]

            for curr_layer in curr_row[1:]:
                if ('activation' in str(type(curr_layer))):
                    most_imporatant_layer = curr_layer

            unregister_hook_function.append(most_imporatant_layer.register_forward_hook(save_activations))

        return unregister_hook_function


all_activations_in_important_layers = []


def save_activations(self, input, output):
    global all_activations_in_important_layers
    all_activations_in_important_layers.append(output.detach().cpu().numpy())
    return None
