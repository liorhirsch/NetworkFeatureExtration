from typing import List

from pandas import np
from scipy.stats import moment
from torch.nn import Linear, Conv2d

from src.FeatureExtractors.BaseFE import BaseFE
from src.utils import pad_layer_outputs


class WeightStatisticsFE(BaseFE):

    def __init__(self, model_with_rows):
        super(WeightStatisticsFE, self).__init__(model_with_rows)
        self.MAX_LAYER_SIZE = 1000
        self.MAX_LAYERS = 10

    def extract_feature_map(self):
        layer_weights_for_each_row: List[Linear | Conv2d] = list(map(lambda row: row[0], self.model_with_rows.all_rows))

        moment_map = [[], [], [], []]
        min_max_map = [[], []]

        for curr_layer in layer_weights_for_each_row:
            if type(curr_layer) == Linear:
                self.handle_linear_layer(curr_layer, min_max_map, moment_map)

        return np.array([*moment_map, *min_max_map])

    def handle_linear_layer(self, curr_layer, min_max_map, moment_map):
        linear_layer: Linear = curr_layer
        curr_layer_weights = linear_layer.weight.tolist()
        mean = np.mean(curr_layer_weights, axis=1)
        other_moments = moment(curr_layer_weights, [2, 3, 4], axis=1)
        all_moments = [mean, *other_moments]
        all_moments_padded = list(map(lambda moment: pad_layer_outputs(moment, self.MAX_LAYER_SIZE), all_moments))
        min_per_neuron = pad_layer_outputs(np.min(curr_layer_weights, axis=1), self.MAX_LAYER_SIZE)
        max_per_neuron = pad_layer_outputs(np.max(curr_layer_weights, axis=1), self.MAX_LAYER_SIZE)
        min_max_map[0].append(min_per_neuron)
        min_max_map[1].append(max_per_neuron)
        for i, curr_moment in enumerate(all_moments_padded):
            moment_map[i].append(curr_moment)
