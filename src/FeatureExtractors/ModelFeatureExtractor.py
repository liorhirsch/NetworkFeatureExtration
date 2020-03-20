from typing import List

from src.FeatureExtractors import BaseFE
from src.FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE
from src.FeatureExtractors.TopologyFE import TopologyFE
from src.FeatureExtractors.WeightStatisticsFE import WeightStatisticsFE
from src.ModelWithRows import ModelWithRows


class FeatureExtractor:
    def __init__(self, model, X):
        self.model_with_rows = ModelWithRows(model)

        self.all_feature_extractors: List[BaseFE] = [
            TopologyFE(self.model_with_rows),
            ActivationsStatisticsFE(self.model_with_rows, X),
            WeightStatisticsFE(self.model_with_rows)
        ]

    def extract_features(self, layer_index):
        a = [curr_fe.extract_feature_map(layer_index) for curr_fe in self.all_feature_extractors]
        return a
