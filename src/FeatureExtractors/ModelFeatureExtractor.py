from src.FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE
from src.FeatureExtractors.TopologyFE import TopologyFE
from src.ModelWithRows import ModelWithRows


class FeatureExtractor():
    def __init__(self, model, X):
        self.model_with_rows = ModelWithRows(model)


        self.all_feature_extractors = [
            TopologyFE(self.model_with_rows),
            ActivationsStatisticsFE(self.model_with_rows, X)
        ]

    def extract_features(self):

        a = [curr_fe.extract_feature_map() for curr_fe in self.all_feature_extractors]
        print(a)
        # for curr_fe in :
        #     curr_fe.extract_feature_map()