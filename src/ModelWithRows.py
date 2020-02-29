import torch


class ModelWithRows():
    def __init__(self, model):
        self.model = model
        self.all_layers = []
        self.main_layer_types = [torch.nn.modules.linear.Linear, torch.nn.modules.conv.Conv2d]

        self.extract_layers_from_model(self.model)
        self.all_rows = self.split_layers_to_rows()

    def extract_layers_from_model(self, layer):
        for layer in layer.children():
            if len(list(layer.children())):
                self.extract_layers_from_model(layer)
            else:
                self.all_layers.append(layer)

    def split_layers_to_rows(self):
        all_rows = []
        curr_row = []
        for curr_layer in self.all_layers:
            if type(curr_layer) in self.main_layer_types and len(curr_row) > 0:
                all_rows.append(curr_row)
                curr_row = []

            curr_row.append(curr_layer)

        if (len(curr_row)) > 0:
            all_rows.append(curr_row)

        return all_rows
