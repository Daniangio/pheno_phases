import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, version, input_features, output_features, **kwargs):
        super().__init__()
        self.version = version
        self.input_features = input_features
        self.output_features = output_features
    
    def get_weights_path(self, root, place, variety) -> str:
        raise NotImplementedError
    
    def run_inference(self, src, device):
        raise NotImplementedError