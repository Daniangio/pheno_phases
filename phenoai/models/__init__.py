import json
from enum import Enum
from settings.constants import VITIGEOSS_CONFIG_FILE
from phenoai.models.transformerlstm import TransformerLSTM

class ModelsEnum(Enum):
    TRANSFORMER_LSTM = TransformerLSTM, 'transformer_lstm'


    @staticmethod
    def make(model, place, variety):
        with open(VITIGEOSS_CONFIG_FILE) as f:
            config = json.loads(f.read())
        parameters = config['model_to_parameters'][f'{model.value[1]}|{place}|{variety}']
        return model.value[0](**parameters), parameters