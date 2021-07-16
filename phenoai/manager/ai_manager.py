from dataclasses import make_dataclass
from datetime import date, timedelta
import logging
import os
from settings.constants import VITIGEOSS_CONFIG_FILE
from pandas.core.frame import DataFrame
import json
import torch
from torch._C import device, dtype
import torch.nn.functional as F
from phenoai.models import ModelsEnum
import pandas as pd
from settings.instance import settings
import numpy as np

logger = logging.getLogger()


class AIManager:
    def __init__(self) -> None:
        with open(VITIGEOSS_CONFIG_FILE) as f:
            self.config = json.loads(f.read())
        self.model = None
        self.last_prediciton = None

    def load_model(self, model_enum: ModelsEnum, place: str, variety: str):
        self.model, self.model_parameters = ModelsEnum.make(model_enum, place=place, variety=variety)
        path = self.model.get_weights_path(root='/mnt/model_weights', place=place, variety=variety)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
        else:
            raise FileNotFoundError(f'Model weights file {path} for model {model_enum.name} not found!')
    
    def run_inference(self, input_df: DataFrame, device='cpu'):
        self.model = self.model.to(device)
        self.model.eval()
        X = input_df.loc[:, self.model.input_features].to_numpy()
        train_data_min = np.array(self.model_parameters['train_data_min'])
        train_data_max = np.array(self.model_parameters['train_data_max'])
        X_std = (X - train_data_min) / (train_data_max - train_data_min)
        X_scaled = X_std * (self.model_parameters['feature_range'][1] - (self.model_parameters['feature_range'][0])) + (self.model_parameters['feature_range'][0])
        src = torch.from_numpy(X_scaled).float()
        src = src[~torch.any(src.isnan(), dim=1)]
        self.last_prediciton = self.model.run_inference(src.unsqueeze(0), device)

    def get_inference_result(self, year: int):
        prediction_df = pd.DataFrame(self.get_phase_change_dates(self.last_prediciton, year=year), columns=['predicted date'])
        #actual_df = pd.DataFrame(self.get_phase_change_dates(self.last_prediciton, year=year), columns=['actual date'])
        comparison_df = pd.concat([prediction_df], axis=1)
        comparison_df.index = pd.Index(self.model.output_features)
        # comparison_df['error[days]'] = (comparison_df['predicted date'] - comparison_df['actual date']).astype('timedelta64[D]')
        return comparison_df
    
    # segna il cambio di fase alla prima occorrenza di un valore sopra il threshold per quella fase
    @staticmethod
    def get_phase_change_dates(phases_array, year, threshold=0.5):
        # adjust initial predictions to avoid phase changes in the first days of the year (due to model convergence period)
        phases_array[:30,:] = 0
        phase_change_dates = []
        base_date = date(year, 1, 1)
        indices_of_phase_change = []
        for phase in range(phases_array.shape[1]):
            try:
                indices_of_phase_change.append(np.where(phases_array[:, phase] > threshold)[0][0])
            except Exception:
                indices_of_phase_change.append(0)
        for index in indices_of_phase_change:
            delta = timedelta(days=int(index))
            phase_change_dates.append(base_date + delta)
        return phase_change_dates
        
    

def ai_manager():
    aim = AIManager()
    try:
        yield aim
    finally:
        del(aim)