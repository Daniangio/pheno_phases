import logging
import os
from phenoai.models import ModelsEnum
from phenoai.manager.ai_manager import AIManager
from phenoai.manager.weather_station_manager import WeatherStationManager
from phenoai.manager.data_persistance_manager import DataPersistanceManager
from typing import Optional
from settings.constants import VITIGEOSS_CONFIG_FILE, VITIGEOSS_DATA_ROOT, VITIGEOSS_PHENO_PHASES_DIR
from settings.instance import settings
from fastapi import HTTPException
import pandas as pd
import json

logger = logging.getLogger()


def get_places_info(place: Optional[str] = None):
    result = {'places': []}
    with open(VITIGEOSS_CONFIG_FILE) as f:
        config = json.loads(f.read())
    places = config['places']
    if place:
        if place not in places:
            raise HTTPException(status_code=404, detail=f'Place {place} not found')
        places = [place]

    for _place in places:
        result['places'].append({
            'name': _place,
            'weather-station': config['place_to_station'][_place],
            'varieties': config['place_to_varieties'][_place]
        })
    return result


def get_pheno_phases_df(place: str, variety: str):
    df_path = get_pheno_phases_csv_path(place, variety)
    if os.path.exists(df_path):
        return pd.read_csv(df_path, sep=',', error_bad_lines=False, index_col=0)
    raise HTTPException(status_code=404, detail=f'Phenological phases dataframe not found. You must build it first.')


def get_pheno_phases_csv_path(place: str, variety: str):
    return os.path.join(VITIGEOSS_DATA_ROOT, VITIGEOSS_PHENO_PHASES_DIR, f'{place}_{variety}_pheno_phases_df.csv')


def get_input_data_df(place: str, variety: str, year: int, dpm: DataPersistanceManager):
    return dpm.load_df(place=place, variety=variety, year=year, force_new=False, fail_if_not_found=True).fillna(0.0)


def update_input_data_df(place: str, variety: str, year: int, force_new: bool,
                         dpm: DataPersistanceManager, wsm: WeatherStationManager):
    df = dpm.load_df(place=place, variety=variety, year=year, force_new=force_new)
    weather_station_missing_rows = df[df[settings.input_data_source] != 'WS']
    if weather_station_missing_rows.empty:
        return df.fillna(0.0)
    update_df = wsm.get_wsdata_df(place, weather_station_missing_rows)
    if update_df is None:
        return df.fillna(0.0)
    df = df.combine_first(update_df)
    dpm.save_df(df)
    return df.fillna(0.0)


def run_inference(place: str, variety: str, year: int, dpm: DataPersistanceManager, aim: AIManager):
    try:
        input_df = dpm.load_df(place=place, variety=variety, year=year, force_new=False, fail_if_not_found=True)
        aim.load_model(ModelsEnum.TRANSFORMER_LSTM, place=place, variety=variety)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    aim.run_inference(input_df, device='cpu')
    return aim.get_inference_result(year=year)
