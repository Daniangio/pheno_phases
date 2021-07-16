from phenoai.manager.ai_manager import AIManager, ai_manager
from phenoai.manager.weather_station_manager import WeatherStationManager, weather_station_manager
from fastapi.params import Depends
from phenoai.manager.data_persistance_manager import DataPersistanceManager, data_persistance_manager
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from phenoai.api.data import domain
import json

router = APIRouter()


@router.get("/data/places", status_code=200, tags=["data"])
def get_places_info():
    return domain.get_places_info()


@router.get("/data/phases/{place}/{variety}",
            status_code=200,
            tags=["data"])
def get_pheno_phases(place: str, variety: str):
    place_info = domain.get_places_info(place=place)['places'][0]
    if variety not in place_info['varieties']:
        raise HTTPException(status_code=404, detail=f'Variety {variety} not found for place {place}')
    pheno_phases_df = domain.get_pheno_phases_df(place=place, variety=variety)
    return pheno_phases_df


@router.get("/data/input/{place}/{variety}/{year}",
            status_code=200,
            tags=["data"])
def get_input_data(place: str, variety: str, year: int,
                   dpm: DataPersistanceManager = Depends(data_persistance_manager)):
    place_info = domain.get_places_info(place=place)['places'][0]
    if variety not in place_info['varieties']:
        raise HTTPException(status_code=404, detail=f'Variety {variety} not found for place {place}')
    try:
        input_data_df = domain.get_input_data_df(place=place, variety=variety, year=year, dpm=dpm)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'Input data not found for place {place}, variety {variety} and year {year}')
    return input_data_df


@router.put("/data/input/{place}/{variety}/{year}",
            status_code=201,
            tags=["data"])
def update_input_data(place: str,
                      variety: str,
                      year: int,
                      force_new: Optional[bool] = False,
                      dpm: DataPersistanceManager = Depends(data_persistance_manager),
                      wsm: WeatherStationManager = Depends(weather_station_manager)):
    place_info = domain.get_places_info(place=place)['places'][0]
    if variety not in place_info['varieties']:
        raise HTTPException(status_code=404, detail=f'Variety {variety} not found for place {place}')
    input_data_df = domain.update_input_data_df(place=place, variety=variety, year=year, force_new=force_new, dpm=dpm, wsm=wsm)
    return input_data_df

@router.put("/data/inference/{place}/{variety}/{year}",
            status_code=200,
            tags=["data"])
def run_inference(place: str, variety: str, year: int,
                  dpm: DataPersistanceManager = Depends(data_persistance_manager),
                  aim: AIManager = Depends(ai_manager)):
    place_info = domain.get_places_info(place=place)['places'][0]
    if variety not in place_info['varieties']:
        raise HTTPException(status_code=404, detail=f'Variety {variety} not found for place {place}')
    inference_df = domain.run_inference(place=place, variety=variety, year=year, dpm=dpm, aim=aim)
    return inference_df