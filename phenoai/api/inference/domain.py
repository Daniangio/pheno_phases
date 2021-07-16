import logging
from phenoai.api.data import domain
from phenoai.models import ModelsEnum
from phenoai.manager.ai_manager import AIManager
from phenoai.manager.data_persistance_manager import DataPersistanceManager
from fastapi import HTTPException

logger = logging.getLogger()


def run_inference(place: str, variety: str, year: int, dpm: DataPersistanceManager, aim: AIManager):
    try:
        input_df = dpm.load_df(place=place, variety=variety, year=year, force_new=False, fail_if_not_found=True)
        aim.load_model(ModelsEnum.TRANSFORMER_LSTM, place=place, variety=variety)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    aim.run_inference(input_df, device='cpu')
    return aim.get_inference_result(year=year)
