from phenoai.api.data.domain import get_places_info
from phenoai.manager.ai_manager import AIManager, ai_manager
from fastapi.params import Depends
from phenoai.manager.data_persistance_manager import DataPersistanceManager, data_persistance_manager
from fastapi import APIRouter, HTTPException
from phenoai.api.inference import domain

router = APIRouter()


@router.put("/inference/{place}/{variety}/{year}",
            status_code=200,
            tags=["inference"])
def run_inference(place: str, variety: str, year: int,
                  dpm: DataPersistanceManager = Depends(data_persistance_manager),
                  aim: AIManager = Depends(ai_manager)):
    _ = get_places_info(place=place, variety=variety) # Check existence of place and variety
    inference_df = domain.run_inference(place=place, variety=variety, year=year, dpm=dpm, aim=aim)
    return inference_df