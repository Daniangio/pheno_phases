import os

from pandas.core.frame import DataFrame
from settings.constants import VITIGEOSS_DATA_ROOT, VITIGEOSS_INPUT_DATA_DIR
import pandas as pd
from settings.instance import settings


class DataPersistanceManager:
    def __init__(self) -> None:
        self.df_path = None

    def load_df(self, place: str, variety: str, year: str, force_new: bool, fail_if_not_found: bool=False) -> pd.DataFrame:
        """
        Look among saved files for a dataframe containing the input data
        :param place, variety, year: parameters that uniquely identify the dataframe to look for
        :param force_new: if True, ignore saved dataframes and return an empty one (initialized with the days of the year as indexes)
        :param fail_if_not_found: if True, raise a FileNotFoundError if the dataframe does not exist
        :return: pandas DataFrame
        """
        self.df_path = self.get_input_data_csv_path(place, variety, year)
        if os.path.exists(self.df_path) and not force_new:
            df = pd.read_csv(self.df_path, sep=',', error_bad_lines=False, index_col=0)
            df.index = pd.to_datetime(df.index)
        else:
            if fail_if_not_found:
                raise FileNotFoundError(f'Dataframe {self.df_path} not found!')
            index = pd.date_range(start=f'1/1/{year}', end=f'31/12/{year}', freq='D')
            df = pd.DataFrame(index=index, columns=settings.get_input_data_columns())
        return df
    
    def save_df(self, df: DataFrame):
        df.to_csv(self.df_path)
    
    @staticmethod
    def get_input_data_csv_path(place: str, variety: str, year: int):
        return os.path.join(VITIGEOSS_DATA_ROOT, VITIGEOSS_INPUT_DATA_DIR, f'{place}_{variety}_{year}_input_data_df.csv')

def data_persistance_manager():
    dpm = DataPersistanceManager()
    try:
        yield dpm
    finally:
        del(dpm)