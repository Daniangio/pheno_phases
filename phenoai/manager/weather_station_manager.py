from dataclasses import make_dataclass
from datetime import datetime, timedelta
import logging
import os
import requests
from settings.constants import VITIGEOSS_CONFIG_FILE
import pandas as pd
import json
from settings.instance import settings

logger = logging.getLogger()


class BearerAuth(requests.auth.AuthBase):
        def __init__(self, token):
            self.token = token
        def __call__(self, r):
            r.headers["authorization"] = "Bearer " + self.token
            return r


# Authentication
def get_bearerAuth(credentials: dict):
    auth_url = os.path.join(settings.vitigeoss_api_base_url, settings.vitigeoss_api_auth_endpoint)
    response = requests.post(auth_url, data=credentials).json()
    b_id, bearer = response['id'], response['token']['hash']
    bearerAuth = BearerAuth(f'{b_id}-{bearer}')
    return bearerAuth


class WeatherStationManager:
    def __init__(self) -> None:
        with open(VITIGEOSS_CONFIG_FILE) as f:
            self.config = json.loads(f.read())
        self.input_data_features = settings.input_data_features

    def get_wsdata_df(self, place, weather_station_missing_rows):
        if weather_station_missing_rows.empty:
            raise Exception('The Dataframe to be updated is empty!')
        weather_station_data_df = None
        for dateStart in pd.date_range(weather_station_missing_rows.index[0].to_pydatetime(),
                                    weather_station_missing_rows.index[-1].to_pydatetime(), freq='7d'):
            # Compute dateEnd, 6 days later than dateStart. maximum dateEnd is the 31st of December of that year
            dateEnd = min(dateStart.to_pydatetime() + timedelta(days=6), datetime(dateStart.year, 12, 31))
            dateEnd = dateEnd + timedelta(hours=23, minutes=59, seconds=59)
            try:
                weekly_sensor_data = self.get_sensor_data(self.config['place_to_station'][place], dateStart, dateEnd)
                weekly_data_df = self.get_df_from_sensor_data(weekly_sensor_data)
                if weather_station_data_df is None:
                    weather_station_data_df = weekly_data_df
                else:
                    weather_station_data_df = weather_station_data_df.append(weekly_data_df, ignore_index=True)
            except Exception as e:
                logger.warning(e)
        if weather_station_data_df is None:
            return None
        pheno_phases_df = None
        update_df = self.organize_weather_station_data(weather_station_data_df, pheno_phases_df)
        update_df = self.feature_engineering(update_df)
        return update_df
    
    def organize_weather_station_data(self, weather_station_data_df: pd.DataFrame, pheno_phases_df: pd.DataFrame):
        dataframes = []
        for ft in self.input_data_features:
            dataframes.append(self.transform_df(weather_station_data_df[weather_station_data_df['typeKey'] == ft]))
        
        transformed_station_data_df = pd.concat(dataframes, axis=1)
        transformed_station_data_df.index.name = 'datetime'
        transformed_station_data_df = transformed_station_data_df.sort_index()
        transformed_station_data_df = transformed_station_data_df.interpolate()
        transformed_station_data_df.index = pd.to_datetime(transformed_station_data_df.index)
        # transformed_station_data_df = add_phenological_phases_one_hot(transformed_station_data_df, phases_df)
        return transformed_station_data_df
    
    @staticmethod
    def get_sensor_data(station: str, dateStart: str, dateEnd: str, bearerAuth: BearerAuth=get_bearerAuth(settings.get_api_auth_credentials())):
        sensor_url = f'{os.path.join(settings.vitigeoss_api_base_url, settings.get_vitigeoss_api_sensor_endpoint(station))}?dateStart={dateStart}&dateEnd={dateEnd}'
        return requests.get(sensor_url, auth=bearerAuth).json()
    
    @staticmethod
    def get_df_from_sensor_data(sensor_data: dict): # Organize sensor data in a dataframe
        Measurement = make_dataclass("Measurement", [("datetime", str), ("typeKey", str), ("measure", float)])
        sensors = sensor_data['sensors']
        measurements_list = []
        for sensor in sensors:
            for sensor_measurement in sensor['measurements']:
                measurement = Measurement(datetime.strptime(sensor_measurement['dateStart'], '%Y-%m-%dT%H:%M:%S.000Z'), sensor['typeKey'], sensor_measurement['measure'])
                measurements_list.append(measurement)
        if len(measurements_list) == 0:
            return None
        return pd.DataFrame(measurements_list)

    @staticmethod
    def transform_df(df: pd.DataFrame):
        df = df.set_index('datetime').sort_index()
        df['measure'] = df['measure'].astype(float)
        df = df.rename(columns={'typeKey': 'drop', 'measure': df.typeKey[0]})
        df = df.drop(columns=['drop'])
        df.index = pd.to_datetime(df.index)
        df_grouped = df.groupby(pd.Grouper(level='datetime',freq='1H'))
        df = df_grouped.mean()
        return df
    
    # Raggruppo i dati giornalmente, traformando ogni feature in 3 diverse features:
    # - la media dei 5 valori minimi
    # - la media dei 5 valori massimi
    # - la somma di tutti i valori misurati nella giornata
    @staticmethod
    def feature_engineering(df: pd.DataFrame):
        daily_grouped_df = df.groupby(pd.Grouper(level='datetime',freq='1D'))
        
        engineered_df = daily_grouped_df.mean()
        
        for feature in settings.input_data_features:
            engineered_df[f'{feature}_min'] = daily_grouped_df[feature].nsmallest(5).groupby(pd.Grouper(level=0, freq='1D')).mean()
            engineered_df[f'{feature}_max'] = daily_grouped_df[feature].nlargest(5).groupby(pd.Grouper(level=0, freq='1D')).mean()
        
        engineered_df = engineered_df.reindex(settings.get_input_data_columns(), axis=1)
        engineered_df.loc[engineered_df.budBreak < 1, 'budBreak'] = 0
        engineered_df.loc[engineered_df.flowering < 1, 'flowering'] = 0
        engineered_df.loc[engineered_df.fruitSet < 1, 'fruitSet'] = 0
        engineered_df.loc[engineered_df.veraison < 1, 'veraison'] = 0
        engineered_df.loc[engineered_df.harvest < 1, 'harvest'] = 0
        engineered_df[settings.input_data_source] = 'WS'
        
        return engineered_df

def weather_station_manager():
    wsm = WeatherStationManager()
    try:
        yield wsm
    finally:
        del(wsm)