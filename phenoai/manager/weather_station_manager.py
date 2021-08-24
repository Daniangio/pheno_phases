from dataclasses import make_dataclass
from datetime import datetime, timedelta
import logging
import os
import requests
from settings.constants import TEST_DATA_DIR, VITIGEOSS_CONFIG_FILE
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

def get_bearerAuth(credentials: dict):
    auth_url = os.path.join(settings.vitigeoss_api_base_url, settings.vitigeoss_api_auth_endpoint)
    response = requests.post(auth_url, data=credentials).json()
    b_id, bearer = response['id'], response['token']['hash']
    bearerAuth = BearerAuth(f'{b_id}-{bearer}')
    return bearerAuth


class WeatherStationDriver:
    """
    Driver class in charge of interfacing with the Vitigeoss APIs to retrieve data coming from the sensors of
    the weather stations
    """
    def __init__(self) -> None:
        """ Configure the endpoints to be called using settings """
        self.api_base_url = settings.vitigeoss_api_base_url
        self.station_endpoint = settings.get_vitigeoss_api_station_endpoint
        self.sensor_endpoint = settings.get_vitigeoss_api_sensor_endpoint
        self.sensor_name_id_dict = None

    def register_station_sensor_ids(self, station: str, type_keys: list, bearerAuth: BearerAuth=get_bearerAuth(settings.get_api_auth_credentials())):
        """
        Retrieve the list of sensors available at the specified station and filter them by their typeKey.
        Save the filtered sensors ids in a dictionary, where the key is the corresponding typeKey

        :param station: string containing the name of the weather station whose sensors are to be listed
        :param type_keys: List of type keys that identify the type of sensor, e.g. ['temp', 'radiation']
        :param bearerAuth: authentication token for the API service
        :return: None - The class attribute "sensor_name_id_dict" is initialized
        """
        station_url = os.path.join(self.api_base_url, self.station_endpoint(station))
        response = requests.get(station_url, auth=bearerAuth)
        data = response.json()
        if response.status_code not in [200, 201]:
                logger.warning(data)
                return
        self.sensor_name_id_dict = dict()
        for sensor in data['sensors']:
            if sensor['typeKey'] in type_keys:
                self.sensor_name_id_dict[sensor['typeKey']] = sensor['_id']
    
    def get_sensor_data(self, dateStart: str, dateEnd: str, bearerAuth: BearerAuth=get_bearerAuth(settings.get_api_auth_credentials())):
        """
        For each type of sensor saved in the class attribute "sensor_name_id_dict" by the "register_station_sensor_ids()" method,
        request to the Vitigeoss APIs the sensor data of the period between the dateStart and dateEnd dates

        :returns: list of dictionaries, one for each sensor
        """
        if self.sensor_name_id_dict is None:
            raise Exception(f'Sensor ids not registered!')
        sensors = []
        for _, _id in self.sensor_name_id_dict.items():
            sensor_url = f'{os.path.join(self.api_base_url, self.sensor_endpoint(_id))}?dateStart={dateStart}&dateEnd={dateEnd}&includeFields=dateStart,measure'
            logger.warning(sensor_url)
            response = requests.get(sensor_url, auth=bearerAuth)
            sensor_data = response.json()
            if response.status_code not in [200, 201]:
                logger.warning(sensor_data)
                continue
            sensors.append(sensor_data)
        return sensors
    
    @staticmethod
    def get_df_from_sensor_data(sensors: list):
        """ Support method to organize the messy data coming from sensors in a single and good looking dataframe """
        Measurement = make_dataclass("Measurement", [("datetime", str), ("typeKey", str), ("measure", float)])
        measurements_list = []
        for sensor in sensors:
            for sensor_measurement in sensor['measurements']:
                measurement = Measurement(datetime.strptime(sensor_measurement['dateStart'], '%Y-%m-%dT%H:%M:%S.000Z'), sensor['typeKey'], sensor_measurement['measure'])
                measurements_list.append(measurement)
        if len(measurements_list) == 0:
            return None
        return pd.DataFrame(measurements_list)


class MockedWeatherStationDriver(WeatherStationDriver):
    """
    Mocked version of the WeatherStationDriver, used for testing purposes.
    Instead of calling the external APIs to retrieve weather station data,
    it reads a mocked sample from a json file and returns its content
    """
    def __init__(self) -> None:
        super().__init__()
    
    def get_sensor_data(self, dateStart: str, dateEnd: str):
        if self.sensor_name_id_dict is None:
            raise Exception(f'Sensor ids not registered!')
        with open(os.path.join(TEST_DATA_DIR, 'mocked_sensor_data.json')) as f:
            mocked_sensor_data = json.loads(f.read())
        return mocked_sensor_data


class WeatherStationManager:
    def __init__(self, driver=WeatherStationDriver()) -> None:
        with open(VITIGEOSS_CONFIG_FILE) as f:
            self.config = json.loads(f.read())
        self.input_data_features = settings.input_data_features
        self.driver = driver

    def get_wsdata_df(self, place, weather_station_missing_rows, chunk_days=366):
        """
        Retrieve weather data from weather station sensors of the specified place.
        This method only requests the data of the dates whose data are missing (not the data of the entire year).
        """
        if weather_station_missing_rows.empty:
            raise Exception('The Dataframe to be updated is empty!')
        self.driver.register_station_sensor_ids(station=self.config['place_to_station'][place], type_keys=self.input_data_features)
        weather_station_data_df = None
        for dateStart in pd.date_range(weather_station_missing_rows.index[0].to_pydatetime(),
                                       weather_station_missing_rows.index[-1].to_pydatetime(), freq=f'{chunk_days}d'):
            # Compute dateEnd, chunk_days - 1 days later than dateStart. maximum dateEnd is the 31st of December of that year
            dateEnd = min(dateStart.to_pydatetime() + timedelta(days=chunk_days - 1), datetime(dateStart.year, 12, 31))
            dateEnd = dateEnd + timedelta(hours=23, minutes=59, seconds=59)
            try:
                weekly_sensor_data = self.driver.get_sensor_data(dateStart, dateEnd)
                weekly_data_df = self.driver.get_df_from_sensor_data(weekly_sensor_data)
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
        update_df = self.manually_fix_df_errors(update_df, place, dateStart.year)
        update_df = update_df.interpolate(limit_direction='both') # Interpolate to fill missing data values
        return update_df
    
    def organize_weather_station_data(self, weather_station_data_df: pd.DataFrame, pheno_phases_df: pd.DataFrame):
        """
        Process data and obtain a refined version of the pandas DataFrame to be used as input for the inference models.
        Still some steps have to be taken to obtain the final version of the df.
        """
        dataframes = []
        for ft in self.input_data_features:
            dataframes.append(self.transform_df(weather_station_data_df[weather_station_data_df['typeKey'] == ft]))
        
        transformed_station_data_df = pd.concat(dataframes, axis=1)
        transformed_station_data_df.index.name = 'datetime'
        transformed_station_data_df = transformed_station_data_df.sort_index()
        transformed_station_data_df.index = pd.to_datetime(transformed_station_data_df.index)
        # TODO pheno_phases_df are missing, find a way to save and update them
        # transformed_station_data_df = add_phenological_phases_one_hot(transformed_station_data_df, phases_df)
        return transformed_station_data_df

    def add_phenological_phases_one_hot(df: pd.DataFrame, phases_df: pd.DataFrame):
        # ADD THE PHENOLOGICAL PHASES COLUMNS TO THE DF. FOR EACH PHASE, USE A 1 ON THE DATES OF THAT PHASE, 0 OTHERWISE
        idx = pd.IndexSlice
        for index, column in enumerate(phases_df.columns):
            df[column] = 0
        for year in phases_df.index:
            if year != df.index[0].year:
                continue
            for index, column in enumerate(phases_df.columns[:-1]):
                i = index
                while i < len(phases_df.columns[:-1]):
                    try:
                        # set to 1 all dates >= date of the phenological phase begin in the column of that phenological phase
                        df.loc[idx[phases_df.loc[phases_df.index == year, column].values[0]
                                :
                                phases_df.loc[phases_df.index == year, phases_df.columns[i + 1]].values[0]],
                            idx[column]] = 1
                        break
                    except Exception:
                        i += 1
                        if i == len(phases_df.columns[:-1]):
                            try:
                                df.loc[idx[phases_df.loc[phases_df.index == year, column].values[0]:], idx[column]] = 1
                            except Exception:
                                pass
            # add harvest after veraison
            column = phases_df.columns[-1]
            try:
                df.loc[idx[phases_df.loc[phases_df.index == year, column].values[0]:], idx[column]] = 1
            except Exception:
                continue
        return df
    
    def manually_fix_df_errors(self, df: pd.DataFrame, place: str, year: int):
        """
        Hardcoded method used to read MIRABELLA ECLANO weather station data from an external file,
        since the weather station data coming from sensors have too many gaps
        """
        if place == 'mastroberardino':
            correct_wsdata = pd.read_csv('/mnt/extra/data/WEATHERSTATIONDATA_MIRABELLA_1974-2020.csv', sep=',', error_bad_lines=False, index_col=0)
            correct_wsdata.columns = ['temp', 'temp_min', 'temp_max', 'humidity', 'humidity_min', 'humidity_max', 'ppt', 'vvm2', 'rad24h']
            correct_wsdata.index = pd.to_datetime(correct_wsdata.index)
            correct_wsdata = correct_wsdata.sort_index()
            correct_wsdata = correct_wsdata[correct_wsdata.index >= datetime(year,1,1)]
            correct_wsdata = correct_wsdata[correct_wsdata.index < datetime(year + 1,1,1)]
            cols = df.columns
            df = df.combine_first(correct_wsdata)
            df = df[cols]
        return df


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
    
    @staticmethod
    def feature_engineering(df: pd.DataFrame):
        """
        Raggruppo i dati giornalmente, traformando ogni feature in 3 diverse features:
        - la media di tutti i valori misurati nella giornata
        - la media dei 5 valori minimi
        - la media dei 5 valori massimi
        """
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
        engineered_df[settings.input_data_source] = 'WS' # Set the source of the data to WS (Weather Station). If a dataframe has some dates
        # whose source is not WS (for example because they are future dates, or data are missing), a next step will compile those missing data
        # using data coming from Weather Forecasts
        
        return engineered_df


def weather_station_manager():
    wsm = WeatherStationManager()
    try:
        yield wsm
    finally:
        del(wsm)

def mocked_weather_station_manager():
    wsm = WeatherStationManager(driver=MockedWeatherStationDriver())
    try:
        yield wsm
    finally:
        del(wsm)