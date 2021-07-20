from typing import List
from pydantic import BaseSettings
from phenoai.version import __version__


class ProjectSettings(BaseSettings):
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

    # Deployment settings
    build_target: str
    api_title: str = "VITIGEOSS AI API"
    api_description: str = ""
    app_secret: str = 'notusedyet'

    app_log_level: str = "info"
    app_log_format: str = "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"

    # Vitigeoss API
    vitigeoss_api_base_url: str
    vitigeoss_api_auth_endpoint: str
    vitigeoss_api_auth_email: str
    vitigeoss_api_auth_password: str

    # AI settings
    input_data_features: List = ['temp', 'radiation']
    input_data_phases: List = ['budBreak', 'flowering', 'fruitSet', 'veraison', 'harvest']
    input_data_source: str = 'source'

    @property
    def app_version(self):
        return __version__
    
    def get_input_data_columns(self):
        extended_data_features = [item for sublist in [[ft, f'{ft}_min', f'{ft}_max',] for ft in self.input_data_features] for item in sublist]
        return extended_data_features + self.input_data_phases + [self.input_data_source]
    
    def get_vitigeoss_api_station_endpoint(self, station: str):
        return f'sensor/stations/{station}/'
    
    def get_vitigeoss_api_sensor_endpoint(self, sensor_id: str):
        return f'sensor/sensors/{sensor_id}/'
    
    def get_api_auth_credentials(self):
        return {
            'email': self.vitigeoss_api_auth_email,
            'password': self.vitigeoss_api_auth_password
        }

settings = ProjectSettings()
