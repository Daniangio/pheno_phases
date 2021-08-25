import logging
import pytest
from fastapi.testclient import TestClient
import pandas as pd

logger = logging.getLogger(pytest.__name__)


@pytest.mark.dependency()
def test_get_places_info(test_client: TestClient):
    response = test_client.get("/data/places")
    data = response.json()
    assert response.status_code == 200
    assert isinstance(data, dict)
    assert isinstance(data['places'], list)
    assert len(data['places']) > 0
    assert isinstance(data['places'][0].get('name'), str)
    assert isinstance(data['places'][0].get('weather-station'), str)
    assert isinstance(data['places'][0].get('varieties'), list)
    assert len(data['places'][0].get('varieties')) > 0
    assert isinstance(data['places'][0].get('varieties')[0], str)

@pytest.mark.dependency(depends=["test_get_places_info"])
def test_get_input_data_wrong_place(test_client: TestClient):
    wrong_place, variety, year = 'notaplace', 'syrah', 2020
    url = f'/data/input/{wrong_place}/{variety}/{year}'
    response = test_client.get(url)
    assert response.status_code == 404
    assert response.json() == {"detail": f"Place {wrong_place} not found"}

@pytest.mark.dependency(depends=["test_get_input_data_wrong_place"])
def test_get_input_data_wrong_variety(test_client: TestClient):
    place, wrong_variety, year = 'torres', 'notavariety', 2020
    url = f'/data/input/{place}/{wrong_variety}/{year}'
    response = test_client.get(url)
    assert response.status_code == 404
    assert response.json() == {"detail": f"Variety {wrong_variety} not found"}

@pytest.mark.dependency(depends=["test_get_input_data_wrong_variety"])
def test_get_input_data_not_created_yet(test_client: TestClient):
    place, variety, year = 'torres', 'syrah', 2020
    url = f'/data/input/{place}/{variety}/{year}'
    response = test_client.get(url)
    assert response.status_code == 404
    assert response.json() == {"detail": f"Input data not found for place {place}, variety {variety} and year {year}"}

@pytest.mark.dependency(depends=["test_get_input_data_not_created_yet"])
def test_update_input_data_create_new(test_client: TestClient):
    place, variety, year = 'torres', 'syrah', 2019
    url = f'/data/input/{place}/{variety}/{year}'
    response = test_client.put(url)
    assert response.status_code == 201
    df = pd.DataFrame(response.json())
    assert len(df) == 365
    assert df.iloc[0, -1] == 'WS'
    assert df['temp'].sum() != 0.0
    assert df['temp_min'].sum() != 0.0
    assert df['temp_max'].sum() != 0.0
