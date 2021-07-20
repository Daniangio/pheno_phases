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
    wrong_place, variety, year = 'notaplace', 'agrolab', 2020
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
    place, variety, year = 'torres', 'agrolab', 2020
    url = f'/data/input/{place}/{variety}/{year}'
    response = test_client.get(url)
    assert response.status_code == 404
    assert response.json() == {"detail": f"Input data not found for place {place}, variety {variety} and year {year}"}

@pytest.mark.dependency(depends=["test_get_input_data_not_created_yet"])
def test_update_input_data_create_new(test_client: TestClient):
    place, variety, year = 'torres', 'agrolab', 2019
    url = f'/data/input/{place}/{variety}/{year}'
    response = test_client.put(url)
    assert response.status_code == 201
    df = pd.DataFrame(response.json())
    assert len(df) == 365
    assert df.iloc[0, -1] == 'WS'
    assert df['temp'].sum() != 0.0
    assert df['temp_min'].sum() != 0.0
    assert df['temp_max'].sum() != 0.0

'''
@pytest.mark.dependency()
def test_get_artifact_empty(test_client: TestClient):
    response = test_client.get("/artifacts/1")
    assert response.status_code == 404
    assert response.json() == {"detail": "Artifact with ID=1 does not exist"}


@pytest.mark.dependency(depends=["test_get_artifacts_empty", "test_get_artifact_empty"])
def test_create_artifact(test_client: TestClient, test_zip: str):
    basename = os.path.basename(test_zip)
    with open(test_zip, "rb") as buffer:
        file = dict(package=(basename, buffer, "multipart/form-data"))
        data = dict(name="squeezenet",
                    version=1,
                    description="test description",
                    handler="model:SqueezeNetHandler")
        response = test_client.post("/artifacts", data=data, files=file)
    LOG.debug(response.text)
    json = response.json()
    assert response.status_code == 201
    assert json["id"] == 1
    assert "created_at" in json
    assert "updated_at" in json
    created_at = parser.isoparse(json["created_at"]).date()
    updated_at = parser.isoparse(json["updated_at"]).date()
    assert datetime.today().date() == created_at
    assert datetime.today().date() == updated_at
    assert created_at == updated_at
    assert json["name"] == "squeezenet"
    assert json["version"] == 1
    assert json["description"] == "test description"
    assert json["handler"] == "model:SqueezeNetHandler"
    destination = os.path.join(settings.models_data_path, "squeezenet", "01")
    assert os.path.exists(destination)
    assert len(os.listdir(destination)) == 6


@pytest.mark.dependency(depends=["test_create_artifact"])
def test_create_artifact_defaults(test_client: TestClient, test_zip: str):
    basename = os.path.basename(test_zip)
    with open(test_zip, "rb") as buffer:
        file = dict(package=(basename, buffer, "multipart/form-data"))
        data = dict(name="squeezenet2", handler="model:SqueezeNetHandler")
        response = test_client.post("/artifacts", data=data, files=file)
    json = response.json()
    assert response.status_code == 201
    assert json["id"] == 2
    assert "created_at" in json
    assert "updated_at" in json
    created_at = parser.isoparse(json["created_at"]).date()
    updated_at = parser.isoparse(json["updated_at"]).date()
    assert datetime.today().date() == created_at
    assert datetime.today().date() == updated_at
    assert created_at == updated_at
    assert json["name"] == "squeezenet2"
    assert json["version"] == 1
    assert json["description"] is None
    assert json["handler"] == "model:SqueezeNetHandler"
    destination = os.path.join(settings.models_data_path, "squeezenet2", "01")
    assert json["model_uri"] == destination
    assert os.path.exists(destination)
    assert len(os.listdir(destination)) == 6


@pytest.mark.dependency(depends=["test_create_artifact", "test_create_artifact_defaults"])
def test_get_artifacts(test_client: TestClient):
    response = test_client.get("/artifacts")
    data = response.json()
    assert len(data) == 2
    for i, item in enumerate(data):
        assert item["id"] == i + 1
        assert "created_at" in item
        assert "updated_at" in item
        assert item["version"] == 1
    assert data[0]["name"] == "squeezenet"
    assert data[1]["name"] == "squeezenet2"
    assert data[0]["description"] == "test description"
    assert data[0]["handler"] == "model:SqueezeNetHandler"
    assert data[1]["description"] is None
    assert data[1]["handler"] == "model:SqueezeNetHandler"


@pytest.mark.dependency(depends=["test_create_artifact"])
def test_get_artifacts_with_non_existent_name(test_client: TestClient):
    params = dict(name="non-existing")
    response = test_client.get("/artifacts", params=params)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 0


@pytest.mark.dependency(depends=["test_create_artifact"])
def test_get_artifacts_with_existing_name(test_client: TestClient):
    params = dict(name="squeezenet")
    response = test_client.get("/artifacts", params=params)
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == 1
    assert data[0]["name"] == "squeezenet"


@pytest.mark.dependency(depends=["test_create_artifact"])
def test_get_artifacts_with_non_existent_versions(test_client: TestClient):
    params = dict(version=2)
    response = test_client.get("/artifacts", params=params)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 0


@pytest.mark.dependency(depends=["test_create_artifact", "test_create_artifact_defaults"])
def test_get_artifacts_with_existing_versions(test_client: TestClient):
    params = dict(version=1)
    response = test_client.get("/artifacts", params=params)
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == 1
    assert data[0]["name"] == "squeezenet"
    assert data[1]["id"] == 2
    assert data[1]["name"] == "squeezenet2"


@pytest.mark.dependency(depends=["test_create_artifact"])
def test_get_non_existent_artifact(test_client: TestClient):
    response = test_client.get("/artifacts/20")
    assert response.status_code == 404
    assert response.json() == {"detail": "Artifact with ID=20 does not exist"}


@pytest.mark.dependency(depends=["test_create_artifact"])
def test_get_existing_artifact(test_client: TestClient):
    response = test_client.get("/artifacts/1")
    assert response.status_code == 200
    item = response.json()
    assert item["id"] == 1
    assert "created_at" in item
    assert "updated_at" in item
    created_at = parser.isoparse(item["created_at"]).date()
    updated_at = parser.isoparse(item["updated_at"]).date()
    assert datetime.today().date() == created_at
    assert datetime.today().date() == updated_at
    assert created_at == updated_at
    assert item["name"] == "squeezenet"
    assert item["version"] == 1
    assert item["description"] == "test description"
    assert item["handler"] == "model:SqueezeNetHandler"


@pytest.mark.dependency(depends=["test_create_artifact"])
def test_update_artifact_wrong_id(test_client: TestClient):
    data = dict(name="example2")
    response = test_client.put("/artifacts/20", data=data)
    assert response.status_code == 404
    assert response.json() == {"detail": "Artifact with ID=20 does not exist"}


@pytest.mark.dependency(depends=["test_create_artifact"])
def test_update_artifact_conflicting_name(test_client: TestClient):
    # try to update the first, using the name of the second
    data = dict(name="squeezenet2")
    response = test_client.put("/artifacts/1", data=data)
    assert response.status_code == 409
    assert response.json() == {"detail": "Artifact (squeezenet2,v1) already exists"}


@pytest.mark.dependency(depends=["test_create_artifact"])
def test_update_artifact_no_file(test_client: TestClient):
    data = dict(name="squeezenet3",
                version=1,
                description="test description updated",
                handler="model:SqueezeNetHandler2")
    response = test_client.put("/artifacts/1", data=data)
    item = response.json()
    assert response.status_code == 201
    assert item["id"] == 1
    assert "created_at" in item
    assert "updated_at" in item
    created_at = parser.isoparse(item["created_at"])
    updated_at = parser.isoparse(item["updated_at"])
    assert created_at < updated_at
    assert item["name"] == "squeezenet3"
    assert item["version"] == 1
    assert item["description"] == "test description updated"
    assert item["handler"] == "model:SqueezeNetHandler2"
    destination = os.path.join(settings.models_data_path, "squeezenet3", "01")
    assert item["model_uri"] == destination
    assert os.path.exists(destination)
    assert len(os.listdir(destination)) == 6


@pytest.mark.dependency(depends=["test_update_artifact_no_file"])
def test_update_artifact_with_file(test_client: TestClient, test_zip: str):
    basename = os.path.basename(test_zip)
    with open(test_zip, "rb") as buffer:
        file = dict(package=(basename, buffer, "multipart/form-data"))
        data = dict(name="squeezenet4",
                    version=1,
                    description="test description updated 2",
                    handler="model:SqueezeNetHandler4")
        response = test_client.put("/artifacts/1", files=file, data=data)
    item = response.json()
    assert response.status_code == 201
    assert item["id"] == 1
    created_at = parser.isoparse(item["created_at"])
    updated_at = parser.isoparse(item["updated_at"])
    assert created_at < updated_at
    assert item["name"] == "squeezenet4"
    assert item["version"] == 1
    assert item["description"] == "test description updated 2"
    assert item["handler"] == "model:SqueezeNetHandler4"
    destination = os.path.join(settings.models_data_path, "squeezenet4", "01")
    assert item["model_uri"] == destination
    assert os.path.exists(destination)
    assert len(os.listdir(destination)) == 6


@pytest.mark.dependency(depends=["test_update_artifact_with_file"])
def test_delete_missing_artifact(test_client: TestClient):
    response = test_client.delete("/artifacts/20")
    assert response.status_code == 404
    assert response.json() == {"detail": "Artifact with ID=20 does not exist"}


@pytest.mark.dependency(depends=["test_update_artifact_with_file"])
def test_delete_artifact(test_client: TestClient):
    response = test_client.delete("/artifacts/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["name"] == "squeezenet4"
    assert test_client.delete("/artifacts/1").status_code == 404
    destination = os.path.join(settings.models_data_path, "squeezenet4", "01")
    assert not os.path.exists(destination)


@pytest.mark.dependency(depends=["test_delete_artifact"])
def test_artifact_cleanup(test_client: TestClient):
    response = test_client.get("/artifacts")
    assert response.status_code == 200
    assert len(response.json()) == 1
    for item in response.json():
        assert item["id"] == 2
        deletion = test_client.delete(f"/artifacts/{item['id']}")
        assert deletion.status_code == 200
    assert os.path.exists(settings.models_data_path)
    assert len(os.listdir(settings.models_data_path)) == 0
'''