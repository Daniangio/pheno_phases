import os
from phenoai.manager.weather_station_manager import mocked_weather_station_manager, weather_station_manager
import ssl
import json
import pytest
import shutil
import logging
import urllib3
from fastapi.testclient import TestClient

from phenoai.factory import create_app
from settings.constants import APP_APIKEY_HEADER, VITIGEOSS_DATA_ROOT, VITIGEOSS_PHENO_PHASES_DIR, VITIGEOSS_INPUT_DATA_DIR
from settings.instance import settings

# disable other loggers, dirty but working
logging.getLogger(urllib3.__name__).setLevel(logging.WARNING)

LOG = logging.getLogger(pytest.__name__)


def json_format(text: str) -> str:
    """Simple utility to parse json text and format it for printing.
    """
    obj = json.loads(text)
    return json.dumps(obj, indent=4, sort_keys=True)


@pytest.fixture(scope="session")
def test_app():
    LOG.info("Initializing webserver and test database...")
    app = create_app(settings=settings)
    app.dependency_overrides[weather_station_manager] = mocked_weather_station_manager
    yield app


@pytest.fixture(scope="session")
def test_client(test_app):
    LOG.debug("Initializing test client...")
    client = TestClient(app=test_app)
    client.headers.update({APP_APIKEY_HEADER: settings.app_secret})
    yield client


@pytest.fixture(scope="session")
def test_zip():
    LOG.debug('Initializing test zip file...')
    path = os.path.join(TEST_DATA_DIR, "package")
    filename = os.path.join(TEST_DATA_DIR, "package")
    shutil.make_archive(filename, format="zip", root_dir=path)
    yield f"{filename}.zip"
    LOG.debug("Deleting test zip file...")
    os.remove(f"{filename}.zip")


@pytest.fixture(scope="session")
def test_image():
    LOG.debug('Initializing test zip file...')
    path = os.path.join(TEST_DATA_DIR, "package")
    filename = os.path.join(path, "kitten.jpg")
    yield filename


@pytest.fixture(scope="session", autouse=True)
def cleanup(request, test_client: TestClient):
    """Cleanup a testing directory once we are finished."""

    def teardown():
        LOG.debug('********* TEARDOWN *********')
        remove_test_folders()

    request.addfinalizer(teardown)


def remove_test_folders():
    try:
        shutil.rmtree(os.path.join(VITIGEOSS_DATA_ROOT, VITIGEOSS_PHENO_PHASES_DIR), ignore_errors=False, onerror=None)
        shutil.rmtree(os.path.join(VITIGEOSS_DATA_ROOT, VITIGEOSS_INPUT_DATA_DIR), ignore_errors=False, onerror=None)
    except FileNotFoundError:
        pass
