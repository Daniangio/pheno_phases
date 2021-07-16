import logging
from phenoai.api import data, inference
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from typing import Callable
from settings.instance import ProjectSettings


def create_app(settings: ProjectSettings) -> FastAPI:
    """Factory method that creates a new FastAPI application.

    :return: configured FastAPI instance
    :rtype: FastAPI
    """
    app = FastAPI(title=settings.api_title,
                  version=settings.app_version,
                  description=settings.api_description)
    register_middlewares(app)
    register_routers(app)
    # customizes logging (partially, due to Uvicorn)
    init_logging(name=__name__, settings=settings)
    return app


def init_logging(name: str, settings: ProjectSettings):
    """Initializes the logging module to default values, so that subpackages can have the sanem formats.

    :param name: name of the root logger
    :type name: str
    :return: logger instance
    :rtype: Logger
    """
    logger = logging.getLogger(name)
    uvilog = logging.getLogger(uvicorn.__name__)
    logger.setLevel(settings.app_log_level.upper())
    if uvilog.handlers:
        handler = uvilog.handlers[0]
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(settings.app_log_format))
    logger.addHandler(handler)
    logger.info("Logger configuration completed")
    return logger


def register_middlewares(app: FastAPI):
    """Registers middlewares to the main application instance.

    :param app: app instance
    :type app: FastAPI
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def register_routers(app: FastAPI):
    """Registers all the available submodules to the main application.

    :param app: FastAPI instance
    :type app: FastAPI
    """
    app.include_router(data.router, tags=["data"])
    app.include_router(inference.router, tags=["inference"])


def update_upload_schema(app: FastAPI, function: Callable, name: str) -> None:
    """
    Updates the Pydantic schema name for a FastAPI function that takes
    in a fastapi.UploadFile = File(...) or bytes = File(...).
    """
    for route in app.routes:
        if route.endpoint == function:
            route.body_field.type_.__name__ = name
            break
