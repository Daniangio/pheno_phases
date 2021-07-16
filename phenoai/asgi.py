from phenoai.factory import create_app
from settings.instance import settings

app = create_app(settings=settings)
