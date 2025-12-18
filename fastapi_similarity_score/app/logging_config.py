import logging
import sys

from pythonjsonlogger import jsonlogger
from app.config import settings


def setup_logging()->None:
    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL.upper())

    log_handler = logging.StreamHandler(stream=sys.stdout)

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    log_handler.setFormatter(formatter)


    logger.handlers  = []
    logger.addHandler(log_handler)
   