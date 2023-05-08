import os
from app.services.logging_service import Logger
from data.config import ORIGIN_DATASETS_DIR


class CheckDatasetsEnvironment:
    @classmethod
    def check_origin_datasets_path(cls):
        Logger.info('Check Origin Datasets Path')
        if not os.path.exists(ORIGIN_DATASETS_DIR):
            os.makedirs(ORIGIN_DATASETS_DIR)