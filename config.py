import os
from os import getenv


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class EnvConfig:
    SECRET_KEY = getenv("SECRET_KEY") or "Hard To Guess String"
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    
    AI_DB_ENGINE = getenv("AI_DB_ENGINE")
    AI_DB_NAME = getenv("AI_DB_NAME")
    AI_DB_USER = getenv("AI_DB_USER")
    AI_DB_PASSWORD = getenv("AI_DB_PASSWORD")
    AI_DB_HOST = getenv("AI_DB_HOST")
    AI_DB_PORT = getenv("AI_DB_PORT")

    AMR_DB_ENGINE = getenv("AMR_DB_ENGINE")
    AMR_DB_NAME = getenv("AMR_DB_NAME")
    AMR_DB_USER = getenv("AMR_DB_USER")
    AMR_DB_PASSWORD = getenv("AMR_DB_PASSWORD")
    AMR_DB_HOST = getenv("AMR_DB_HOST")
    AMR_DB_PORT = getenv("AMR_DB_PORT")
    
    CVAT_DB_ENGINE = getenv("CVAT_DB_ENGINE")
    CVAT_DB_NAME = getenv("CVAT_DB_NAME")
    CVAT_DB_USER = getenv("CVAT_DB_USER")
    CVAT_DB_PASSWORD = getenv("CVAT_DB_PASSWORD")
    CVAT_DB_HOST = getenv("CVAT_DB_HOST")
    CVAT_DB_PORT = getenv("CVAT_DB_PORT")
    
    @staticmethod
    def init_app(app):
        pass
    
    
class DevelopmentConfig(EnvConfig):
    SQLALCHEMY_DATABASE_URI = "postgresql://main"
    SQLALCHEMY_BINDS = {
        "ai": getenv("AI_DEV_DATABASE_URL") or f"{EnvConfig.AI_DB_ENGINE}://{EnvConfig.AI_DB_USER}:{EnvConfig.AI_DB_PASSWORD}@{EnvConfig.AI_DB_HOST}:{EnvConfig.AI_DB_PORT}/{EnvConfig.AI_DB_NAME}", 
        "amr": getenv("AMR_DEV_DATABASE_URL") or f"{EnvConfig.AMR_DB_ENGINE}://{EnvConfig.AMR_DB_USER}:{EnvConfig.AMR_DB_PASSWORD}@{EnvConfig.AMR_DB_HOST}:{EnvConfig.AMR_DB_PORT}/{EnvConfig.AMR_DB_NAME}",
        "cvat": getenv("CVAT_DEV_DATABASE_URL") or f"{EnvConfig.CVAT_DB_ENGINE}://{EnvConfig.CVAT_DB_USER}:{EnvConfig.CVAT_DB_PASSWORD}@{EnvConfig.CVAT_DB_HOST}:{EnvConfig.CVAT_DB_PORT}/{EnvConfig.CVAT_DB_NAME}"
    }


class TestingConfig(EnvConfig):
    SQLALCHEMY_DATABASE_URI = "postgresql://main"
    SQLALCHEMY_BINDS = {
        "ai": getenv("AI_TEST_DATABASE_URL") or f"{EnvConfig.AI_DB_ENGINE}://{EnvConfig.AI_DB_USER}:{EnvConfig.AI_DB_PASSWORD}@{EnvConfig.AI_DB_HOST}:{EnvConfig.AI_DB_PORT}/{EnvConfig.AI_DB_NAME}", 
        "amr": getenv("AMR_TEST_DATABASE_URL") or f"{EnvConfig.AMR_DB_ENGINE}://{EnvConfig.AMR_DB_USER}:{EnvConfig.AMR_DB_PASSWORD}@{EnvConfig.AMR_DB_HOST}:{EnvConfig.AMR_DB_PORT}/{EnvConfig.AMR_DB_NAME}",
        "cvat": getenv("CVAT_TEST_DATABASE_URL") or f"{EnvConfig.CVAT_DB_ENGINE}://{EnvConfig.CVAT_DB_USER}:{EnvConfig.CVAT_DB_PASSWORD}@{EnvConfig.CVAT_DB_HOST}:{EnvConfig.CVAT_DB_PORT}/{EnvConfig.CVAT_DB_NAME}"
    }
    

class ProductionConfig(EnvConfig):
    SQLALCHEMY_DATABASE_URI = "postgresql://main"
    SQLALCHEMY_BINDS = {
        "ai": getenv("AI_DATABASE_URL") or f"{EnvConfig.AI_DB_ENGINE}://{EnvConfig.AI_DB_USER}:{EnvConfig.AI_DB_PASSWORD}@{EnvConfig.AI_DB_HOST}:{EnvConfig.AI_DB_PORT}/{EnvConfig.AI_DB_NAME}", 
        "amr": getenv("AMR_DATABASE_URL") or f"{EnvConfig.AMR_DB_ENGINE}://{EnvConfig.AMR_DB_USER}:{EnvConfig.AMR_DB_PASSWORD}@{EnvConfig.AMR_DB_HOST}:{EnvConfig.AMR_DB_PORT}/{EnvConfig.AMR_DB_NAME}",
        "cvat": getenv("CVAT_DATABASE_URL") or f"{EnvConfig.CVAT_DB_ENGINE}://{EnvConfig.CVAT_DB_USER}:{EnvConfig.CVAT_DB_PASSWORD}@{EnvConfig.CVAT_DB_HOST}:{EnvConfig.CVAT_DB_PORT}/{EnvConfig.CVAT_DB_NAME}"
    }
    
    
ENV_CONFIG = {
    "development": DevelopmentConfig,
    "testing": TestingConfig, 
    "production": ProductionConfig, 
    "default": DevelopmentConfig
}




# Training Platform Backend Settings

SITE = 'TW'
WATCHING_DATABASE_CYCLE_TIME = 600


# Record Status

RECORD_STATUSES = {
    'iri_record': ['Init', 'OD_Initialized', 'CLS_Initialized'],
    'urd_record': ['Init', 'Categorizing', 'OD_Initialized', 'CLS_Initialized'],
}


# Images Assign Light Type Settings

IMAGES_ASSIGN_LIGHT_TYPE = {
    'ZJ': {
        'ChipRC': 'side',
        'XTAL': 'side', 
        'SAW': 'side', 
        'WLCSP567L': 'top', 
        'MC': 'side'
    }
}


# Data Settings

# Image Pool Settings

IMAGE_DOWNLOAD_PREFIX = 'images/'

DOWNLOADS_DATA_DIR = 'resources/downloads'

MODELS_DIR = 'resources/models'

BASELINE_DATASETS_DIR = 'resources/datasets/baseline'

TRAIN_DATASETS_DIR = 'resources/datasets/train'

VALIDATION_DATASETS_DIR = 'resources/datasets/validation'

UNDERKILLS_DATASETS_DIR = 'resources/datasets/underkills'

YAMLS_DIR = 'resources/yamls'

SMART_FILTER_NUMBER = 200

TRAIN_TEST_RATIO = 0.2


# Training Platform Record Status List

TRAINING_PLATFORM_RECORD_STATUS = {
    # Init Status
    'INFERENCE_ON_GOING': 'Inference On going',
    'INFERENCE_FINISH': 'Inference finish', 
    'UPLOAD_IMAGE_WITH_LOG_ON_GOING': 'Upload imagewith log on going', 
    'UPLOAD_IMAGE_WITH_LOG_FINISH': 'Upload imagewith log finish', 

    # OD Training Status
    'TRIGGER_TRAINING_FOR_OD': 'Trigger training for OD', 
    'TRAINING_FOR_OD': 'Training for OD', 
    'VERIFYING_FOR_OD': 'Verifying for OD', 
    'FINISH_FOR_OD': 'Finish for OD', 

    # CLS Training Status
    'TRIGGER_TRAINING_FOR_CLS': 'Trigger training for CLS', 
    'TRAINING_FOR_CLS': 'Training for CLS', 
    'VERIFYING_FOR_CLS': 'Verifying for CLS', 
    'FINISH_FOR_CLS': 'Finish for CLS', 

    # Finish Status
    'FINISHED': 'Finished'
}


# Training Flow Settings
# Only Selection as Following
# 1. object_detection
# 2. classification

TRAINING_FLOW = {
    # NK Site
    'TW': {
        'NK_DAOI_CHIPRC_2': {'object_detection': 'yolov5'},
        'NK_PCIE_2': {'object_detection': 'yolov5'}
    },
    # ZJ Site
    'ZJ': {
        'ZJ_CHIPRC': {'object_detection': 'yolov5'},
        'ZJ_IC': {'object_detection': 'yolov5'}, 
        'ZJ_XTAL': {'object_detection': 'yolov5'}, 
        'ZJ_SAW': {'object_detection': 'yolov5'}, 
        'ZJ_WLCSP567L': {'object_detection': 'yolov5'}, 
        'ZJ_MC': {'object_detection': 'yolov5'}, 
        'ZJ_SAW_POLARITY': {'classification': 'mobilenetv2', 'metric_learning': ''}
    },
    # HZ Site
    'HZ': {
        'HZ_CHIPRC': {'object_detection': 'yolov5'}, 
        'HZ_PCIE': {'object_detection': 'yolov5'}
    },
    # JQ Site
    'JQ': {
        'JQ_4PINS': {'object_detection': 'yolov5'}, 
        'JQ_CHIPRC': {'object_detection': 'yolov5'}, 
        'JQ_ICBGA': {'object_detection': 'yolov5'}, 
        'JQ_FILTER': {'object_detection': 'yolov5'}, 
        'JQ_NEFANG': {'object_detection': 'yolov5'}, 
        'JQ_XTAL': {'object_detection': 'yolov5'}, 
        'JQ_SOT': {'object_detection': 'yolov5'}
    }
}




# CVAT Settings

CVAT_INFO = {
    'CVAT_URL': 'http://10.0.13.80:8080',
    'CVAT_LOGIN_API': '/api/auth/login',
    'CVAT_LOGOUT_API': '/api/auth/logout',
    'CVAT_TASKS_API': '/api/tasks',
    'CVAT_TASKS_DATA_API': '/api/tasks/{}/data',
    'CVAT_TASKS_STATUS_API': '/api/tasks/{}/status',
    'CVAT_TASKS_ANNOTATION_API': '/api/tasks/{}/annotations',
    'CVAT_TASKS_DATASET_API': '/api/tasks/{}/dataset',
    'CVAT_LOGIN_INFORMATION': {
        'username': 'admin',
        'password': '!QAZ2wsx3edc',
    },
    'CVAT_ANNOTATION_FORMAT': 'PASCAL VOC 1.1',
    'CVAT_DOWNLOAD_FORMAT': 'PASCAL VOC 1.1',
    'CVAT_CLASSIFICATION_FORMAT': 'VGGFace2 1.0',
    'CVAT_UPLOAD_INFORMATION': {'image_quality': 70},
}


# Algorithm Settings

# Yolov5

YOLOV5_DATA_YAML = 'data.yaml'

YOLOV5_MODEL_YAML = 'yolov5s.yaml'

YOLOV5_CLOSE_RANDOM_HYP_YAML = 'hyp.random-crop-close.yaml'

YOLOV5_HYP_YAML = 'hyp.scratch-low.yaml'

YOLOV5_HYP_RANDOM_CROP_CLOSE_PROJECT = ['NK_DAOI_CHIPRC_2', 'ZJ_CHIPRC', 'HZ_CHIPRC', 'JQ_CHIPRC']

YOLOV5_DIR = 'algorithm/yolov5'

YOLOV5_BATCH_SIZE = 8

YOLOV5_EPOCHS = 3

YOLOV5_SEED = 42