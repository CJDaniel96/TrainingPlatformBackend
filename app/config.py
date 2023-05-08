IRI_RECORD_STATUS = [
    'Init',
    'OD_Initialized', 
    'CLS_Initialized'
]

URD_RECORD_STATUS = [
    'Init',
    'Categorizing',
    'OD_Initialized', 
    'CLS_Initialized'
]

LISTEN_DATABASE_TIME_SLEEP = 60

# Image Pool Settings

IMAGE_POOL_DOWNLOAD_URL = 'http://172.20.20.10:8888/imagesinzip'

IMAGE_POOL_DOWNLOAD_PROXIES = {'http': 'http://172.20.20.10:8888/imagesinzip'}

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

PROJECTS = [
    'NK_DAOI_CHIPRC', 
    'NK_DAOI_CHIPRC_2',
    'NK_PCIE_2'
]

# Model Folder Path

INFERENCE_MODEL_DIR = './app/models/inference'

YOLO_TRAIN_MODEL_DIR = './app/models/yolo_train'

# CVAT Settings

CVAT_URL = 'http://10.0.13.80:8080'

CVAT_LOGIN_API = '/api/auth/login'

CVAT_LOGOUT_API = '/api/auth/logout'

CVAT_TASKS_API = '/api/tasks'

CVAT_TASKS_DATA_API = '/api/tasks/{}/data'

CVAT_TASKS_STATUS_API = '/api/tasks/{}/status'

CVAT_TASKS_ANNOTATION_API = '/api/tasks/{}/annotations'

CVAT_LOGIN_INFORMATION = {
    'username': 'admin',
    'password': '!QAZ2wsx3edc',
}

CVAT_ANNOTATION_FORMAT = 'PASCAL VOC 1.1'

CVAT_UPLOAD_INFORMATION = {'image_quality': 70}

YOLOV5_DIR = 'app/yolov5'