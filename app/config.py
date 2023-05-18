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

SITE = 'TW'

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

TRAINING_STATUS = {
    'RUNNING': 'Running',
    'DONE': 'Done'
}

PROJECTS = [
    'NK_DAOI_CHIPRC', 
    'NK_DAOI_CHIPRC_2',
    'NK_PCIE_2'
]

# Model Folder Path

MOBILENET_TRAIN_MODEL_DIR = './app/models/mobilenet_train'

YOLO_INFERENCE_MODEL_DIR = './app/models/yolo_inference'

YOLO_TRAIN_MODEL_DIR = './app/models/yolo_train'

GAN_INFERENCE_MODEL_DIR = './app/models/gan_inference'

CLASSIFICATION_INFERNCE_MODEL_DIR = './app/models/classification_inference'

# CVAT Settings

CVAT_URL = 'http://10.0.13.80:8080'

CVAT_LOGIN_API = '/api/auth/login'

CVAT_LOGOUT_API = '/api/auth/logout'

CVAT_TASKS_API = '/api/tasks'

CVAT_TASKS_DATA_API = '/api/tasks/{}/data'

CVAT_TASKS_STATUS_API = '/api/tasks/{}/status'

CVAT_TASKS_ANNOTATION_API = '/api/tasks/{}/annotations'

CVAT_TASKS_DATASET_API = '/api/tasks/{}/dataset'

CVAT_LOGIN_INFORMATION = {
    'username': 'admin',
    'password': '!QAZ2wsx3edc',
}

CVAT_ANNOTATION_FORMAT = 'PASCAL VOC 1.1'

CVAT_DOWNLOAD_FORMAT = 'YOLO 1.1'

CVAT_UPLOAD_INFORMATION = {'image_quality': 70}

# YOLOv5 Settings

YOLOV5_DIR = 'app/yolov5'

YOLOV5_BATCH_SIZE = 8

YOLOV5_EPOCHS = 300

# Training Flow Settings

TRAINING_FLOW = {
    'NK_DAOI_CHIPRC_2': ['object_detection'],
    'NK_PCIE_2': ['object_detection']
}

# Object Detection PCIE Settings

OBJECT_DETECTION_PCIE_CLASSIFICATION_MEAN = [0.1522, 0.2014, 0.3004]

OBJECT_DETECTION_PCIE_CLASSIFICATION_STD = [0.1180, 0.1408, 0.1840]

OBJECT_DETECTION_PCIE_CLASSIFICATION_CLASS_NAMES = ['NG-dark', 'NG-melt', 'NG-others', 'OK']

OBJECT_DETECTION_PCIE_CLASSIFICATION_NGS = ['MELT', 'FOV', 'BIT', 'PIN_CHK', 'STAN', 'SHIFT', 'PIN_CHK']

OBJECT_DETECTION_PCIE_WAYS = {
    '-DO':0, 
    '-UP':1, 
    '-L':2, 
    '-R':3
}

OBJECT_DETECTION_PCIE_PART_NUMBER = {
    'DIMM':0, 
    'JPCIE':1, 
    'PCIE':2
}

OBJECT_DETECTION_PCIE_PCIE_THRESHOLD = 0.7

OBJECT_DETECTION_PCIE_BODY_THRESHOLD = 0.7

OBJECT_DETECTION_PCIE_PICKLE_MODEL_NAME = 'pcie_pinlocation_model.pkl'
