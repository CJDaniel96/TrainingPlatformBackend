# Record Status

RECORD_STATUSES = {
    'IRI': ['Init', 'OD_Initialized', 'CLS_Initialized'],
    'URD': ['Init', 'Categorizing', 'OD_Initialized', 'CLS_Initialized'],
}

LISTEN_DATABASE_TIME_SLEEP = 60

SITE = 'TW'

SMART_FILTER_NUMBER = 200

# Image Pool Settings

IMAGE_POOL_DOWNLOAD_PREFIX = 'images'

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

TRAINING_STATUS = {
    'RUNNING': 'Running',
    'DONE': 'Done'
}

# Project List

PROJECTS = {
    'TW': ['NK_DAOI_CHIPRC', 'NK_DAOI_CHIPRC_2', 'NK_PCIE_2'],
    'ZJ': ['ZJ_CHIPRC', 'ZJ_IC', 'ZJ_XTAL', 'ZJ_SAW', 'ZJ_WLCSP567L', 'ZJ_MC', 'ZJ_SAW_POLARITY'],
    'HZ': ['HZ_CHIPRC', 'HZ_PCIE'],
    'JQ': ['JQ_4PINS', 'JQ_CHIPRC', 'JQ_ICBGA', 'JQ_FILTER', 'JQ_NEFANG', 'JQ_XTAL', 'JQ_SOT'],
}

# Model Folder Path

MODEL_DIRS = {
    'METRIC_LEARNING_TRAIN_MODEL_DIR': './app/models/metric_learning_train',
    'MOBILENET_TRAIN_MODEL_DIR': './app/models/mobilenet_train',
    'YOLO_INFERENCE_MODEL_DIR': './app/models/yolo_inference',
    'YOLO_TRAIN_MODEL_DIR': './app/models/yolo_train',
    'GAN_INFERENCE_MODEL_DIR': './app/models/gan_inference',
    'CLASSIFICATION_INFERNCE_MODEL_DIR': './app/models/classification_inference',
}

# CVAT Settings

CVAT = {
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
    'CVAT_DOWNLOAD_FORMAT': 'YOLO 1.1',
    'CVAT_CLASSIFICATION_FORMAT': 'VGGFace2 1.0',
    'CVAT_UPLOAD_INFORMATION': {'image_quality': 70},
}

# YOLOv5 Settings

YOLOV5 = {
    'YOLOV5_DIR': 'app/yolov5',
    'YOLOV5_BATCH_SIZE': 8,
    'YOLOV5_EPOCHS': 300,
    'YOLOV5S_WEIGHT': 'app/yolov5/yolov5s.pt',
    'YOLOV5_SEED': 42,
    'YOLOV5_HYP_RANDOM_CROP_CLOSE_PROJECT': ['NK_DAOI_CHIPRC_2', 'ZJ_CHIPRC', 'HZ_CHIPRC', 'JQ_CHIPRC'],
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

# Training Flow Settings
# Only Selection as Following
# 1. object_detection
# 2. classification

TRAINING_FLOW = {
    # NK Site
    'NK_DAOI_CHIPRC_2': ['object_detection'],
    'NK_PCIE_2': ['object_detection'], 
    # ZJ Site
    'ZJ_CHIPRC': ['object_detection'],
    'ZJ_IC': ['object_detection'], 
    'ZJ_XTAL': ['object_detection'], 
    'ZJ_SAW': ['object_detection'], 
    'ZJ_WLCSP567L': ['object_detection'], 
    'ZJ_MC': ['object_detection'], 
    'ZJ_SAW_POLARITY': ['classification', 'metric_learning'],
    # HZ Site
    'HZ_CHIPRC': ['object_detection'], 
    'HZ_PCIE': ['object_detection'],
    # JQ Site
    'JQ_4PINS': ['object_detection'], 
    'JQ_CHIPRC': ['object_detection'], 
    'JQ_ICBGA': ['object_detection'], 
    'JQ_FILTER': ['object_detection'], 
    'JQ_NEFANG': ['object_detection'], 
    'JQ_XTAL': ['object_detection'], 
    'JQ_SOT': ['object_detection']
}

# Validation Flow Settings

VALIDATION_FLOW = {
    'yolo_fanogan': {
        'NK_DAOI_CHIPRC_2': {
            'confidence': 0.5, 
            'gan_settings':{
                'img_size': 256,
                'latent_dim': 100,
                'channels': 3,
                'kappa': 1.0, 
                'anormaly_threshold': 0.2
            }
        }, 
        'ZJ_WLCSP567L': {
            'confidence': 0, 
            'gan_settings':{
                'img_size': 128,
                'latent_dim': 100,
                'channels': 3,
                'kappa': 1.0, 
                'anormaly_threshold': 0.22
            }
        }
    },
    'yolo': {
        'HZ_CHIPRC': {
            'confidence': 0.5
        },
        'ZJ_CHIPRC': {},
        'ZJ_SAW': {},
        'ZJ_XTAL': {}, 
        'ZJ_IC': {}, 
        'ZJ_MC': {}, 
        'JQ_4PINS': {}, 
        'JQ_CHIPRC': {}, 
        'JQ_ICBGA': {}, 
        'JQ_FILTER': {}, 
        'JQ_NEFANG': {}, 
        'JQ_XTAL': {}, 
        'JQ_SOT': {}
    },
    'mobilenetv2': {

    },
    'mobilenetv2_fanogan': {
        # 'ZJ_IC': {
        #     'confidence': 0, 
        #     'mean': [0.3248, 0.3176, 0.3038],
        #     'std': [0.2565, 0.2558, 0.2521], 
        #     'gan_settings':{
        #         'kappa': 1.0, 
        #         'anormaly_threshold': 0.2334
        #     }
        # }
    }, 
    'mobilenetv2_yolo_iforest': {
        'NK_PCIE_2': {}, 
        'HZ_PCIE': {}
    },
    'metric_learning': {
        'ZJ_SAW_POLARITY': {
            'INFERENCE_MODE': 'general',
            'CONFIDENCE': 0.95,
            'QUERY_IMAGE_TOP': 'data/datasets/classification_validation_datasets/ZJ_SAW_POLARITY/query_image/golden_sample_top.jpg',
            'QUERY_IMAGE_SIDE': 'data/datasets/classification_validation_datasets/ZJ_SAW_POLARITY/query_image/golden_sample_side.jpg'
        }
    }
}

# Object Detection PCIE Settings

OBJECT_DETECTION_PCIE = {
    'CLASSIFICATION_MEAN': [0.1522, 0.2014, 0.3004],
    'CLASSIFICATION_STD': [0.1180, 0.1408, 0.1840],
    'CLASSIFICATION_CLASS_NAMES': ['NG-dark', 'NG-melt', 'NG-others', 'OK'],
    'CLASSIFICATION_NGS': ['MELT', 'FOV', 'BIT', 'PIN_CHK', 'STAN', 'SHIFT', 'PIN_CHK'],
    'WAYS': {'-DO': 0, '-UP': 1, '-L': 2, '-R': 3},
    'PART_NUMBER': {'DIMM': 0, 'JPCIE': 1, 'PCIE': 2},
    'PCIE_THRESHOLD': 0.7,
    'BODY_THRESHOLD': 0.7,
    'PICKLE_MODEL_NAME': 'pcie_pinlocation_model.pkl',
}

# Underkill Rate

UNDERKILL_RATE = 0.01

# Classification Settings

# MobileNet V2 Settings

MOBILENETV2 = {
    'BATCH_SIZE': 64,
    'EPOCHS': 40,
}

# Metric Learning Settings

# EfficientNet V2 Settings

EFFICIENTNETV2_EMBEDDING = {
    'SEED': 42,
    'BATCH_SIZE': 64,
    'EPOCHS': 100,
    'NUM_CLASSES': 4,
    'EMBEDDING_SIZE': 512,
    'LEARING_RATE': 1e-3,
    'LOSS_LEARING_RATE': 1e-4,
    'PROJECT': ['ZJ_SAW_POLARITY'],
}