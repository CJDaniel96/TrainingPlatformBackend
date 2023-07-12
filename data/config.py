# Database Settings

DATABASES = {
    'ai': {
        'ENGINE': 'postgresql',
        'NAME': 'ai',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': '10.0.4.188',
        'PORT': '5432',
    }, 
    'amr_info': {
        'ENGINE': 'postgresql',
        'NAME': 'amr_nifi_test',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': '10.0.4.188',
        'PORT': '5432',
    }, 
    'cvat': {
        'ENGINE': 'postgresql',
        'NAME': 'cvat',
        'USER': 'root',
        'PASSWORD': 'password',
        'HOST': '10.0.13.80',
        'PORT': '5432',
    }
}

# Datasets Environment Settings

OBJECT_DETECTION_BASICLINE_DATASETS_DIR = './data/datasets/object_detection_basicline_datasets'

CLASSIFICATION_BASICLINE_DATASETS_DIR = './data/datasets/classification_basicline_datasets'

OBJECT_DETECTION_TRAIN_DATASETS_DIR = './data/datasets/object_detection_train_datasets'

CLASSIFICATION_TRAIN_DATASETS_DIR = './data/datasets/classification_train_datasets'

OBJECT_DETECTION_VALIDATION_DATASETS_DIR = './data/datasets/object_detection_validation_datasets'

CLASSIFICATION_VALIDATION_DATASETS_DIR = './data/datasets/classification_validation_datasets'

OBJECT_DETECTION_UNDERKILL_DATASETS_DIR = './data/datasets/object_detection_underkill_datasets'

CLASSIFICATION_UNDERKILL_DATASETS_DIR = './data/datasets/classification_underkill_datasets'

OBJECT_DETECTION_INFERENCE_DATASETS_DIR = './data/datasets/object_detection_inference_datasets'

CLASSIFICATION_INFERENCE_DATASETS_DIR = './data/datasets/classification_inference_datasets'

ORIGIN_DATASETS_DIR = './data/datasets/origin_datasets'

ORIGIN_DATASETS_FOLDER_PROFIX = 'org_data'

YOLO_TRAIN_DATA_YAML_DIR = './data/datasets/yolo_train_yamls/data'

YOLO_TRAIN_MODELS_YAML_DIR = './data/datasets/yolo_train_yamls/models'

YOLO_TRAIN_HYPS_YAML_DIR = './data/datasets/yolo_train_yamls/hyps'

# Data Tmp Settings

TMP_DIR = './data/tmp'