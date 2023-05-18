import os
import shutil
from app.config import GAN_INFERENCE_MODEL_DIR, MOBILENET_TRAIN_MODEL_DIR, PROJECTS, YOLO_INFERENCE_MODEL_DIR, YOLO_TRAIN_MODEL_DIR
from app.services.logging_service import Logger
from data.config import CLASSIFICATION_BASICLINE_DATASETS_DIR, CLASSIFICATION_INFERENCE_DATASETS_DIR, CLASSIFICATION_UNDERKILL_DATASETS_DIR, CLASSIFICATION_VALIDATION_DATASETS_DIR, OBJECT_DETECTION_INFERENCE_DATASETS_DIR, OBJECT_DETECTION_UNDERKILL_DATASETS_DIR, OBJECT_DETECTION_VALIDATION_DATASETS_DIR, ORIGIN_DATASETS_DIR, OBJECT_DETECTION_BASICLINE_DATASETS_DIR, OBJECT_DETECTION_TRAIN_DATASETS_DIR, CLASSIFICATION_TRAIN_DATASETS_DIR, TMP_DIR, YOLO_TRAIN_DATA_YAML_DIR, YOLO_TRAIN_MODELS_YAML_DIR, YOLO_TRAIN_HYPS_YAML_DIR


class CheckTmpEnvironment:
    @classmethod
    def check_tmp_path(cls):
        Logger.info('Check Tmp Path')
        if not os.path.exists(TMP_DIR):
            Logger.info('Create Tmp Path')
            os.makedirs(TMP_DIR)


class CheckDatasetsEnvironment:
    @classmethod
    def check_origin_datasets_path(cls):
        Logger.info('Check Origin Datasets Path')
        if not os.path.exists(ORIGIN_DATASETS_DIR):
            Logger.info('Create Origin Datasets Path')
            os.makedirs(ORIGIN_DATASETS_DIR)

    @classmethod
    def check_object_detection_basicline_datasets_path(cls):
        Logger.info('Check Object Detection Basicline Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(OBJECT_DETECTION_BASICLINE_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Object Detection Basicline Datasets Path')
                Logger.warn(f'Please Manually Add {project} Object Detection Basicline Datasets')
                os.makedirs(folder)

    @classmethod
    def check_classification_basicline_datasets_path(cls):
        Logger.info('Check Classification Basicline Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(CLASSIFICATION_BASICLINE_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Classification Basicline Datasets Path')
                Logger.warn(f'Please Manually Add {project} Classification Basicline Datasets')
                os.makedirs(folder)

    @classmethod
    def check_object_detection_train_datasets_path(cls):
        Logger.info('Check Object Detection Training Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(OBJECT_DETECTION_TRAIN_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Object Detection Training Datasets Path')
                os.makedirs(folder)

    @classmethod
    def check_classification_train_datasets_path(cls):
        Logger.info('Check Classification Training Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(CLASSIFICATION_TRAIN_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Classification Training Datasets Path')
                os.makedirs(folder)

    @classmethod
    def check_object_detection_validation_datasets_path(cls):
        Logger.info('Check Object Detection Validation Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(OBJECT_DETECTION_VALIDATION_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Object Detection Validation Datasets Path')
                os.makedirs(folder)

    @classmethod
    def check_classification_validation_datasets_path(cls):
        Logger.info('Check Classification Validation Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(CLASSIFICATION_VALIDATION_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Classification Validation Datasets Path')
                os.makedirs(folder)

    @classmethod
    def check_object_detection_inference_datasets_path(cls):
        Logger.info('Check Object Detection Inference Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(OBJECT_DETECTION_INFERENCE_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Object Detection Inference Datasets Path')
                os.makedirs(folder)

    @classmethod
    def check_classification_inference_datasets_path(cls):
        Logger.info('Check Classification Inference Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(CLASSIFICATION_INFERENCE_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Classification Inference Datasets Path')
                os.makedirs(folder)

    @classmethod
    def check_object_detection_underkill_datasets_path(cls):
        Logger.info('Check Object Detection Underkill Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(OBJECT_DETECTION_UNDERKILL_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Object Detection Underkill Datasets Path')
                os.makedirs(folder)

    @classmethod
    def check_classification_underkill_datasets_path(cls):
        Logger.info('Check Classification Underkill Datasets Path')
        for project in PROJECTS:
            folder = os.path.join(CLASSIFICATION_UNDERKILL_DATASETS_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Classification Underkill Datasets Path')
                os.makedirs(folder)

    @classmethod
    def check_yolo_train_yamls_path(cls):
        Logger.info('Check YOLO Train YAMLs Path')
        if not os.path.exists(YOLO_TRAIN_DATA_YAML_DIR):
            Logger.info(f'Create {YOLO_TRAIN_DATA_YAML_DIR}')
            os.makedirs(YOLO_TRAIN_DATA_YAML_DIR)
        if not os.path.exists(YOLO_TRAIN_MODELS_YAML_DIR):
            Logger.info(f'Create {YOLO_TRAIN_MODELS_YAML_DIR}')
            os.makedirs(YOLO_TRAIN_MODELS_YAML_DIR)
        if not os.path.exists(YOLO_TRAIN_HYPS_YAML_DIR):
            Logger.info(f'Create {YOLO_TRAIN_HYPS_YAML_DIR}')
            os.makedirs(YOLO_TRAIN_HYPS_YAML_DIR)


class CheckModelEnvironment:
    @classmethod
    def check_gan_inference_models_dir(cls):
        Logger.info('Check Gan Inference Models Dir')
        for project in PROJECTS:
            folder = os.path.join(GAN_INFERENCE_MODEL_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Gan Inference Models Dir')
                os.makedirs(folder)

    @classmethod
    def check_mobilenet_train_models_dir(cls):
        Logger.info('Check MobileNet Train Models Dir')
        for project in PROJECTS:
            folder = os.path.join(MOBILENET_TRAIN_MODEL_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} MobileNet Train Models Dir')
                os.makedirs(folder)

    @classmethod
    def check_yolo_inference_models_dir(cls):
        Logger.info('Check YOLO Inference Models Dir')
        for project in PROJECTS:
            folder = os.path.join(YOLO_INFERENCE_MODEL_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} YOLO Inference Models Dir')
                os.makedirs(folder)

    @classmethod
    def check_yolo_train_models_dir(cls):
        Logger.info('Check YOLO Train Models Dir')
        for project in PROJECTS:
            folder = os.path.join(YOLO_TRAIN_MODEL_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} YOLO Train Models Dir')
                os.makedirs(folder)

    @classmethod
    def check_classification_inference_models_dir(cls):
        Logger.info('Check Classification Inference Models Dir')
        for project in PROJECTS:
            folder = os.path.join(YOLO_TRAIN_MODEL_DIR, project)
            if not os.path.exists(folder):
                Logger.info(f'Create {project} Classification Inference Models Dir')
                os.makedirs(folder)


class ClearLocalDataset:
    @classmethod
    def clear_object_detection_local_dataset(cls, project, task_name):
        if os.path.exists(os.path.join(OBJECT_DETECTION_TRAIN_DATASETS_DIR, project, task_name)):
            shutil.rmtree(os.path.join(OBJECT_DETECTION_TRAIN_DATASETS_DIR, project, task_name), ignore_errors=True)
        if os.path.exists(os.path.join(CLASSIFICATION_TRAIN_DATASETS_DIR, project, task_name)):
            shutil.rmtree(os.path.join(CLASSIFICATION_TRAIN_DATASETS_DIR, project, task_name), ignore_errors=True)
        if os.path.exists(os.path.join(MOBILENET_TRAIN_MODEL_DIR, project, task_name)):
            shutil.rmtree(os.path.join(MOBILENET_TRAIN_MODEL_DIR, project, task_name), ignore_errors=True)

    @classmethod
    def clear_classification_local_dataset(cls, project, task_name):
        if os.path.exists(os.path.join(CLASSIFICATION_TRAIN_DATASETS_DIR, project, task_name)):
            shutil.rmtree(os.path.join(CLASSIFICATION_TRAIN_DATASETS_DIR, project, task_name), ignore_errors=True)
        if os.path.exists(os.path.join(MOBILENET_TRAIN_MODEL_DIR, project, task_name)):
            shutil.rmtree(os.path.join(MOBILENET_TRAIN_MODEL_DIR, project, task_name), ignore_errors=True)
        