from app.services.check_environment_service import CheckDatasetsEnvironment, ClearLocalDataset
from app.services.logging_service import Logger


class CheckEnvironmentController:
    @classmethod
    def check_datasets_environment(cls):
        CheckDatasetsEnvironment.check_origin_datasets_path()
        CheckDatasetsEnvironment.check_object_detection_basicline_datasets_path()
        CheckDatasetsEnvironment.check_classification_basicline_datasets_path()
        CheckDatasetsEnvironment.check_object_detection_train_datasets_path()
        CheckDatasetsEnvironment.check_classification_train_datasets_path()
        CheckDatasetsEnvironment.check_object_detection_validation_datasets_path()
        CheckDatasetsEnvironment.check_classification_validation_datasets_path()
        CheckDatasetsEnvironment.check_yolo_train_yamls_path()

    @classmethod
    def clear_local_data(cls, status, project, task_name):
        if status == 'OD_Initialized':
            Logger.warn('Clear object detecion local dataset!')
            ClearLocalDataset.clear_object_detection_local_dataset(project, task_name)
        elif status == 'CLS_Initialized':
            Logger.warn('Clear classification local dataset!')
            ClearLocalDataset.clear_classification_local_dataset(project, task_name)