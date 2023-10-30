import os
import numpy as np
from app.config import EFFICIENTNETV2_EMBEDDING, MODEL_DIRS, MOBILENETV2, YOLOV5
from app.mobilenet_v2.train import run as mobilenet_run
from app.yolov5.train import run as yolo_run
from app.metric_learning.train import run as metric_learning_run


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class YOLOTrain:
    def __init__(self) -> None:
        pass

    @classmethod
    def train_model(cls, project, task_name, data, cfg, hyp=os.path.join(YOLOV5['YOLOV5_DIR'], 'data', 'hyps', 'hyp.scratch-low.yaml')):
        save_result_dir = os.path.join(MODEL_DIRS['YOLO_TRAIN_MODEL_DIR'], project)
        yolo_run(weights=YOLOV5['YOLOV5S_WEIGHT'], batch_size=YOLOV5['YOLOV5_BATCH_SIZE'], epochs=YOLOV5['YOLOV5_EPOCHS'], 
                 data=data, cfg=cfg, hyp=hyp, seed=YOLOV5['YOLOV5_SEED'], project=save_result_dir, name=task_name)


class MobileNetV2Train:
    def __init__(self) -> None:
        pass

    @classmethod
    def train_model(cls, project, task_name, data):
        save_result_dir = os.path.join(MODEL_DIRS['MOBILENET_TRAIN_MODEL_DIR'], project, task_name)
        mobilenet_run(data_dir=data, batch_size=MOBILENETV2['MOBILENETV2_BATCH_SIZE'], num_epochs=MOBILENETV2['MOBILENETV2_EPOCHS'], 
                      save_dir=save_result_dir)


class MetricLearningTrain:
    def __init__(self) -> None:
        pass

    @classmethod
    def train_model(cls, project, task_name, data):
        save_result_dir = os.path.join(MODEL_DIRS['METRIC_LEANRING_TRAIN_MODEL_DIR'], project, task_name)
        metric_learning_run(data, epochs=EFFICIENTNETV2_EMBEDDING['EPOCHS'], batch_size=EFFICIENTNETV2_EMBEDDING['BATCH_SIZE'], 
                            num_classes=EFFICIENTNETV2_EMBEDDING['NUM_CLASSES'], embedding_size=EFFICIENTNETV2_EMBEDDING['EMBEDDING_SIZE'], 
                            lr=EFFICIENTNETV2_EMBEDDING['LEARNING_RATE'], loss_lr=EFFICIENTNETV2_EMBEDDING['LOSS_LEARNING_RATE'], 
                            seed=EFFICIENTNETV2_EMBEDDING['SEED'], save_dir=save_result_dir)