import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from app.config import MOBILENET_TRAIN_MODEL_DIR, MOBILENETV2_BATCH_SIZE, MOBILENETV2_EPOCHS, YOLOV5_BATCH_SIZE, YOLOV5_DIR, YOLOV5_EPOCHS, YOLO_TRAIN_MODEL_DIR, YOLOV5_SEED, YOLOV5S_WEIGHT
from app.mobilenet_v2.train import run as mobilenet_run
from app.yolov5.train import run as yolo_run


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
    def train_model(cls, project, task_name, data, cfg, hyp=os.path.join(YOLOV5_DIR, 'data', 'hyps', 'hyp.scratch-low.yaml')):
        save_result_dir = os.path.join(YOLO_TRAIN_MODEL_DIR, project)
        yolo_run(weights=YOLOV5S_WEIGHT, atch_size=YOLOV5_BATCH_SIZE, epochs=YOLOV5_EPOCHS, data=data, cfg=cfg, hyp=hyp, seed=YOLOV5_SEED, project=save_result_dir, name=task_name)


class MobileNetV2Train:
    def __init__(self) -> None:
        pass

    @classmethod
    def train_model(cls, project, task_name, data):
        save_result_dir = os.path.join(MOBILENET_TRAIN_MODEL_DIR, project, task_name)
        mobilenet_run(data_dir=data, batch_size=MOBILENETV2_BATCH_SIZE, num_epochs=MOBILENETV2_EPOCHS, save_dir=save_result_dir)