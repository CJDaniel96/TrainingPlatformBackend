import os
from app.config import YOLOV5_BATCH_SIZE, YOLOV5_EPOCHS, YOLO_TRAIN_MODEL_DIR
from app.yolov5.train import run


class YOLOTrain:
    def __init__(self) -> None:
        pass

    @classmethod
    def train_model(cls, project, task_name, data, cfg, hyp=os.path.join(YOLO_TRAIN_MODEL_DIR, 'data', 'hyps', 'hyp.scratch-low.yaml')):
        save_result_dir = os.path.join(YOLO_TRAIN_MODEL_DIR, project)
        run(batch_size=YOLOV5_BATCH_SIZE, epochs=YOLOV5_EPOCHS, data=data, cfg=cfg, hyp=hyp, project=save_result_dir, name=task_name)