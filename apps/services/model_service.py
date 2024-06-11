from flask_restx import marshal_with
from apps.serializers.model_serializer import *
from pathlib import Path
from algorithm.yolov5.train import run as yolo_run
from config import *
import torch
import cv2


class InferenceService:
    @classmethod
    @marshal_with(yolo_inference_result_serializer)
    def get_yolov5_inference(cls, model_path, images):
        results = []
        if Path(model_path).exists() and Path(model_path).suffix == '.pt':
            model = torch.hub.load('algorithm/yolov5', 'custom', path=model_path, source='local')
            model.conf = 0.25
            model.cuda()
            
            for image_path in images:
                result = model(image_path).pandas().xyxy[0]
                image = cv2.imread(image_path)
                image_size = image.shape
                
                results.append({
                    'image_path': image_path,
                    'image_size': image_size,
                    'defect_name': list(result['name'].values),
                    'defect_position': list(result[['xmin', 'ymin', 'xmax', 'ymax']].values),
                    'confidence': result['confidence']
                })
                
        return {'results': results}
    

class TrainingService:
    @classmethod
    @marshal_with(yolo_train_result_serializer)
    def train(cls, **kwargs):
        if kwargs['weights']:
            weights = kwargs['weights']
        else:
            weights = str(Path(YOLOV5_DIR, 'yolov5s.pt'))

        if kwargs['data']:
            data = kwargs['data']
        if kwargs['cfg']:
            cfg = kwargs['cfg']
        if kwargs['hyp']:
            hyp = kwargs['hyp']
        if kwargs['batch_size']:
            batch_size = kwargs['batch_size']
        if kwargs['epochs']:
            epochs = kwargs['epochs']
        if kwargs['project']:
            project = kwargs['project']
        if kwargs['name']:
            name = kwargs['name']
        if kwargs['seed']:
            seed = kwargs['seed']
        
        if OBJECT_DETECTION_ALGORITHM == 'yolov5':
            yolo_run(weights=weights, data=data, cfg=cfg, hyp=hyp, batch_size=batch_size, epochs=epochs, project=project, name=name, seed=seed, exist_ok=True)
        elif OBJECT_DETECTION_ALGORITHM == 'ultralytics_yolov5':
            from ultralytics import YOLO
            model = YOLO('yolov5s.yaml')
            model.train(data=data, hyp=hyp, epochs=epochs, batch_size=batch_size, project=project, name=name, exist_ok=True)
        
        return {
            'best_model_path': str(Path(project, name, 'weights', 'best.pt')),
            'last_model_path': str(Path(project, name, 'weights', 'last.pt'))
        }