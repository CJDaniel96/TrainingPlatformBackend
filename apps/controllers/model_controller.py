from flask import jsonify, make_response, request
from flask_restx import Resource
from apps.services.model_service import *


class Yolov5InferenceController(Resource):
    def post(self):
        json_data = request.get_json()
        model = json_data['model']
        images = json_data['images']
        
        data = InferenceService.get_yolov5_inference(model, images)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    
    
class Yolov5TrainingController(Resource):
    def post(self):
        json_data = request.get_json()
        
        data = TrainingService.train(**json_data)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)