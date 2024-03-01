from flask import jsonify, request, make_response
from flask_restx import Resource
from apps.services.category_service import *
from apps.services.information_service import *


class BaseTrainingInfoController(Resource):
    def _get_data(self, params):
        raise NotImplementedError("Subclasses must implement _get_data method.")
    
    def _post_data(self, json_data):
        raise NotImplementedError("Subclasses must implement _put_data method.")
    
    def _make_response(self, data):
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
        
    def post(self):
        json_data = request.get_json()
        
        data = self._post_data(json_data)
        
        return self._make_response(data)
    
    def get(self):
        params = {
            'task_id': request.args.get('task_id')
        }

        data = self._get_data(params)
        
        return self._make_response(data)


class ODTrainingInfoController(BaseTrainingInfoController):
    def _post_data(self, json_data):
        return ODTrainingInfoService.update_new_info(**json_data)
    
    def _get_data(self, params):
        return ODTrainingInfoService.get_training_info_val_status(**params)
    
    
class CLSTrainingInfoController(BaseTrainingInfoController):
    def _post_data(self, json_data):
        return CLSTrainingInfoService.update_new_info(**json_data)
    
    def _get_data(self, params):
        return CLSTrainingInfoService.get_training_info_val_status(**params)
    

class AIModelInformationController(Resource):
    def post(self):
        json_data = request.get_json()
        
        data = AIModelInformationService.update_ai_model_info(**json_data)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    
    
class AIModelPerformanceController(Resource):
    def post(self):
        json_data = request.get_json()
        
        data = AIModelPerformanceSerivce.update_ai_model_perf(**json_data)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)