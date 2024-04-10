from flask import jsonify, request, make_response
from flask_restx import Resource
from apps.services.category_service import *


class CategoryMappingLabelsController(Resource):
    def post(self):
        json_data = request.get_json()
        
        data = CategoryMappingLabelsService.get_labels(**json_data)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    

class CategoryMappingOKLabelsController(Resource):
    def post(self):
        json_data = request.get_json()
        
        data = CategoryMappingLabelsService.get_ok_labels(**json_data)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    
    
class CropCategoryRecordController(Resource):
    def put(self):
        json_data = request.get_json()
        
        data = UploadCropCategoryRecordService.upload_crop_category_record(**json_data)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    
    
class CriticalNGController(Resource):
    def post(self):
        json_data = request.get_json()
        
        data = CriticalNGService.get_critical_ng(**json_data)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)