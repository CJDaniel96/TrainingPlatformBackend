from flask import jsonify, request, make_response
from flask_restx import Resource
from apps.services.category_service import *


class CategoryMappingLabelsController(Resource):
    def get(self):
        project = request.args.get('project')
        line = request.args.get('line')
        group_type = request.args.get('group_type')
        
        data = CategoryMappingLabelsService.get_labels(line, group_type, project)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    

class CategoryMappingOKLabelsController(Resource):
    def get(self):
        project = request.args.get('project')
        line = request.args.get('line')
        group_type = request.args.get('group_type')
        
        data = CategoryMappingLabelsService.get_ok_labels(line, group_type, project)
        
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