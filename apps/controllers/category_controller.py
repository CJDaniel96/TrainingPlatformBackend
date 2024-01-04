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