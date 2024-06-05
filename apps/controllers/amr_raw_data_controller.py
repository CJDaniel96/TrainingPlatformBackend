from flask import jsonify, request, make_response
from flask_restx import Resource
from apps.services.amr_raw_data_service import *


class BaseImageController(Resource):
    def _get_args(self):
        return {
            'site': request.args.get('site'),
            'line': request.args.get('line'),
            'group_type': request.args.get('group_type'),
            'start_date': request.args.get('start_date'),
            'end_date': request.args.get('end_date')
        }
        
    def _get_data(self, args):
        raise NotImplementedError("Subclasses must implement _get_data method.")
    
    def _get_data_from_json(self, json_data):
        raise NotImplementedError("Subclasses must implement _get_data_from_json method.")
    
    def _make_response(self, data):
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    
    def get(self):
        args = self._get_args()
        data = self._get_data(args)
        
        return self._make_response(data)
    
    def post(self):
        json_data = request.get_json()
        data = self._get_data_from_json(json_data)
        
        return self._make_response(data)


class AssignLightImageController(BaseImageController):
    def _get_data(self, args):
        return ImageService.get_image_by_assign_light_type(**args)


class ImageController(BaseImageController):
    def _get_data(self, args):
        return ImageService.get_image(**args)

    
class UuidsImageController(BaseImageController):
    def _get_data_from_json(self, json_data):
        image_mode = json_data['image_mode']
        uuids = json_data['uuids']
        
        return ImageService.get_image_by_uuid(image_mode, uuids)