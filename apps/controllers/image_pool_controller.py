from flask import jsonify, make_response, request
from flask_restx import Resource
from apps.services.image_pool_service import ImagePoolInfoService


class ImagePoolInfoController(Resource):
    def get(self):
        line = request.args.get('line')
        
        data = ImagePoolInfoService.get_image_pool_by_line(line)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)