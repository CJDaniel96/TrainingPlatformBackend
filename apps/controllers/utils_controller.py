from flask import jsonify, make_response, request
from flask_restx import Resource
from apps.services.utils_service import *


class DateTimeBarcodeController(Resource):
    def get(self):
        data = DateTimeService.get_datetime_barcode()
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    

class UnzipDataController(Resource):
    def post(self):
        json_data = request.get_json()
        zip_paths = json_data['paths']
        name = json_data['name']
        
        data = UnzipDataService.unzip_data(zip_paths, name)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    

class OutputXMLController(Resource):
    def post(self):
        json_data = request.get_json()
        results = json_data['results']
        
        data = XMLService.output_xmls(results)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    

class XMLToYOLOFormatController(Resource):
    def post(self):
        json_data = request.get_json()
        classes = json_data['classes']
        xml_dir = json_data['annotations_path']
        
        data = XMLtoYOLOService.xml_to_yolo(classes, xml_dir)
        
        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
        