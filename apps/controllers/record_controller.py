from flask import jsonify, make_response, request
from flask_restx import Resource
from apps.services.record_service import *


class RecordStatusController(Resource):
    service:BaseRecordService = None
    fields_to_update = ['site', 'line', 'status', 'task_id', 'task']
    
    def get(self):
        record_id = request.args.get('id')
        
        data = self.service.get_status(record_id)

        return make_response(jsonify({'data': data, 'message': 'Success'}), 200)
    
    def put(self):
        json_data = request.get_json()
        
        if not json_data:
            return make_response(jsonify({'message': 'No Input'}), 400)
        
        for field in self.fields_to_update:
            if field in json_data and json_data[field]:
                value = json_data[field]
                if field == 'line' and isinstance(value, list):
                    value = repr(value)
                self.service.update(json_data['id'], field, value)
    
        return make_response(jsonify({'message': f'The {self.service.table.__tablename__} {json_data["id"]} task have been update'}), 200)


class IRIRecordStatusController(RecordStatusController):
    service = IRIRecordService()
    

class URDRecordStatusController(RecordStatusController):
    service = URDRecordService()