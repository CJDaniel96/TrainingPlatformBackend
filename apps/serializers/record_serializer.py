from flask_restx import fields
from apps.serializers import Eval


record_serializer = {
    'id': fields.Integer, 
    'project': fields.String, 
    'task': fields.String,
    'status': fields.String,
    'site': fields.String,
    'line': Eval(fields.String(atrrribute='line')),
    'group_type': fields.String,
    'start_date': fields.Date,
    'end_date': fields.Date,
    'labeling': fields.Boolean,
    'od_training': fields.String,
    'cls_training': fields.String,
    'update_time': fields.DateTime,
    'create_time': fields.DateTime,
    'task_id': fields.Integer, 
    'project_id': fields.Integer, 
    'smart_filter': fields.Boolean, 
    'smart_filter_value': fields.Integer,
    'images': Eval(fields.String(atrrribute='images')),
    'image_mode': fields.String,
    '__tablename__': fields.String
}
