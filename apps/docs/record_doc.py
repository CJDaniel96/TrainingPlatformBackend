from flask_restx import fields
from apps.serializers.record_serializer import *


iri_record_status_doc = {
    'get': {
        'params': {
            'id': 'Record ID'
        },
        'payload': None
    },
    'put': {
        'params': None,
        'payload': {
            'id': fields.String,
            'status': fields.String(default=None),
            'site': fields.String(default=None),
            'line': fields.String(default=None),
            'od_training': fields.String(default=None),
            'cls_training': fields.String(default=None),
            'task_id': fields.String(default=None),
        }
    }
}

urd_record_status_doc = {
    'get': {
        'params': {
            'id': 'Record ID'
        },
        'payload': None
    },
    'put': {
        'params': None,
        'payload': {
            'id': fields.String,
            'status': fields.String(default=None),
            'site': fields.String(default=None),
            'line': fields.String(default=None),
            'od_training': fields.String(default=None),
            'cls_training': fields.String(default=None),
            'task_id': fields.String(default=None),
        }
    }
}