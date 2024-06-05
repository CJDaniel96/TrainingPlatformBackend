from flask_restx import fields
from apps.serializers.amr_raw_data_serializer import *


# Documents Settings

image_controller_doc = {
    'get': {
        'params': {
            'site': 'site',
            'line': 'line',
            'group_type': 'group_type',
            'start_date': 'start_date',
            'end_date': 'end_date'
        },
        'payload': None
    },
    'post': {
        'params': {
            'site': 'site',
            'line': 'line',
            'group_type': 'group_type',
            'start_date': 'start_date',
            'end_date': 'end_date'
        },
        'payload': None
    }
}

uuids_image_controller_doc = {
    'post': {
        'params': None,
        'payload': {
            'image_mode': fields.String,
            'uuids': fields.List(fields.String)
        }
    }
}

