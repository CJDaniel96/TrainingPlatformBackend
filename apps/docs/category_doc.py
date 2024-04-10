from flask_restx import fields


category_mapping_labels_controller_doc = {
    'get': {
        'params': {
            'line': fields.List(fields.String(attribute='Line')),
            'group_type': 'Group Type',
            'project': 'Project'
        },
        'payload': None
    }
}

category_mapping_ok_labels_controller_doc = {
    'get': {
        'params': {
            'project': 'Project'
        },
        'payload': None
    }
}

category_record_upload_doc = {
    'post': {
        'params': None,
        'payload': {
            'images_path': fields.List(fields.String),
            'group_type': fields.String
        }
    }
}

critical_ng_controller_doc = {
    'post': {
        'params': None,
        'payload': {
            'line_id': fields.String,
            'group_type': fields.String,
            'critical_ng_images': fields.List(fields.String)
        }
    }
}

crop_category_record_controller_doc = {
    'put': {
        'params': None,
        'payload': {
            'finetune_id': fields.String, 
            'image_id': fields.String, 
            'image_wide': fields.Integer, 
            'image_hight': fields.Integer, 
            'finetune_type': fields.String
        }
    }
}