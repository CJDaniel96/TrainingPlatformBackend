from flask_restx import fields


category_mapping_labels_controller_doc = {
    'get': {
        'params': {
            'line': 'Line',
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