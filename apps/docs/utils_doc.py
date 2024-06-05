from flask_restx import fields


datatime_barcode_doc = {
    'get': {
        'params': None,
        'payload': None
    }
}

unzip_data_doc = {
    'post': {
        'params': None,
        'payload': {
            'paths': fields.List(fields.String),
            'barcode': fields.String
        }
    }
}

output_xmls_doc = {
    'post': {
        'params': None,
        'payload': {
            'results': fields.List(fields.Raw({
                'image_path': '/Image/Path',
                'image_size': '(int, int, int)',
                'defect_name': ['string'],
                'defect_position': [['float', 'float', 'float', 'float']],
                'confidence': ['string']
            }))
        }
    }
}

xml_to_yolo_doc = {
    'post': {
        'params': None,
        'payload': {
            'classes': fields.Wildcard(fields.String),
            'annotations_path': fields.String
        }
    }
}