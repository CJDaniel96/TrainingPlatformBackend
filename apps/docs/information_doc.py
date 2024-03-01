from flask_restx import fields


image_pool_info_doc = {
    'get': {
        'params': {
            'line': 'Image Pool Line'    
        },
        'payload': None
    }
}

training_info_doc = {
    'get': {
        'params': {
            'task_id': 'Task ID'    
        },
        'payload': None
    },
    'post': {
        'params': None,
        'payload': {
            'task_id': fields.String, 
            'comp_type': fields.String, 
            'validate_result': fields.Boolean
        }
    }
}

ai_model_info_doc = {
    'post': {
        'params': None,
        'payload': {
            'model_type': fields.String,
            'model_path': fields.String,
            'ip_address': fields.String,
            'verified_status': fields.String,
            'finetune_id': fields.Integer,
            'finetune_type': fields.String
        }
    }
}

ai_model_performance_doc = {
    'post': {
        'params': None,
        'payload': {
            'model_id': fields.Integer,
            'metrics_result': {
                'LOSS': fields.Float,
                'ACCURACY': fields.Float,
                'FALSE_NEGATIVE_NUM': fields.Integer,
                'FALSE_POSITIVE_NUM': fields.Integer,
                'FINE_TUNE_CONFIRM_TASK_ID': fields.String
            },
            'false_negative_imgs': {
                'NUMBER_OF_IMG': fields.Integer,
                'CROP_IMG_ID_LIST': fields.List(fields.String)
            },
            'false_positive_imgs': {
                'NUMBER_OF_IMG': fields.Integer
            }
        }
    }
}