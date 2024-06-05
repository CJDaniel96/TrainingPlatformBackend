from flask_restx import fields


yolov5_inference_doc = {
    'post': {
        'params': None,
        'payload': {
            'model': fields.String,
            'images': fields.List(fields.String)
        }
    }
}

yolov5_train_doc = {
    'post': {
        'params': None,
        'payload': {
            'weights': fields.String, 
            'data': fields.String, 
            'cfg': fields.String, 
            'hyp': fields.String, 
            'batch_size': fields.Integer, 
            'epochs': fields.Integer, 
            'seed': fields.Integer,
            'project': fields.String, 
            'name': fields.String
        }
    }
}