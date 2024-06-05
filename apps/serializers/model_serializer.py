from flask_restx import fields


yolo_inference_result_serializer = {
    'results': fields.List(fields.Nested({
        'image_path': fields.String,
        'image_size': fields.String,
        'defect_name': fields.List(fields.String),
        'defect_position': fields.List(fields.List(fields.Float)),
        'confidence': fields.List(fields.Float)
    }))
}

yolo_train_result_serializer = {
    'best_model_path': fields.String,
    'last_model_path': fields.String
}