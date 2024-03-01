from flask_restx import fields


update_new_info_serializer = {
    'status': fields.String
}

get_training_info_val_status_serializer = {
    'val_status': fields.String
}

update_new_ai_model_info_serializer = {
    'model_id': fields.Integer
}
