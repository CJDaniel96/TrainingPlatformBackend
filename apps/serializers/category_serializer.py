from flask_restx import fields
from apps.serializers import Eval


labels_serializer = {
    'labels': Eval(fields.String)
}

ok_label_serializer = {
    'ok_category': Eval(fields.String)
}

upload_crop_category_record_serializer = {
    'crop_img_id': fields.String
}

critical_ng_serializer = {
    'img_id': fields.String, 
    'image_path': fields.String
}

critical_ng_list_serializer = {
    'critical_ngs': fields.List(fields.Nested(critical_ng_serializer))
}