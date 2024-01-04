from flask_restx import fields
from apps.serializers import Eval


labels_serializer = {
    'labels': Eval(fields.String)
}

ok_label_serializer = {
    'ok_category': Eval(fields.String)
}