from flask_restx import fields


image_serializer = {
    'line_id': fields.String,
    'image_path': fields.String
}

image_list_serializer = {
    'images': fields.List(fields.Nested(image_serializer))
}