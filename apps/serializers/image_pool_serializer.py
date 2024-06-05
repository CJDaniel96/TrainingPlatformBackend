from flask_restx import fields


image_pool_serializer = {
    'line': fields.String,
    'ip': fields.String,
    'prefix': fields.String
}