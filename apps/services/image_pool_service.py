from flask_restx import marshal_with
from apps.serializers.image_pool_serializer import *
from apps.databases.ai import ImagePool


class ImagePoolInfoService:
    @classmethod
    @marshal_with(image_pool_serializer)
    def get_image_pool_by_line(cls, line):
        if line:
            return ImagePool.query.filter(ImagePool.line == line).first()
        else:
            return ImagePool.query.all()