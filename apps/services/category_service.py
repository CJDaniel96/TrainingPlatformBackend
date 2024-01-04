from flask_restx import marshal_with
from apps.serializers.category_serializer import *
from apps.databases.ai import CategoryMapping
from config import *


class CategoryMappingLabelsService:
    @classmethod
    @marshal_with(labels_serializer)
    def get_labels(cls, line, group_type, project):
        data = CategoryMapping.query.filter(
            CategoryMapping.line == line,
            CategoryMapping.group_type == group_type,
            CategoryMapping.project == project
        ).first()
        
        return data
    
    @classmethod
    @marshal_with(ok_label_serializer)
    def get_ok_labels(cls, line, group_type, project):
        data = CategoryMapping.query.filter(
            CategoryMapping.line == line,
            CategoryMapping.group_type == group_type,
            CategoryMapping.project == project
        ).first()
        
        return data