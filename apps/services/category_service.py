from flask_restx import marshal_with
from apps.serializers.category_serializer import *
from apps.databases.ai import CategoryMapping, CriticalNg, CropCategorizingRecord
from config import *
from apps.databases import db


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
    

class UploadCropCategoryRecordService:
    @classmethod
    @marshal_with(upload_crop_category_record_serializer)
    def upload_crop_category_record(cls, finetune_id, image_id, image_wide, image_hight, finetune_type):
        row = CropCategorizingRecord(
            finetune_id=finetune_id, 
            img_id=image_id, 
            crop_name='ORG',
            x_min=0,
            y_min=0,
            x_max=image_wide,
            y_max=image_hight,
            categorizing_code='OK',
            finetune_type=finetune_type,
            critical_ng=True
        )
        db.session.add(row)
        db.session.commit()
        
        return {'crop_img_id': finetune_type + '@' + str(finetune_id) + '@' + image_id + '@' + 'ORG'}
    
    
class CriticalNGService:
    @classmethod
    @marshal_with(critical_ng_list_serializer)
    def get_critical_ng(cls, line_id, group_type, critical_ng_images):
        data = CriticalNg.query.filter(
            CriticalNg.line_id == line_id,
            CriticalNg.group_type == group_type,
            CriticalNg.image_path.in_(critical_ng_images)
        ).all()
        
        return {'critical_ngs': data}