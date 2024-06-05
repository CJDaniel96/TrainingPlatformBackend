from flask_restx import marshal_with
from apps.serializers.amr_raw_data_serializer import *
from apps.databases.amr import AmrRawData
from apps.databases.ai import UploadData
from config import *


class ImageService:    
    def _basic_image_query(self, site, line, group_type, start_date, end_date):
        return AmrRawData.query.filter(
            AmrRawData.site == site,
            AmrRawData.line_id == line,
            AmrRawData.group_type == group_type,
            AmrRawData.create_time.between(start_date, end_date),
            AmrRawData.is_covered == True,
            AmrRawData.ai_result == '0'
        )
        
    @classmethod
    @marshal_with(image_list_serializer)
    def get_image(cls, site, line, group_type, start_date, end_date):
        query = cls()._basic_image_query(site, line, group_type, start_date, end_date)
        data = query.all()
        
        return {'images': data}
    
    @classmethod
    @marshal_with(image_list_serializer)
    def get_image_by_assign_light_type(cls, site, line, group_type, start_date, end_date):
        query = cls()._basic_image_query(site, line, group_type, start_date, end_date)
        data = query.filter(
            AmrRawData.image_name.like(f'%{IMAGES_ASSIGN_LIGHT_TYPE[site][group_type]}%')
        ).all()

        return {'images': data}
    
    @classmethod
    @marshal_with(image_list_serializer)
    def get_image_by_uuid(cls, image_mode, uuids):
        if image_mode == 'upload':
            query = AmrRawData.query.filter(AmrRawData.uuid.in_(uuids))
        elif image_mode == 'upload_image':
            query = UploadData.query.filter(UploadData.uuid.in_(uuids))
        data = query.all()
        
        return {'images': data}