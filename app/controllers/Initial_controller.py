from app.services.database_service import IRIRecordService, ImageDataService, ImagePoolService, URDRecordService, UploadData, UploadImageDataService
from app.services.datasets_service import OriginDataProcessing
from app.services.image_pool_service import ImagePool


class InitialController:
    def __init__(self) -> None: 
        pass

    @classmethod
    def get_query_data(cls, site, line, group_type, start_date, end_date, smart_filter):
        return ImageDataService.get_images(site, line, group_type, start_date, end_date, smart_filter)
    
    @classmethod
    def get_upload_data(cls, uuids):
        return UploadData.get_images(uuids)
    
    @classmethod
    def get_upload_image_data(cls, uuids):
        return UploadImageDataService.get_images(uuids)
    
    @classmethod
    def update_record_line(cls, tablename, lines):
        if tablename == 'iri_record':
            IRIRecordService.update_line(id, lines)
        elif tablename == 'urd_record':
            URDRecordService.update_line(id, lines)

    @classmethod
    def download_images(cls, images: dict):
        image_pools = ImagePoolService.get_image_pool()
        for image_pool in image_pools:
            if image_pool.line in images.keys():
                image_list = images[image_pool.line]
                ImagePool.download(image_pool, image_list)

    @classmethod
    def get_serial_number(cls):
        return OriginDataProcessing.get_serial_number()

    @classmethod
    def arrange_origin_images(cls, serial_number):
        org_data_folder = OriginDataProcessing.get_origin_image_folder(serial_number)
        org_image_folder = OriginDataProcessing.unzip_origin_data(org_data_folder)

        return org_image_folder
    
