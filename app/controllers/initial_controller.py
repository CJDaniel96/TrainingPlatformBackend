from app.services.database_service import IRIRecordService, ImageDataService, ImagePoolService, URDRecordService, UploadDataService, UploadImageDataService
from app.services.datasets_service import OriginDataProcessing
from app.services.image_pool_service import ImagePool


class InitialController:
    def __init__(self) -> None: 
        pass

    @classmethod
    def get_query_data(cls, site, line, group_type, start_date, end_date, smart_filter):
        if ImageDataService.check_assign_image_light_type(site, group_type):
            return ImageDataService.get_image_by_assign_light_type(site, line, group_type, start_date, end_date, smart_filter)
        else:
            return ImageDataService.get_images(site, line, group_type, start_date, end_date, smart_filter)
    
    @classmethod
    def get_upload_data(cls, uuids):
        return UploadDataService.get_images(uuids)
    
    @classmethod
    def get_upload_image_data(cls, uuids):
        return UploadImageDataService.get_images(uuids)
    
    @classmethod
    def get_image_lines(cls, images):
        return [line for line in images]
    
    @classmethod
    def update_record_line(cls, tablename, id, lines):
        if tablename == 'iri_record':
            IRIRecordService.update_line(id, lines)
        elif tablename == 'urd_record':
            URDRecordService.update_line(id, lines)

    @classmethod
    def update_record_site(cls, tablename, id):
        if tablename == 'iri_record':
            IRIRecordService.update_site(id)
        elif tablename == 'urd_record':
            URDRecordService.update_site(id)

    @classmethod
    def download_images(cls, images: dict, image_mode):
        image_pools = ImagePoolService.get_image_pool()
        for image_pool in image_pools:
            if image_mode == image_pool.line:
                image_list = list(images.values())[0]
                ImagePool.download(image_pool, image_list)
            elif image_pool.line in images.keys():
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
    
