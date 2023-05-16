from app.services.database_service import IRIRecordService, ImageDataService, ImagePoolService, URDRecordService, UploadData, UploadImageDataService
from app.services.datasets_service import OriginDataProcessing
from app.services.image_pool_service import ImagePool


class InitialController:
    def __init__(self, record) -> None:
        self.site = record.site
        self.line = record.line
        self.group_type = record.group_type
        self.start_date = record.start_date
        self.end_date = record.end_date
        self.smart_filter = record.smart_filter
        self.uuids = record.images
        self.id = record.id
        self.tablename = record.__tablename__

    def get_query_data(self):
        return ImageDataService.get_images(self.site, self.line, self.group_type, self.start_date, self.end_date, self.smart_filter)

    def get_upload_data(self):
        return UploadData.get_images(self.uuids)

    def get_upload_image_data(self):
        return UploadImageDataService.get_images(self.uuids)
    
    def update_record_line(self, lines):
        if self.tablename == 'iri_record':
            IRIRecordService.update_line(id, lines)
        elif self.tablename == 'urd_record':
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
    
