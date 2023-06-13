from app.services.database_service import CategoryMappingService, CropCategorizingRecordService, ImageDataService
from app.services.datasets_service import CategorizeDataProcessing
from app.services.inference_service import YOLOInference
from app.services.logging_service import Logger


class CategorizeController:
    @classmethod
    def categorizing(cls):...

    @classmethod
    def upload_categorizing_record(cls, record_id, site, project, group_type, train_data_folder, finetune_type='URD'):
        Logger.info('Upload Database Record Crop Categorizing')
        images = CategorizeDataProcessing.get_images(train_data_folder)
        for image_path in images:
            image_uuid = ImageDataService.get_image_uuid(image_path, group_type)
            image = YOLOInference.read_image(image_path)
            image_size = YOLOInference.get_image_size(image)

            class_dict = CategoryMappingService.get_class_dict(site, group_type, project)
            ok_category = CategoryMappingService.get_ok_category(site, group_type, project)
            txt_file = CategorizeDataProcessing.get_image_txt_file(image_path)
            categorizing_code = CategorizeDataProcessing.check_image_result(ok_category, class_dict, txt_file)

            _ = CropCategorizingRecordService.update_underkill_image(record_id, image_uuid, image_size[0], image_size[1], finetune_type, categorizing_code)

        return 