from app.services.database_service import CriticalNG
from app.services.inference_service import YOLOInference
from app.services.manage_service import ManageService


class ManageController:
    def __init__(self) -> None:
        pass

    @classmethod
    def insert_validated_images_to_db(cls, project, group_type):
        validated_images = YOLOInference.get_validation_images(project)
        for image in validated_images:
            ManageService.check_image_type(image)
            ManageService.parse_chinses_to_english(image)
            CriticalNG.insert_new_images(group_type, image)