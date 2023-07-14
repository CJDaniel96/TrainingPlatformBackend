from app.config import MOBILENET_TRAIN_MODEL_DIR, TRAINING_FLOW, VALIDATION_FLOW
from app.services.database_service import AIModelInformationService, AIModelPerformanceService, CriticalNGService, CropCategorizingRecordService, IRIRecordService, TrainingInfoService, URDRecordService
from app.services.datasets_service import ClassificationTrainDataProcessing, UnderkillDataProcessing
from app.services.inference_service import MobileNetGANInference, YOLOInference
from app.services.logging_service import Logger
from app.services.train_service import MobileNetV2Train


class ClassificationController:
    def __init__(self) -> None:
        pass

    @classmethod
    def check_only_train_classification_model(cls, project):
        Logger.info('Check Whether Train Classification Model')
        if 'classification' in TRAINING_FLOW[project]:
            return True
        else:
            return False

    @classmethod
    def check_coutinue_train_classification_model(cls, project):
        Logger.info('Check Whether Train Classification Model')
        if 'object_detection' in TRAINING_FLOW[project] and 'classification' in TRAINING_FLOW[project]:
            return True
        else:
            return False
        
    @classmethod
    def get_train_model_path(project, task_name):
        return MobileNetGANInference.get_train_model_path(project, task_name)
        
    @classmethod
    def get_train_data_folder(cls, project, task_name):
        return ClassificationTrainDataProcessing.get_classification_train_data_folder(project, task_name)
    
    @classmethod
    def get_train_dataset(cls, task_zip_file, train_data_folder):
        if ClassificationTrainDataProcessing.check_zip_file(task_zip_file):
            ClassificationTrainDataProcessing.get_classification_train_data(train_data_folder)
            ClassificationTrainDataProcessing.clear_zip_file(task_zip_file)

        return
    
    @classmethod
    def get_classification_tasks(cls, task_id, group_type, tablename):
        tasks_id = TrainingInfoService.get_classification_tasks_id(task_id, group_type)
        if tablename == 'iri_record':
            return IRIRecordService.get_tasks(tasks_id)
        elif tablename == 'urd_record':
            return URDRecordService.get_tasks(tasks_id)

    @classmethod
    def merge_basicline_dataset(cls, train_data_folder, project):
        Logger.info('Merge Local Basicline Dataset in Train Dataset')
        ClassificationTrainDataProcessing.merge_classification_basicline_data(train_data_folder, project)

    @classmethod
    def train(cls, project, task_name, data):
        Logger.info('MobileNet V2 Training Start...')
        MobileNetV2Train.train_model(project, task_name, data)

    @classmethod
    def validate(cls, project, task_name):
        Logger.info('MobilNet V2 Validating Start...')
        if project in VALIDATION_FLOW['mobilenetv2_fanogan']:
            model_file = MobileNetGANInference.get_train_model_path(project, task_name)
            model = MobileNetGANInference.load_model(model_file)
            generator_model_file = MobileNetGANInference.get_generator_model_path(project)
            discriminator_model_file = MobileNetGANInference.get_discriminator_model_path(project)
            encoder_model_file = MobileNetGANInference.get_encoder_model_path(project)
            generator = MobileNetGANInference.get_generator_model(generator_model_file)
            discriminator = MobileNetGANInference.get_discriminator_model(discriminator_model_file)
            encoder = MobileNetGANInference.get_encoder_model(encoder_model_file)
            criterion = MobileNetGANInference.get_criterion()
            mean, std = MobileNetGANInference.get_mean_std(project, task_name)
            data_transforms = MobileNetGANInference.get_transform(mean, std)
            class_list = MobileNetGANInference.get_classes(project, task_name)

            validation_images = MobileNetGANInference.get_validation_images(project)
            validation_count = MobileNetGANInference.check_validation_count(validation_images)
            underkill_folder = MobileNetGANInference.get_underkill_folder(project, task_name)

            kappa = VALIDATION_FLOW['mobilenetv2_fanogan'][project]['gan_settings']['kappa']
            anormaly_threshold = VALIDATION_FLOW['mobilenetv2_fanogan'][project]['gan_settings']['anormaly_threshold']
            confidence = VALIDATION_FLOW['mobilenetv2_fanogan'][project]['confidence']

            underkill_count = 0
            for validation_image in validation_images:
                answer = MobileNetGANInference.classification_inference(validation_image, model, class_list, data_transforms, confidence)
                if answer and MobileNetGANInference.gan_inference(
                    validation_image, generator, discriminator, encoder, criterion, kappa, anormaly_threshold
                ):
                    underkill_count += 1
                    MobileNetGANInference.output_underkill_image(validation_image, underkill_folder)
            
            final_answer = MobileNetGANInference.check_validation_result(underkill_count, validation_count)

            return final_answer
        elif project in VALIDATION_FLOW['mobilenetv2']:
            model_file = MobileNetGANInference.get_train_model_path(project, task_name)
            model = MobileNetGANInference.load_model(model_file)
            mean, std = MobileNetGANInference.get_mean_std(project, task_name)
            data_transforms = MobileNetGANInference.get_transform(mean, std)
            class_list = MobileNetGANInference.get_classes(project, task_name)

            validation_images = MobileNetGANInference.get_validation_images(project)
            validation_count = MobileNetGANInference.check_validation_count(validation_images)
            underkill_folder = MobileNetGANInference.get_underkill_folder(project, task_name)

            underkill_count = 0
            for validation_image in validation_images:
                answer = MobileNetGANInference.classification_inference(validation_image, model, class_list, data_transforms)
                if answer:
                    underkill_count += 1
                    MobileNetGANInference.output_underkill_image(validation_image, underkill_folder)
            
            final_answer = MobileNetGANInference.check_validation_result(underkill_count, validation_count)

    @classmethod
    def get_finetune_type(cls, tablename):
        return tablename.split('_')[0].upper()
    
    @classmethod
    def check_underkills(cls, project, task_name):
        object_detection_underkills = UnderkillDataProcessing.get_object_detection_underkill_path(project, task_name)
        classification_underkills = UnderkillDataProcessing.get_classification_underkill_path(project, task_name)
        object_detection_validations = UnderkillDataProcessing.get_object_detection_validations(project)
        classification_validations = UnderkillDataProcessing.get_classification_validations(project)

        if object_detection_underkills and classification_underkills:
            ...
        elif object_detection_underkills:
            result = UnderkillDataProcessing.check_model_pass_or_fail(object_detection_underkills, object_detection_validations)
            if result:
                return [], result
            else:
                return object_detection_underkills, result
        elif classification_validations:
            result = UnderkillDataProcessing.check_model_pass_or_fail(classification_underkills, classification_validations)
            if result:
                return [], result
            else:
                return classification_underkills, result
        else:
            return [], True
        
    @classmethod
    def upload_crop_categorizing(cls, images, group_type, record_id, finetune_type):
        Logger.info('Upload Database Record Crop Categorizing')
        crop_image_ids = []
        for image_path in images:
            image_uuid = CriticalNGService.get_image_uuid(image_path, group_type)
            image = YOLOInference.read_image(image_path)
            image_size = YOLOInference.get_image_size(image)
            crop_image_id = CropCategorizingRecordService.update_underkill_image(record_id, image_uuid, image_size[0], image_size[1], finetune_type)
            crop_image_ids.append(crop_image_id)

        return crop_image_ids

    @classmethod
    def upload_ai_model_information(cls, model_path, record_id, finetune_type, group_type, model_validate_result): 
        Logger.info('Upload Database Record AI Model Information')
        ip_address = UnderkillDataProcessing.get_ip_address()
        if model_validate_result:
            verified_status = 'APPROVE'
        else:
            verified_status = 'FAIL'
        AIModelInformationService.update(group_type, model_path, ip_address, record_id, finetune_type, verified_status)
        model_id = AIModelInformationService.get_model_id()
        Logger.info(f'Model Path: {model_path}, Validate Result: {verified_status}')

        return model_id

    @classmethod
    def upload_ai_model_performance(cls, project, task_name, model_id, underkills, model_path, crop_image_ids, training_datasets_inferece_task_id):
        Logger.info('Upload Database Record AI Model Performance')
        if 'best.pt' in model_path:
            loss = UnderkillDataProcessing.get_loss(project, task_name)
            accuracy = UnderkillDataProcessing.get_accuracy(underkills, project)

        metrics_result = UnderkillDataProcessing.get_metrics_result(loss, accuracy, crop_image_ids, training_datasets_inferece_task_id)
        false_negative_images = UnderkillDataProcessing.get_false_negative_images(crop_image_ids)
        false_positive_images = UnderkillDataProcessing.get_false_positive_images()

        AIModelPerformanceService.update(model_id, metrics_result, false_negative_images, false_positive_images)
        