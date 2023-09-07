from app.config import TRAINING_FLOW, VALIDATION_FLOW, YOLOV5
from app.services.database_service import CategoryMappingService, IRIRecordService, TrainingInfoService, URDRecordService
from app.services.datasets_service import ObjectDetectionTrainDataProcessing
from app.services.inference_service import YOLOFanoGANInference, MobileNetYOLOIForestInference, YOLOInference
from app.services.logging_service import Logger
from app.services.train_service import YOLOTrain


class ObjectDetectionController:
    @classmethod
    def check_coutinue_train_classification_model(cls, project):
        Logger.info('Check Whether Train Object Detection Model')
        if 'object_detection' in TRAINING_FLOW[project]:
            return True
        else:
            return False

    @classmethod
    def get_object_detection_tasks(cls, task_id, group_type, tablename):
        tasks_id = TrainingInfoService.get_object_detection_tasks_id(task_id, group_type)
        if tablename == 'iri_record':
            return IRIRecordService.get_tasks(tasks_id)
        elif tablename == 'urd_record':
            return URDRecordService.get_tasks(tasks_id)

    @classmethod
    def get_train_data_folder(cls, project, task_name):
        return ObjectDetectionTrainDataProcessing.get_object_detection_train_data_folder(project, task_name)

    @classmethod
    def get_train_dataset(cls, task_zip_file, train_data_folder):
        if ObjectDetectionTrainDataProcessing.check_zip_file(task_zip_file):
            ObjectDetectionTrainDataProcessing.get_object_detection_train_data(train_data_folder)
            ObjectDetectionTrainDataProcessing.clear_zip_file(task_zip_file)

        return
        
    @classmethod
    def merge_basicline_dataset(cls, train_data_folder, project):
        Logger.info('Merge Local Basicline Dataset in Train Dataset')
        ObjectDetectionTrainDataProcessing.merge_object_detection_basicline_data(train_data_folder, project)

    @classmethod
    def get_data_yaml(cls, site, group_type, project, train_data_folder):
        class_names = CategoryMappingService.get_class_names(site, group_type, project)
        return ObjectDetectionTrainDataProcessing.write_data_yaml(project, class_names, train_data_folder)

    @classmethod
    def get_models_yaml(cls, project):
        return ObjectDetectionTrainDataProcessing.get_models_yaml(project)
    
    @classmethod
    def train(cls, project, task_name, data, cfg):
        Logger.info('YOLO Training Start...')
        if project in YOLOV5['YOLOV5_HYP_RANDOM_CROP_CLOSE_PROJECT']:
            hyp = ObjectDetectionTrainDataProcessing.get_random_crop_close_hyp_yaml()
            YOLOTrain.train_model(project, task_name, data, cfg, hyp)
        else:
            hyp = ObjectDetectionTrainDataProcessing.get_default_hyp_yaml()
            YOLOTrain.train_model(project, task_name, data, cfg, hyp)

    @classmethod
    def validate(cls, project, task_name):
        Logger.info('YOLO Validating Start...')
        if project in VALIDATION_FLOW['yolo_fanogan']:
            model_file = YOLOFanoGANInference.get_train_model_path(project, task_name)
            model = YOLOFanoGANInference.load_model(model_file)
            generator_model_file = YOLOFanoGANInference.get_generator_model_path(project)
            discriminator_model_file = YOLOFanoGANInference.get_discriminator_model_path(project)
            encoder_model_file = YOLOFanoGANInference.get_encoder_model_path(project)
            generator = YOLOFanoGANInference.get_generator_model(
                generator_model_file, 
                VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['img_size'],
                VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['latent_dim'], 
                VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['channels']
            )
            discriminator = YOLOFanoGANInference.get_discriminator_model(
                discriminator_model_file, 
                VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['img_size'],  
                VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['channels']
            )
            encoder = YOLOFanoGANInference.get_encoder_model(
                encoder_model_file, 
                VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['img_size'],
                VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['latent_dim'], 
                VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['channels']
            )
            transform = YOLOFanoGANInference.get_transform()
            criterion = YOLOFanoGANInference.get_criterion()

            validation_images = YOLOFanoGANInference.get_validation_images(project)
            validation_count = YOLOFanoGANInference.check_validation_count(validation_images)
            underkill_folder = YOLOFanoGANInference.get_underkill_folder(project, task_name)

            kappa = VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['kappa']
            anormaly_threshold = VALIDATION_FLOW['yolo_fanogan'][project]['gan_settings']['anormaly_threshold']
            confidence = VALIDATION_FLOW['yolo_fanogan'][project]['confidence']
            
            underkill_count = 0
            for validation_image in validation_images:
                answer, target_df = YOLOFanoGANInference.yolo_predict(model, validation_image, project, confidence)
                if answer and YOLOFanoGANInference.vae_predict(
                    validation_image, target_df, transform, generator, discriminator, encoder, criterion, kappa, anormaly_threshold
                ):
                    underkill_count += 1
                    YOLOFanoGANInference.output_underkill_image(validation_image, underkill_folder)

            final_answer = YOLOFanoGANInference.check_validation_result(underkill_count, validation_count)

            return final_answer

        elif project in VALIDATION_FLOW['yolo']:
            model_file = YOLOFanoGANInference.get_train_model_path(project, task_name)
            model = YOLOFanoGANInference.load_model(model_file)

            validation_images = YOLOFanoGANInference.get_validation_images(project)
            validation_count = YOLOFanoGANInference.check_validation_count(validation_images)
            underkill_folder = YOLOFanoGANInference.get_underkill_folder(project, task_name)

            underkill_count = 0
            for validation_image in validation_images:
                answer = YOLOFanoGANInference.yolo_predict(model, validation_image, project)
                if answer:
                    underkill_count += 1
                    YOLOFanoGANInference.output_underkill_image(validation_image, underkill_folder)

            final_answer = YOLOFanoGANInference.check_validation_result(underkill_count, validation_count)

            return final_answer
        
        elif project in VALIDATION_FLOW['mobilenetv2_yolo_iforest']:
            model_file = MobileNetYOLOIForestInference.get_train_model_path(project, task_name)
            classification_model_path = MobileNetYOLOIForestInference.get_classification_model_path(project)
            pinlocation_model_path = MobileNetYOLOIForestInference.get_pinlocation_model_path(project)
            model = MobileNetYOLOIForestInference.load_model(model_file)
            classification_model = MobileNetYOLOIForestInference.get_classification_model(classification_model_path)
            pinlocation_model = MobileNetYOLOIForestInference.get_pinlocation_model(pinlocation_model_path)

            validation_images = MobileNetYOLOIForestInference.get_validation_images(project)
            validation_count = YOLOFanoGANInference.check_validation_count(validation_images)
            underkill_folder = MobileNetYOLOIForestInference.get_underkill_folder(project, task_name)

            underkill_count = 0
            for validation_image in validation_images:
                answer = MobileNetYOLOIForestInference.classification_inference(validation_image, classification_model)
                if answer and MobileNetYOLOIForestInference.object_detection_inference(model, pinlocation_model, validation_image):
                    underkill_count += 1
                    MobileNetYOLOIForestInference.output_underkill_image(validation_image, underkill_folder)

            final_answer = MobileNetYOLOIForestInference.check_validation_result(underkill_count, validation_count)

            return final_answer