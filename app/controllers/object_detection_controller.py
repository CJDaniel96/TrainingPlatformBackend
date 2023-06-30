from app.services.database_service import CategoryMappingService, IRIRecordService, TrainingInfoService, URDRecordService
from app.services.datasets_service import ObjectDetectionTrainDataProcessing
from app.services.inference_service import CHIPRCInference, PCIEInference, YOLOInference
from app.services.logging_service import Logger
from app.services.train_service import YOLOTrain


class ObjectDetectionController:
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
    def get_hyps_yaml(cls, project):
        return ObjectDetectionTrainDataProcessing.get_hyps_yaml(project)
    
    @classmethod
    def train(cls, project, task_name, data, cfg):
        Logger.info('YOLO Training Start...')
        if project == 'NK_DAOI_CHIPRC_2':
            hyp = cls().get_hyps_yaml(project)
            YOLOTrain.train_model(project, task_name, data, cfg, hyp)
        else:
            YOLOTrain.train_model(project, task_name, data, cfg)

    @classmethod
    def validate(cls, project, task_name):
        Logger.info('YOLO Validating Start...')
        if project == 'NK_DAOI_CHIPRC_2':
            model_file = CHIPRCInference.get_train_model_path(project, task_name)
            model = CHIPRCInference.load_model(model_file)
            generator_model_file = CHIPRCInference.get_generator_model_path(project)
            discriminator_model_file = CHIPRCInference.get_discriminator_model_path(project)
            encoder_model_file = CHIPRCInference.get_encoder_model_path(project)
            generator = CHIPRCInference.get_generator_model(generator_model_file)
            discriminator = CHIPRCInference.get_discriminator_model(discriminator_model_file)
            encoder = CHIPRCInference.get_encoder_model(encoder_model_file)
            transform = CHIPRCInference.get_transform()
            criterion = CHIPRCInference.get_criterion()

            validation_images = CHIPRCInference.get_validation_images(project)
            validation_count = CHIPRCInference.check_validation_count(validation_images)
            underkill_folder = CHIPRCInference.get_underkill_folder(project, task_name)
            
            underkill_count = 0
            for validation_image in validation_images:
                answer, chiprcs = CHIPRCInference.yolo_predict(model, validation_image)
                if answer and CHIPRCInference.vae_predict(validation_image, chiprcs, transform, generator, discriminator, encoder, criterion):
                    underkill_count += 1
                    CHIPRCInference.output_underkill_image(validation_image, underkill_folder)

            final_answer = CHIPRCInference.check_validation_result(underkill_count, validation_count)

            return final_answer
        
        elif project == 'NK_PCIE_2':
            model_file = PCIEInference.get_train_model_path(project, task_name)
            classification_model_path = PCIEInference.get_classification_model_path(project)
            pinlocation_model_path = PCIEInference.get_pinlocation_model_path(project)
            model = PCIEInference.load_model(model_file)
            classification_model = PCIEInference.get_classification_model(classification_model_path)
            pinlocation_model = PCIEInference.get_pinlocation_model(pinlocation_model_path)

            validation_images = PCIEInference.get_validation_images(project)
            validation_count = CHIPRCInference.check_validation_count(validation_images)
            underkill_folder = PCIEInference.get_underkill_folder(project, task_name)

            underkill_count = 0
            for validation_image in validation_images:
                answer = PCIEInference.classification_inference(validation_image, classification_model)
                if answer and PCIEInference.object_detection_inference(model, pinlocation_model, validation_image):
                    underkill_count += 1
                    PCIEInference.output_underkill_image(validation_image, underkill_folder)

            final_answer = PCIEInference.check_validation_result(underkill_count, validation_count)

            return final_answer