from app.config import TRAINING_PLATFORM_RECORD_STATUS, TRAINING_STATUS
from app.controllers.Initial_controller import InitialController
from app.controllers.categorize_controller import CategorizeController
from app.controllers.check_environment_controller import CheckEnvironmentController
from app.controllers.classification_controller import ClassificationController
from app.controllers.cvat_controller import CVATController
from app.controllers.inference_controller import YOLOInferenceController
from app.controllers.listen_controller import Listener
from app.controllers.object_detection_controller import ObjectDetectionController


def app_run():
    # Get Task
    record = Listener.listen()
    status = record.status
    id = record.id
    tablename = record.__tablename__
    
    if status == 'Init':
        cvat_cookie = CVATController.login()
        image_mode = record.image_mode
        project = record.project
        project_id = record.project_id
        group_type = record.group_type

        if image_mode == 'query':
            site = record.site
            line = record.line
            group_type = record.group_type
            start_date = record.start_date
            end_date = record.end_date
            smart_filter = record.smart_filter
            images = InitialController.get_query_data(site, line, group_type, start_date, end_date, smart_filter)
            lines = eval(record.line)
        elif image_mode == 'upload':
            uuids = record.images
            images = InitialController.get_upload_data(uuids)
            lines = InitialController.get_image_lines(images)
            InitialController.update_record_line(tablename, id, lines)
        elif image_mode == 'upload_image':
            uuids = record.images
            images = InitialController.get_upload_image_data(uuids)
            lines = InitialController.get_image_lines(images)
            InitialController.update_record_line(tablename, id, lines)

        InitialController.download_images(images, image_mode)
        serial_number = InitialController.get_serial_number()
        org_image_folder = InitialController.arrange_origin_images(serial_number)

        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['INFERENCE_ON_GOING'])
        org_xml_folder = YOLOInferenceController.inference(org_image_folder, project)
        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['INFERENCE_FINISH'])

        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['UPLOAD_IMAGE_WITH_LOG_ON_GOING'])
        task_name = CVATController.get_task_name(lines, group_type, serial_number)
        task_id = CVATController.upload(org_image_folder, org_xml_folder, project_id, task_name, cvat_cookie)
        Listener.update_record_task_id(tablename, id, task_id)
        Listener.update_record_task_name(tablename, id, task_name)
        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['UPLOAD_IMAGE_WITH_LOG_FINISH'])
        CVATController.logout()
    elif status == 'Categorizing' and tablename == 'urd_record':
        category_ready = record.category_ready
        project = record.project
        task_id = record.task_id
        task_name = record.task
        group_type = record.group_type
        site = record.site

        if category_ready != True:
            cvat_cookie = CVATController.login()
            task_zip_file = CVATController.download(task_id, task_name, cvat_cookie)
            train_data_folder = ObjectDetectionController.get_train_data_folder(project, task_name)
            ObjectDetectionController.get_train_dataset(task_zip_file, train_data_folder)
            CategorizeController.upload_categorizing_record(id, site, project, group_type, train_data_folder)
            Listener.update_category_ready(id)
            CVATController.logout()
        Listener.timesleep()
    elif status == 'OD_Initialized':
        cvat_cookie = CVATController.login()
        project = record.project
        task_name = record.task
        task_id = record.task_id
        group_type = record.group_type
        site = record.site

        if ObjectDetectionController.check_coutinue_train_classification_model(project):
            CheckEnvironmentController.clear_local_data(status, project, task_name)

            Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['TRIGGER_TRAINING_FOR_OD'])
            task_zip_file = CVATController.download(task_id, task_name, cvat_cookie)
            train_data_folder = ObjectDetectionController.get_train_data_folder(project, task_name)
            ObjectDetectionController.get_train_dataset(task_zip_file, train_data_folder)
            
            tasks = ObjectDetectionController.get_object_detection_tasks(task_id, group_type, tablename)
            if tasks:
                for id, name in tasks:
                    task_zip_file = CVATController.download(id, name, cvat_cookie)
                    train_data_folder = ObjectDetectionController.get_train_dataset(task_zip_file, project, task_name)
            ObjectDetectionController.merge_basicline_dataset(train_data_folder, project)

            Listener.update_record_object_detection_training_status(tablename, id, TRAINING_STATUS['RUNNING'])
            
            data_yaml = ObjectDetectionController.get_data_yaml(site, group_type, project, train_data_folder)
            models_yaml = ObjectDetectionController.get_models_yaml(project)

            Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['TRAINING_FOR_OD'])
            ObjectDetectionController.train(project, task_name, data_yaml, models_yaml)

            Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['VERIFYING_FOR_OD'])
            result = ObjectDetectionController.validate(project, task_name)
            Listener.update_record_object_detection_training_info(result, task_id, group_type)
            Listener.update_record_object_detection_training_status(tablename, id, TRAINING_STATUS['DONE'])
        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['FINISH_FOR_OD'])
        CVATController.logout()
    elif status == 'CLS_Initialized':
        cvat_cookie = CVATController.login()
        project = record.project
        project_id = record.project_id
        task_id = record.task_id
        task_name = record.task
        group_type = record.group_type
        if ClassificationController.check_coutinue_train_classification_model(project):
            ...
        elif ClassificationController.check_only_train_classification_model(project):
            CheckEnvironmentController.clear_local_data(status, project, task_name)
            Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['TRIGGER_TRAINING_FOR_CLS'])
            task_zip_file = CVATController.download(task_id, task_name, cvat_cookie, 'classification')
            train_data_folder = ClassificationController.get_train_data_folder(project, task_name)
            ClassificationController.get_train_dataset(task_zip_file, train_data_folder)

            tasks = ClassificationController.get_classification_tasks(task_id, group_type, tablename)
            if tasks:
                for id, name in tasks:
                    task_zip_file = CVATController.download(id, name, cvat_cookie)
                    train_data_folder = ClassificationController.get_train_dataset(task_zip_file, project, task_name)
            ClassificationController.merge_basicline_dataset(train_data_folder, project)

            Listener.update_record_classification_training_status(tablename, id, TRAINING_STATUS['RUNNING'])
            Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['TRAINING_FOR_CLS'])

            ClassificationController.train(project, task_name, train_data_folder)

            Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['VERIFYING_FOR_CLS'])

            result = ClassificationController.validate(project, task_name)
            Listener.update_record_classification_training_info(result, task_id, group_type)
            Listener.update_record_classification_training_status(tablename, id, TRAINING_STATUS['DONE'])

            train_model_path = ClassificationController.get_train_model_path(project, task_name)
        else:
            train_dataset_inference_task_name = CVATController.custom_task_name(task_name, 'inference')
            task_zip_file = CVATController.download(task_id, train_dataset_inference_task_name, cvat_cookie)
            train_data_folder = ObjectDetectionController.get_train_data_folder(project, train_dataset_inference_task_name)
            ObjectDetectionController.get_train_dataset(task_zip_file, train_data_folder)

            train_model_path = YOLOInferenceController.get_train_model_path(project, task_name)
            train_dataset_inference_image_folder, train_dataset_inference_xml_folder = YOLOInferenceController.train_dataset_inference(project, task_name, train_model_path, train_data_folder)
            train_dataset_inference_task_id = CVATController.upload(train_dataset_inference_image_folder, train_dataset_inference_xml_folder, project_id, train_dataset_inference_task_name, cvat_cookie)

        finetune_type = ClassificationController.get_finetune_type(tablename)
        underkills, result = ClassificationController.check_underkills(project, task_name)
        crop_image_ids = ClassificationController.upload_crop_categorizing(underkills, group_type, id, finetune_type)
        model_id = ClassificationController.upload_ai_model_information(train_model_path, id, finetune_type, group_type, result)
        ClassificationController.upload_ai_model_performance(project, task_name, model_id, underkills, train_model_path, crop_image_ids, train_dataset_inference_task_id)
            
        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['FINISHED'])
        CVATController.logout()