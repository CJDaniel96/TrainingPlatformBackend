from app.config import TRAINING_PLATFORM_RECORD_STATUS
from app.controllers.Initial_controller import InitialController
from app.controllers.cvat_controller import CVATController
from app.controllers.inference_controller import YOLOInferenceController
from app.controllers.listen_controller import Listener


def app_run():
    # Get Task
    record = Listener.listen()
    status = record.status
    id = record.id
    tablename = record.__tablename__
    
    if status == 'Init':
        image_mode = record.image_mode
        initial_controller = InitialController(record)

        if image_mode == 'query':
            images = initial_controller.get_query_data()
            lines = eval(record.line)
        elif image_mode == 'upload':
            images = initial_controller.get_upload_data()
            lines = list(images.keys())
            initial_controller.update_record_line(lines)
        elif image_mode == 'upload_image':
            images = initial_controller.get_upload_image_data()
            lines = list(images.keys())
            initial_controller.update_record_line(lines)

        InitialController.download_images(images)
        serial_number = InitialController.get_serial_number()
        org_image_folder = InitialController.arrange_origin_images(serial_number)

        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['INFERENCE_ON_GOING'])
        project = record.project
        org_xml_folder = YOLOInferenceController.inference(org_image_folder, project)
        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['INFERENCE_FINISH'])

        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['UPLOAD_IMAGE_WITH_LOG_ON_GOING'])
        project_id = record.project_id
        group_type = record.group_type
        cvat_cookie = CVATController.login()
        task_id, task_name = CVATController.upload(org_image_folder, org_xml_folder, project_id, lines, group_type, serial_number, cvat_cookie)
        Listener.update_record_task_id(tablename, id, task_id)
        Listener.update_record_task_name(tablename, id, task_name)
        Listener.update_record_status(tablename, id, TRAINING_PLATFORM_RECORD_STATUS['UPLOAD_IMAGE_WITH_LOG_FINISH'])
    elif status == 'Categorizing' and tablename == 'urd_record':
        ...
    elif status == 'OD_Initialized':
        ...
    elif status == 'CLS_Initialized':
        ...