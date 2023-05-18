import time
from app.services.cvat_service import CVATService
from app.services.datasets_service import OriginDataProcessing
from app.services.logging_service import Logger


class CVATController:
    @classmethod
    def login(cls):
        response =  CVATService.get_login_cookies()
        if response.status_code != 200:
            raise Exception('CVAT login fail!')
        else:
            return response.json()['key']

    @classmethod
    def logout(cls):
        response = CVATService.get_logout()
        if response.status_code != 200:
            raise Exception('CVAT logout fail!')
        
    @classmethod
    def custom_task_name(*names):
        task_name = ''
        for name in names:
            if type(name) is str:
                task_name += name + '_'
        task_name = task_name[:-1]

        return task_name
        
    @classmethod
    def get_task_name(cls, lines, group_type, serial_number):
        return CVATService.get_task_name(lines, group_type, serial_number)

    @classmethod
    def upload(cls, images_folder, xml_folder, project_id, task_name, token):
        Logger.info('Upload Data To CVAT')
        auth_header = CVATService.get_auth_header(token)
        task_create_information = CVATService.get_task_create_infomation(task_name, project_id)
        task_id = CVATService.create_task(auth_header, task_create_information)
        task_data = CVATService.get_task_data_json(images_folder)
        CVATService.upload_task_data(task_id, auth_header, task_data)
        task_annotation = OriginDataProcessing.zip_xml_data(xml_folder)
        CVATService.upload_task_annotation(task_id, auth_header, task_annotation)

        return task_id

    @classmethod
    def download(cls, task_id, task_name, token):
        Logger.info('Download Data From CVAT')
        auth_header = CVATService.get_auth_header(token)
        task_zip_file = CVATService.task_download(task_id, task_name, auth_header)

        return task_zip_file
