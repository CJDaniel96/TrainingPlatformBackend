import time
from app.services.cvat_service import CVATService
from app.services.datasets_service import OriginDataProcessing


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
    def upload(cls, images_folder, xml_folder, project_id, lines, group_type, serial_number, token):
        task_name = CVATService.get_task_name(lines, group_type, serial_number)
        auth_header = CVATService.get_auth_header(token)
        task_create_information = CVATService.get_task_create_infomation(task_name, project_id)
        task_id = CVATService.create_task(auth_header, task_create_information)
        task_data = CVATService.get_task_data_json(images_folder)
        CVATService.upload_task_data(task_id, auth_header, task_data)
        task_annotation = OriginDataProcessing.zip_xml_data(xml_folder)
        CVATService.upload_task_annotation(task_id, auth_header, task_annotation)

        return task_id, task_name

    @classmethod
    def download(cls):...