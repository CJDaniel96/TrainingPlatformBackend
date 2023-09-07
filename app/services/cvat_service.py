import os
from urllib.parse import quote, urlencode, urljoin
from app.config import CVAT
import requests

from data.config import TMP_DIR


class CVATService:
    @classmethod
    def get_login_cookies(cls):
        login_api_url = urljoin(CVAT['CVAT_URL'], CVAT['CVAT_LOGIN_API'])
        response = requests.post(login_api_url, json=CVAT['CVAT_LOGIN_INFORMATION'])

        return response
    
    @classmethod
    def get_logout(cls):
        logout_api_url = urljoin(CVAT['CVAT_URL'], CVAT['CVAT_LOGOUT_API'])
        response = requests.post(logout_api_url)

        return response

    @classmethod
    def get_task_name(cls, lines, group_type, serial_number):
        line_name = ''
        for line in lines:
            line_name += line

        return group_type + '_' + line_name + '_' + serial_number
    
    @classmethod
    def get_auth_header(cls, token):
        return {'Authorization': 'Token ' + token}
    
    @classmethod
    def get_task_create_infomation(cls, task, project_id):
        return {
            'name': task, 
            'labels': [], 
            'project_id': project_id
        }
    
    @classmethod
    def create_task(cls, auth_header, task_create_information):
        tasks_api_url = urljoin(CVAT['CVAT_URL'], CVAT['CVAT_TASKS_API'])
        response = requests.post(tasks_api_url, headers=auth_header, data=task_create_information)
        task_id = response.json()['id']

        return task_id
    
    @classmethod
    def get_task_data_json(cls, folder):
        return {'client_files[{}]'.format(i): open(os.path.join(folder, f), 'rb') for i, f in enumerate(os.listdir(folder))}

    @classmethod
    def upload_task_data(cls, task_id, auth_header, task_data):
        upload_url = urljoin(CVAT['CVAT_URL'], CVAT['CVAT_TASKS_DATA_API']).format(task_id)
        response = requests.post(upload_url, data=CVAT['CVAT_UPLOAD_INFORMATION'], headers=auth_header, files=task_data)
        response.raise_for_status()

        status_url = urljoin(CVAT['CVAT_URL'], CVAT['CVAT_TASKS_STATUS_API']).format(task_id)
        while True:
            response = requests.get(status_url, headers=auth_header)
            status = response.json()
            if status['state'] == 'Finished':
                return 
            elif status['state'] == 'Failed':
                raise Exception(status['message'])

    @classmethod
    def upload_task_annotation(cls, task_id, auth_header, task_annotation):
        upload_url = urljoin(CVAT['CVAT_URL'], CVAT['CVAT_TASKS_ANNOTATION_API']).format(task_id)
        upload_data = {
            'annotation_file': open(task_annotation, 'rb')
        }
        parameters = {
            'format': CVAT['CVAT_ANNOTATION_FORMAT']
        }
        parameters = urlencode(parameters, quote_via=quote)
        while True:
            response = requests.put(upload_url, headers=auth_header, params=parameters, files=upload_data)
            response.raise_for_status()
            if response.status_code == 201:
                return

    @classmethod
    def check_task_download_format(cls, task_type):
        if task_type == 'object_detection':
            return CVAT['CVAT_DOWNLOAD_FORMAT']
        elif task_type == 'classification':
            return CVAT['CVAT_CLASSIFICATION_FORMAT']

    @classmethod
    def task_download(cls, task_id, task_name, auth_header, format=CVAT['CVAT_DOWNLOAD_FORMAT']):
        download_url = urljoin(CVAT['CVAT_URL'], CVAT['CVAT_TASKS_DATASET_API']).format(task_id)
        parameters = {
            'action': 'download', 
            'format': format,
            'filename': task_name
        }
        parameters = urlencode(parameters, quote_via=quote)
        while True:
            response = requests.get(download_url, headers=auth_header, params=parameters)
            response.raise_for_status()
            if len(response.content) != 0:
                zip_file = os.path.join(TMP_DIR, task_name + '.zip')
                with open(zip_file, 'wb') as f:
                    f.write(response.content)
                return zip_file
