from pathlib import Path
from urllib.parse import urljoin
import zipfile
from config import *
from requests.auth import HTTPBasicAuth
import requests




class CVATScript:
    def _make_request(self, method, endpoint, **kwargs):
        url = urljoin(CVAT_INFO['CVAT_URL'], endpoint)
        response = method(url, **kwargs)
        response.raise_for_status()
        
        return response
    
    def _post_request(self, endpoint, **kwargs):
        return self._make_request(requests.post, endpoint, **kwargs)

    def _get_request(self, endpoint, **kwargs):
        return self._make_request(requests.get, endpoint, **kwargs)

    def _put_request(self, endpoint, **kwargs):
        return self._make_request(requests.put, endpoint, **kwargs)
    
    def _session_post_request(self, session:requests.Session, endpoint, **kwargs):
        return self._make_request(session.post, endpoint, **kwargs)
    
    def _session_get_request(self, session:requests.Session, endpoint, **kwargs):
        return self._make_request(session.get, endpoint, **kwargs)
    
    def _upload_images(self, task_id, upload_folder, auth):
        with requests.Session() as session:
            response = self._session_post_request(
                session,
                CVAT_INFO['CVAT_TASKS_DATA_API'].format(task_id), 
                data=CVAT_INFO['CVAT_UPLOAD_INFORMATION'], 
                files={f'client_files[{i}]': f.open('rb') for i, f in enumerate(Path(upload_folder).glob('*.jp*'))}, 
                auth=auth
            )
        
            if response.status_code == 202:
                while True:
                    status = self._session_get_request(session, CVAT_INFO['CVAT_TASKS_STATUS_API'].format(task_id), auth=auth).json()
                    if status['state'] == 'Finished':
                        break 
                    elif status['state'] == 'Failed':
                        raise Exception(status['message'])
            else:
                raise Exception(response.text)
        
    def _upload_annotation(self, task_id, upload_folder, auth):
        xmls = Path(upload_folder).glob('*.xml')
        if not xmls:
            return

        task_annotation_zip_file = Path(upload_folder, 'xmls.zip')
        with zipfile.ZipFile(task_annotation_zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for xml in xmls:
                zf.write(str(xml), arcname=xml.name)
        upload_data = {
            'annotation_file': open(task_annotation_zip_file, 'rb')
        }
        parameters = {
            'format': CVAT_INFO['CVAT_ANNOTATION_FORMAT']
        }
        # parameters = urlencode(parameters, quote_via=quote)
        while True:
            response = self._put_request(CVAT_INFO['CVAT_TASKS_ANNOTATION_API'].format(task_id), params=parameters, files=upload_data, auth=auth)
            if response.status_code == 201:
                return
    
    @classmethod
    def get_auth(cls, username, password) -> HTTPBasicAuth:
        return HTTPBasicAuth(username, password)
        
    @classmethod
    def create_task(cls, lines, group_type, barcode, project_id, auth) -> str:
        data = {
            'name': '_'.join([group_type] + lines + [barcode]), 
            'labels': [], 
            'project_id': project_id
        }
        response = cls()._post_request(CVAT_INFO['CVAT_TASKS_API'], json=data, auth=auth).json()

        return response['id'], response['name']
    
    @classmethod
    def upload_task(cls, task_id, upload_folder, auth):
        cls()._upload_images(task_id, upload_folder, auth)
        cls()._upload_annotation(task_id, upload_folder, auth)

    @classmethod
    def check_task_download_format(cls, task_type):
        if task_type == 'object_detection':
            return CVAT_INFO['CVAT_DOWNLOAD_FORMAT']
        elif task_type == 'classification':
            return CVAT_INFO['CVAT_CLASSIFICATION_FORMAT']

    @classmethod
    def download_task(cls, task_id, task_name, auth, format=CVAT_INFO['CVAT_DOWNLOAD_FORMAT']) -> str:
        parameters = {
            'action': 'download', 
            'format': format
        }
        while True:
            response = cls()._get_request(CVAT_INFO['CVAT_TASKS_DATASET_API'].format(task_id), stream=True, params=parameters, auth=auth)
            if response.status_code == 200:
                break
        zip_file_path = Path(DOWNLOADS_DATA_DIR, task_name + '.zip')
        with Path(DOWNLOADS_DATA_DIR, task_name + '.zip').open('wb') as zip_file:
            for chunk in response.iter_content(chunk_size=8096):
                zip_file.write(chunk)
            
        return str(zip_file_path)