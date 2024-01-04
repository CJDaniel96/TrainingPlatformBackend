from sklearn.model_selection import train_test_split
from config import *
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urljoin
from scripts.cvat_script import CVATScript
from scripts.yaml_script import Yolov5Yaml
from scripts import validation_scripts
from scripts.logger import Logger
import requests
import os
import random
import shutil




load_dotenv()
FLASK_RUN_HOST = os.getenv('FLASK_RUN_HOST')
FLASK_RUN_PORT = os.getenv('FLASK_RUN_PORT')
FLASK_API_PREFIX = os.getenv('FLASK_API_PREFIX')
FLASK_DEBUG = os.getenv('FLASK_DEBUG')
CVAT_USERNAME = os.getenv('CVAT_USERNAME')
CVAT_PASSWORD = os.getenv('CVAT_PASSWORD')
API_URL = urljoin(f'http://{FLASK_RUN_HOST}:{FLASK_RUN_PORT}', FLASK_API_PREFIX)

    
class Inference:
    @classmethod
    def yolov5_inference(cls, model, images):
        data = {
            'model': model,
            'images': images
        }
        response = requests.post(urljoin(API_URL, 'models/yolov5/inference'), json=data)
        inference_results = response.json()
        
        return inference_results
    
    @classmethod
    def to_xml(cls, results):
        response = requests.post(urljoin(API_URL, 'utils/output_xmls'), json=results)
        data = response.json()['data']
        for info in data:
            if info['status'] == 'error':
                return False, info['message']
        
        return True, 'success'
    
    @classmethod
    def object_detection_inference(cls, algorithm, model, images):
        method_name = f'{algorithm}_inference'
        if method_name in cls().__dir__():
            method = getattr(cls(), method_name)
            inference_results = method(model, images)
            
        return inference_results
    
    @classmethod
    def _yolov5_validate_underkill(cls, project, inference_results):
        underkill_images = []
        params = {
            'project': project
        }
        ok_labels = requests.post(urljoin(API_URL, 'category_mapping/ok_labels'), params=params)
        method = getattr(validation_scripts, project)
        for result in inference_results:
            if method.predict(result, ok_labels):
                underkill_images.append(result['image_path'])

        return underkill_images
    
    @classmethod
    def validate_underkill(cls, algorithm, project, inference_results):
        method_name = f'_{algorithm}_validate_underkill'
        if method_name in cls().__dir__():
            method = getattr(cls(), method_name)
            underkill_images = method(project, inference_results)
            
        return underkill_images
    

class TrainModel:
    def _yolov5_train(self, algorithm, datasets, classes, project, task_name):
        data_yaml_path = Yolov5Yaml.get_data_yaml(algorithm, datasets, classes)
        model_yaml_path = Yolov5Yaml.get_model_yaml(algorithm, len(classes))
        hyp_yaml_path = Yolov5Yaml.get_hyp_yaml(algorithm, project)
        
        if Path(MODELS_DIR, project, 'inference/yolo_model.pt').exists():
            weights = str(Path(MODELS_DIR, project, 'inference/yolo_model.pt'))
        else:
            weights = str(Path(YOLOV5_DIR, 'yolov5s.pt'))
            
        save_dir = Path(MODELS_DIR, project, 'train')
        
        data = {
            'weights': weights,
            'data': data_yaml_path, 
            'cfg': model_yaml_path,
            'hyp': hyp_yaml_path, 
            'batch_size': YOLOV5_BATCH_SIZE, 
            'epochs': YOLOV5_EPOCHS, 
            'seed': YOLOV5_SEED, 
            'project': str(save_dir), 
            'name': task_name
        }
        
        response = requests.post(urljoin(API_URL, 'models/yolov5/train'), json=data)
        
        model_path = response.json()['data']['best_model_path']
            
        return model_path
    
    @classmethod
    def object_detection_train(cls, algorithm, datasets, classes, project, task_name):
        method_name = f'_{algorithm}_train'
        if method_name in cls().__dir__():
            method = getattr(cls(), method_name)
            model_path = method(algorithm, datasets, classes, project, task_name)
            
        return model_path


class Monitor:
    def __init__(self) -> None:
        self.api_url = API_URL
        self.logger = Logger(name=__name__, level='DEBUG' if FLASK_DEBUG else 'OFF')
    
    def _inference(self, unzip_data_folder, project, flow) -> None:
        self.logger.info('Model Inference...')
        if TRAINING_FLOW[SITE][project][flow] == 'yolov5':
            model = str(Path(MODELS_DIR, project, 'inference', 'yolo_model.pt').resolve())
            images = [str(image_path) for image_path in sorted(Path(unzip_data_folder).glob('*.jp[eg]*'))]
            inference_result = Inference.yolov5_inference(model, images)
            status, message = Inference.to_xml(inference_result['data'])
            if not status:
                self.logger.error(message)
        self.logger.info('Model Inference Finish')

    def _update_record(self, tablename, id, site='', lines=None, status='', task_id='', task_name='', od_training=None, cls_training=None):
        data = {
            'id': id,
            'site': site,
            'line': lines,
            'status': status, 
            'task_id': task_id,
            'task': task_name, 
            'od_training': od_training,
            'cls_training': cls_training
        }
        response = requests.put(urljoin(API_URL, f'status/{tablename}'), json=data)

        return response.status_code == 200

    
    def _get_images(self, image_mode, site, line, group_type, start_date, end_date, smart_filter, uuids) -> dict:
        self.logger.info('Get Image Path From Database')
        if image_mode == 'query':
            params = {
                'site': site,
                'line': line,
                'group_type': group_type,
                'start_date': start_date,
                'end_date': end_date,
                'smart_filter': smart_filter
            }
            if site in IMAGES_ASSIGN_LIGHT_TYPE and group_type in IMAGES_ASSIGN_LIGHT_TYPE[site]:
                url = urljoin(API_URL, 'data/images/assignlight')
            else:
                url = urljoin(API_URL, 'data/images')
            response = requests.get(url, params=params)
        else:
            data = {
                'image_mode': image_mode,
                'uuids': uuids
            }
            response = requests.post(urljoin(API_URL, 'data/images/uuids'), json=data)
            
        return response.json()
    
    def _images_download_format(self, images, smart_filter) -> dict:
        self.logger.info('Format Download Image')
        response = requests.get(urljoin(API_URL, 'image_pool/info'))
        image_pool_list = response.json().get('data')
        for image_pool in image_pool_list:
            image_pool['images'] = []
            
        if smart_filter:
            random.shuffle(images)
            images = images[:SMART_FILTER_NUMBER]

        for image_info in images:
            for image_pool in image_pool_list:
                if image_pool['line'] == image_info['line_id']:
                    image_pool['images'].append(IMAGE_DOWNLOAD_PREFIX + image_pool['prefix'] + '/' + image_info['image_path'])
                elif image_pool['line'] == 'upload_image' and image_info['line_id'] == 'other':
                    image_pool['images'].append(IMAGE_DOWNLOAD_PREFIX + 'upload_image' + '/' + image_info['image_path'])
                
        return image_pool_list
    
    def _download_images(self, download_image_list) -> bool:
        self.logger.info('Download Image...')
        for download_image_dict in download_image_list:
            if download_image_dict['images']:
                download_url = f'http://{download_image_dict["ip"]}/imagesinzip'
                download_proxies = {'http': download_url}
                response = requests.post(download_url, proxies=download_proxies, json={
                    "paths": download_image_dict['images']
                })
                if response.status_code == 200:
                    with Path(DOWNLOADS_DATA_DIR, download_image_dict['line'] + '.zip').open('wb') as zip_file:
                        zip_file.write(response.content)
        
        self.logger.info('Download Image Finish')
               
        return True
    
    def _get_barcode(self):
        response = requests.get(urljoin(API_URL, 'utils/barcode'))
        barcode_json = response.json()
        barcode = barcode_json['data']['datetime_barcode']
        
        return barcode
    
    def _get_download_zip_data(self):
        zip_data = sorted(Path(DOWNLOADS_DATA_DIR).glob('*.zip'))
        
        return [str(zip_file.resolve()) for zip_file in zip_data]
            
    def _unzip_data(self, paths, name) -> str:
        self.logger.info('Unzip Data...')
        data = {
            'name': name,
            'paths': paths
        }
        response = requests.post(urljoin(API_URL, 'utils/unzip_data'), json=data)
        unzip_data_json = response.json()
        self.logger.info('Unzip Data Finish')
        
        return unzip_data_json['data']['unzip_data_folder']
    
    def _upload_cvat(self, lines, group_type, barcode, project_id, unzip_data_folder) -> None:
        auth = CVATScript.get_auth(CVAT_USERNAME, CVAT_PASSWORD)
        task_id, task_name = CVATScript.create_task(lines, group_type, barcode, project_id, auth)
        CVATScript.upload_task(task_id, unzip_data_folder, auth)
        
        return task_id, task_name
    
    def _organize_to_training_dataset(self, project, task_name, algorithm, image_folder_path, label_folder_path):
        baseline = Path(BASELINE_DATASETS_DIR, project, algorithm)
        datasets = Path(TRAIN_DATASETS_DIR, project, task_name)
        train_images = datasets.joinpath('images/train')
        train_labels = datasets.joinpath('labels/train')
        val_images = datasets.joinpath('images/val')
        val_labels = datasets.joinpath('labels/val')
        if baseline.exists():
            shutil.copytree(str(baseline), datasets, dirs_exist_ok=True)
        else:
            train_images.mkdir(parents=True, exist_ok=True)
            train_labels.mkdir(parents=True, exist_ok=True)
            val_images.mkdir(parents=True, exist_ok=True)
            val_labels.mkdir(parents=True, exist_ok=True)

        images = sorted(Path(image_folder_path).glob('*.jp*'))
        x_train, x_test = train_test_split(images, test_size=TRAIN_TEST_RATIO)
        for image_path in x_train:
            txt_path = Path(label_folder_path, image_path.stem + '.txt')
            shutil.copy(str(image_path), train_images)
            shutil.copy(str(txt_path), train_labels)
        for image_path in x_test:
            txt_path = Path(label_folder_path, image_path.stem + '.txt')
            shutil.copy(str(image_path), val_images)
            shutil.copy(str(txt_path), val_labels)
            
        return str(datasets)
        
    def _xml2yolo(self, classes:dict, xml_folder_path):
        data = {
            'classes': classes, 
            'annotations_path': xml_folder_path,
        }
        response = requests.post(urljoin(API_URL, 'utils/xml2yolo'), json=data)
        labels_path = response.json()['data']['labels_path']
        
        return labels_path
        
    def _get_project_classes(self, line, group_type, project):
        params = {
            'line': line,
            'group_type': group_type,
            'project': project
        }
        response = requests.get(urljoin(API_URL, 'category_mapping/labels'), params=params)
        classes = response.json()['data']['labels']
        
        return classes
    
    def _save_underkill_images(self, images_path, project, task_name):
        underkill_folder = Path(UNDERKILLS_DATASETS_DIR, project, task_name)
        underkill_folder.mkdir(parents=True, exist_ok=True)
        for image_path in images_path:
            src = str(Path(VALIDATION_DATASETS_DIR, project, 'images', image_path).resolve())
            dst = str(Path(underkill_folder, image_path).resolve())
            shutil.copy(src, dst)
            
    def _validate_underkill_amount(self, project, task_name, images_amount, underkill_rate=0.01):
        underkills_amount = len(sorted(Path(UNDERKILLS_DATASETS_DIR, project, task_name).glob('*.jp*')))
        if (underkills_amount / images_amount) > underkill_rate:
            return False
        else:
            return True
        

    def _upload_category_record(self, images_path, group_type):
        data = {
            'images_path': images_path,
            'group_type': group_type
        }
        response = requests.post(urljoin(API_URL, 'category_record/upload'), json=data)
        
        if response.status_code == 200:
            return True
        else:
            return False
        

    def get_record_status(self) -> dict:
        self.logger.info('Get Record Status')
        iri_record:dict = requests.get(urljoin(API_URL, 'status/iri_record')).json()
        urd_record:dict = requests.get(urljoin(API_URL, 'status/urd_record')).json()
        
        iri_record_data = iri_record.get('data')
        urd_record_data = urd_record.get('data')
        
        if iri_record_data and urd_record_data:
            if iri_record_data.get('update_time') > urd_record_data.get('update_time'):
                return urd_record_data
            else:
                return iri_record_data
        elif not iri_record_data and urd_record_data:
            return urd_record_data
        elif iri_record_data and not urd_record_data:
            return iri_record_data

    def init(self, project, project_id, image_mode, site, line, group_type, start_date, end_date, smart_filter, **kwargs):
        images = self._get_images(image_mode, site, line, group_type, start_date, end_date, smart_filter, kwargs['images'])
        image_download_list = self._images_download_format(images['data']['images'], smart_filter)
        if self._download_images(image_download_list):
            barcode = self._get_barcode()
            zip_paths = self._get_download_zip_data()
            unzip_data_folder = self._unzip_data(zip_paths, barcode)
        
        if self._update_record(kwargs['__tablename__'], kwargs['id'], status=TRAINING_PLATFORM_RECORD_STATUS['INFERENCE_ON_GOING']):
            self._inference(unzip_data_folder, project, 'object_detection')
        task_id, task_name = self._upload_cvat(line, group_type, barcode, project_id, unzip_data_folder)
        self._update_record(kwargs['__tablename__'], kwargs['id'], task_id=task_id, task_name=task_name, status=TRAINING_PLATFORM_RECORD_STATUS['UPLOAD_IMAGE_WITH_LOG_FINISH'])
        
    def categorizing(self, **kwargs):
        ...
        
    def od_initialized(self, project, task, task_id, **kwargs):
        self.logger.info('OD Initialized')
        if 'object_detection' in TRAINING_FLOW[SITE][project]:
            self._update_record(kwargs['__tablename__'], kwargs['id'], status=TRAINING_PLATFORM_RECORD_STATUS['TRIGGER_TRAINING_FOR_OD'])
            self.logger.info('CVAT Download Datasets')
            auth = CVATScript.get_auth(CVAT_USERNAME, CVAT_PASSWORD)
            zip_file_path = CVATScript.download_task(task_id, task, auth)
            unzip_data_folder = self._unzip_data([zip_file_path], task)
            image_folder_path = str(Path(unzip_data_folder, 'JPEGImages'))
            xml_folder_path = str(Path(unzip_data_folder, 'Annotations'))
            classes = self._get_project_classes(kwargs['line'], kwargs['group_type'], project)
            label_folder_path = self._xml2yolo(classes, xml_folder_path)
            algorithm = TRAINING_FLOW[SITE][project]['object_detection']
            datasets = self._organize_to_training_dataset(project, task, algorithm, image_folder_path, label_folder_path)
            
            self._update_record(kwargs['__tablename__'], kwargs['id'], od_training='RUNNING', status=TRAINING_PLATFORM_RECORD_STATUS['TRAINING_FOR_OD'])
            self.logger.info('Training Model')
            trained_model_path = TrainModel.object_detection_train(algorithm, datasets, classes, project, task)
            
            validation_images = [str(images) for images in Path(VALIDATION_DATASETS_DIR, project, 'images').glob('*.jp*')]
            self.logger.info('Object Detection Inference')
            validation_images_inference_results = Inference.object_detection_inference(algorithm, trained_model_path, validation_images)
            self.logger.info('Validate underkill')
            underkill_images = Inference.validate_underkill(algorithm, project, validation_images_inference_results['data']['results'])
            if underkill_images:
                self._save_underkill_images(underkill_images, project, task)
                validate_answer = self._validate_underkill_amount(project, task, len(validation_images))
            else:
                validate_answer = True
            
            self.logger.info(f'Object Detection Training Model Validation is {validate_answer}')
        
    def cls_initialized(self, project, task, task_id, **kwargs):
        self.logger.info('OD Initialized')
        if 'classification' in TRAINING_FLOW[SITE][project]:
            ...
        
        
            
        
    def run(self) -> None:
        record:dict = self.get_record_status()
        
        for status in RECORD_STATUSES.get(record.get('__tablename__')):    
            method = getattr(self, status.lower())
            method(**record['data'])