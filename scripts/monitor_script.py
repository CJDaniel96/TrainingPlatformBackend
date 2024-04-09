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
import socket
import cv2
import pandas as pd


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
    def get_yolov5_loss(cls, project, task_name):
        result_csv = Path(MODELS_DIR, project, 'train', task_name, 'yolov5', 'results.csv')
        result = pd.read_csv(result_csv)
        
        return result.iloc[-1, -4]
    
    @classmethod
    def get_yolov5_accuracy(cls, underkills_count, project):
        validate_count = len(sorted(Path(VALIDATION_DATASETS_DIR, project, 'images').glob('*.jp*')))
        accuracy = (validate_count - underkills_count) / validate_count

        return accuracy

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
        ok_labels = requests.post(urljoin(API_URL, 'category/category_mapping/ok_labels'), params=params)
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
            
        save_dir = Path(MODELS_DIR, project, 'train', task_name)
        
        data = {
            'weights': weights,
            'data': data_yaml_path, 
            'cfg': model_yaml_path,
            'hyp': hyp_yaml_path, 
            'batch_size': YOLOV5_BATCH_SIZE, 
            'epochs': YOLOV5_EPOCHS, 
            'seed': YOLOV5_SEED, 
            'project': str(save_dir), 
            'name': 'yolov5'
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
        Path(DOWNLOADS_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(TRAIN_DATASETS_DIR).mkdir(parents=True, exist_ok=True)
        Path(UNDERKILLS_DATASETS_DIR).mkdir(parents=True, exist_ok=True)
    
    def _inference(self, unzip_data_folder, project, flow, model) -> None:
        self.logger.info('Model Inference...')
        if TRAINING_FLOW[SITE][project][flow] == 'yolov5':
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
        response = requests.get(urljoin(API_URL, 'info/image_pool'))
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
        response = requests.get(urljoin(API_URL, 'category/category_mapping/labels'), params=params)
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
        
    def _get_critical_ngs(self, group_type, project, task_name):
        underkill_folder = Path(UNDERKILLS_DATASETS_DIR, project, task_name)
        data = {
            'line_id': 'critical_NG',
            'group_type': group_type,
            'critical_ng_images': [f'{group_type}/ORG/{image.name}' for image in underkill_folder.glob('*.jp*')]
        }
        response = requests.post(urljoin(API_URL, 'category/critical_ng'), json=data)
        
        return response.json()['data']['critical_ngs']
        
        
    def _upload_crop_category_record(self, tablename:str, record_id, project, task_name, critical_ngs):
        crop_img_ids = []
        for critical_ng in critical_ngs:
            image_path = Path(UNDERKILLS_DATASETS_DIR, project, task_name, Path(critical_ng['image_path']).name)
            im = cv2.imread(str(image_path))
            h, w, c = im.shape
            data = {
                'finetune_id': record_id, 
                'image_id': critical_ng['img_id'], 
                'image_wide': w, 
                'image_hight': h, 
                'finetune_type': tablename.split('_')[0].upper()
            }
            
            response = requests.post(urljoin(API_URL, 'category/crop_category_record'), json=data)
            crop_img_ids.append(response.json()['data']['crop_img_id'])
        
        return crop_img_ids
        
    def _upload_od_training_info(self, task_id, comp_type, validate_result):
        data = {
            'task_id': task_id,
            'comp_type': comp_type, 
            'validate_result': validate_result
        }
        response = requests.post(urljoin(API_URL, 'info/od_training_info'), json=data)

        return response.json()['data']['status']
        
    def _get_od_training_info_val_status(self, task_id):
        params = {
            'task_id': task_id
        }
        response = requests.get(urljoin(API_URL, 'info/od_training_info'), params=params)

        return response.json()['data']['val_status']
    
    def _get_cls_training_info_val_status(self, task_id):
        params = {
            'task_id': task_id
        }
        response = requests.get(urljoin(API_URL, 'info/cls_training_info'), params=params)
        
        return response.json()['data']['val_status']
    
    def _upload_ai_model_information(self, model_path, group_type, val_status, record_id, tablename: str):
        ip_address = socket.gethostbyname(socket.gethostname())
        data = {
            'model_type': group_type,
            'model_path': model_path,
            'ip_address': ip_address,
            'verified_status': val_status,
            'finetune_id': record_id,
            'finetune_type': tablename.split('_')[0].upper()
        }
        response = requests.post(urljoin(API_URL, 'info/ai_model_info'), json=data)
        
        return response.json()['data']['model_id']
    
    def _get_trained_model_path(self, project, task_name):
        task_model_folder = Path(MODELS_DIR, project, 'train', task_name)
        models = sorted(task_model_folder.glob('*'))
        for model in models:
            if model.name == 'yolov5' and task_model_folder.joinpath(model.name, 'weights', 'best.pt').exists():
                return str(task_model_folder.joinpath(model.name, 'weights', 'best.pt'))
            elif task_model_folder.joinpath(model.name, 'best.pt').exists():
                return str(task_model_folder.joinpath(model.name, 'best.pt'))
            else:
                return ''
            
    def _inference_training_datasets(self, project, task_name, trained_model, flow):
        inference_training_datasets_path = Path(DOWNLOADS_DATA_DIR, task_name + '_inference')
        training_images = Path(DOWNLOADS_DATA_DIR, task_name, 'JPEGImages')
        if training_images.exists():
            shutil.copytree(str(training_images), str(inference_training_datasets_path), dirs_exist_ok=True)
            self._inference(str(inference_training_datasets_path), project, flow, trained_model)
            
        return str(inference_training_datasets_path), task_name.split('_')[-1] + '_inference'

    def _upload_ai_model_performance(self, model_id, project, task_name, inference_task_id, crop_image_ids): 
        if 'object_detection' in TRAINING_FLOW[SITE][project]:
            loss = Inference.get_yolov5_loss(project, task_name)
            accuracy = Inference.get_yolov5_accuracy(len(crop_image_ids), project)
        data = {
            'model_id': model_id,
            'metrics_result': {
                'LOSS': loss,
                'ACCURACY': accuracy,
                'FALSE_NEGATIVE_NUM': len(crop_image_ids),
                'FALSE_POSITIVE_NUM': 0,
                'FINE_TUNE_CONFIRM_TASK_ID': inference_task_id
            },
            'false_negative_imgs': {
                'NUMBER_OF_IMG': len(crop_image_ids),
                'CROP_IMG_ID_LIST': crop_image_ids
            },
            'false_positive_imgs': {
                'NUMBER_OF_IMG': 0
            }
        }
        response = requests.post(urljoin(API_URL, 'info/ai_model_performance'), json=data)
        
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
        
        if not all(value is None for value in iri_record_data.values()) and not all(value is None for value in urd_record_data.values()):
            if iri_record_data.get('update_time') > urd_record_data.get('update_time'):
                return urd_record_data
            else:
                return iri_record_data
        elif not all(value is None for value in iri_record_data.values()) and all(value is None for value in urd_record_data.values()):
            return iri_record_data
        elif all(value is None for value in iri_record_data.values()) and not all(value is None for value in urd_record_data.values()):
            return urd_record_data
        else:
            return {}

    def init(self, project, project_id, image_mode, site, line, group_type, start_date, end_date, smart_filter, **kwargs):
        images = self._get_images(image_mode, site, line, group_type, start_date, end_date, smart_filter, kwargs['images'])
        image_download_list = self._images_download_format(images['data']['images'], smart_filter)
        if self._download_images(image_download_list):
            barcode = self._get_barcode()
            zip_paths = self._get_download_zip_data()
            unzip_data_folder = self._unzip_data(zip_paths, barcode)
        
        if self._update_record(kwargs['__tablename__'], kwargs['id'], status=TRAINING_PLATFORM_RECORD_STATUS['INFERENCE_ON_GOING']):
            model = str(Path(MODELS_DIR, project, 'inference', 'yolo_model.pt').resolve())
            self._inference(unzip_data_folder, project, 'object_detection', model)
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
                
            self.logger.info(self._upload_od_training_info(task_id, kwargs['group_type'], validate_answer))
            self._update_record(kwargs['__tablename__'], kwargs['id'], status=TRAINING_PLATFORM_RECORD_STATUS['FINISH_FOR_OD'])
            self.logger.info(f'Object Detection Training Model Validation is {validate_answer}')
        
    def cls_initialized(self, project, task, task_id, **kwargs):
        self.logger.info('CLS Initialized')
        self._update_record(kwargs['__tablename__'], kwargs['id'], status=TRAINING_PLATFORM_RECORD_STATUS['TRIGGER_TRAINING_FOR_CLS'])
        inference_task_id = None
        if 'classification' in TRAINING_FLOW[SITE][project]:
            ...
        if 'object_detection' in TRAINING_FLOW[SITE][project]:
            self.logger.info('Inference Training Datasets')
            model_path = self._get_trained_model_path(project, task)
            inference_training_datasets_path, inference_barcode = self._inference_training_datasets(project, task, model_path, 'object_detection')
            inference_task_id, _ = self._upload_cvat(
                kwargs['line'], kwargs['group_type'], inference_barcode, kwargs['project_id'], inference_training_datasets_path
            )

        self.logger.info('Upload Crop Category Record')
        critical_ngs = self._get_critical_ngs(kwargs['group_type'], project, task)
        crop_img_ids = self._upload_crop_category_record(kwargs['__tablename__'], kwargs['id'], project, task, critical_ngs)
        
        self.logger.info('Check Validate Status')
        od_val_status = self._get_od_training_info_val_status(task_id)
        cls_val_status = self._get_cls_training_info_val_status(task_id)
        if od_val_status == 'APPROVE' and cls_val_status == 'APPROVE':
            val_status = 'APPROVE'
        else:
            val_status = 'FAIL'
        model_path = self._get_trained_model_path(project, task)
        self.logger.info('Upload AI Model Information')
        model_id = self._upload_ai_model_information(model_path, kwargs['group_type'], val_status, kwargs['id'], kwargs['__tablename__'])
        self.logger.info('Upload AI Model Performance')
        self._upload_ai_model_performance(model_id, project, task, inference_task_id, crop_img_ids)
        self._update_record(kwargs['__tablename__'], kwargs['id'], status=TRAINING_PLATFORM_RECORD_STATUS['FINISHED'])
        
    def run(self) -> None:
        while True:
            try:
                record:dict = self.get_record_status()

                if record and record.get('status') in RECORD_STATUSES.get(record.get('__tablename__')):   
                    method = getattr(self, record.get('status').lower())
                    method(**record)
            except Exception as e:
                self.logger.error(e)
                continue