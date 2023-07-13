import shutil
from app.config import YOLO_TRAIN_MODEL_DIR
from data.config import CLASSIFICATION_BASICLINE_DATASETS_DIR, CLASSIFICATION_TRAIN_DATASETS_DIR, CLASSIFICATION_VALIDATION_DATASETS_DIR, OBJECT_DETECTION_BASICLINE_DATASETS_DIR, OBJECT_DETECTION_TRAIN_DATASETS_DIR, OBJECT_DETECTION_VALIDATION_DATASETS_DIR, OBJECT_DETECTION_UNDERKILL_DATASETS_DIR, CLASSIFICATION_UNDERKILL_DATASETS_DIR, ORIGIN_DATASETS_FOLDER_PROFIX, ORIGIN_DATASETS_DIR, YOLO_TRAIN_DATA_YAML_DIR, YOLO_TRAIN_HYPS_YAML_DIR, YOLO_TRAIN_MODELS_YAML_DIR
from datetime import datetime
from glob import glob
import os
import zipfile
import socket
import pandas as pd


class UnderkillDataProcessing:
    @classmethod
    def get_ip_address(cls):
        return socket.gethostbyname(socket.gethostname())

    @classmethod
    def get_object_detection_underkill_path(cls, project, task_name):
        underkill_folder = os.path.join(OBJECT_DETECTION_UNDERKILL_DATASETS_DIR, project, task_name)
        underkills = glob(os.path.join(underkill_folder, '*.jpg')) + glob(os.path.join(underkill_folder, '*.jpeg'))

        return underkills

    @classmethod
    def get_classification_underkill_path(cls, project, task_name):
        underkill_folder = os.path.join(CLASSIFICATION_UNDERKILL_DATASETS_DIR, project, task_name)
        underkills = glob(os.path.join(underkill_folder, '*.jpg')) + glob(os.path.join(underkill_folder, '*.jpeg'))

        return underkills
    
    @classmethod
    def get_object_detection_validations(cls, project):
        validation_folder = os.path.join(OBJECT_DETECTION_VALIDATION_DATASETS_DIR, project, 'images')
        validations = glob(os.path.join(validation_folder, '*.jpg')) + glob(os.path.join(validation_folder, '*.jpeg'))

        return validations

    @classmethod
    def get_classification_validations(cls, project):
        validation_folder = os.path.join(CLASSIFICATION_VALIDATION_DATASETS_DIR, project, 'images')
        validations = glob(os.path.join(validation_folder, '*.jpg')) + glob(os.path.join(validation_folder, '*.jpeg'))

        return validations
    
    @classmethod
    def check_model_pass_or_fail(cls, underkills, validations, underkill_rate=0.01):
        if len(underkills) / len(validations) > underkill_rate:
            return False
        else:
            return True
    
    @classmethod
    def get_loss(cls, project, task_name):
        result_csv = os.path.join(YOLO_TRAIN_MODEL_DIR, project, task_name, 'results.csv')
        result = pd.read_csv(result_csv)
        
        return result.iloc[-1, -4]
    
    @classmethod
    def get_accuracy(cls, underkills, project):
        validate_count = len(os.listdir(os.path.join(OBJECT_DETECTION_VALIDATION_DATASETS_DIR, project, 'images')))
        underkill_count = len(underkills)
        accuracy = (validate_count - underkill_count) / validate_count

        return accuracy
    
    @classmethod
    def get_metrics_result(cls, loss, accuracy, crop_image_ids, training_datasets_inferece_task_id):
        return {
            "LOSS": loss,
            "ACCURACY": accuracy,
            "FALSE_NEGATIVE_NUM": len(crop_image_ids),
            "FALSE_POSITIVE_NUM": 0, 
            "FINE_TUNE_CONFIRM_TASK_ID": training_datasets_inferece_task_id
        }

    @classmethod
    def get_false_negative_images(cls, crop_image_ids):
        return {
            "NUMBER_OF_IMG": len(crop_image_ids),
            "CROP_IMG_ID_LIST": crop_image_ids
        }

    @classmethod
    def get_false_positive_images(cls):
        return {
            "NUMBER_OF_IMG": 0
        }


class OriginDataProcessing:
    @classmethod
    def get_serial_number(cls):
        return datetime.now().strftime('%Y%m%d%H%M%S')
    
    @classmethod
    def get_origin_image_folder(cls, serial_number):
        origin_image_folder_name = ORIGIN_DATASETS_FOLDER_PROFIX + '_' + serial_number
        return os.path.join(ORIGIN_DATASETS_DIR, origin_image_folder_name)

    @classmethod
    def unzip_origin_data(cls, dst_folder):
        org_image_folder = os.path.join(dst_folder, 'images')

        for zip in glob(os.path.join(ORIGIN_DATASETS_DIR, '*.zip')):
            zip_file = zipfile.ZipFile(zip, 'r')
            zip_file.extractall(org_image_folder)
            zip_file.close()
            os.remove(zip)

        return org_image_folder
    
    @classmethod
    def zip_xml_data(cls, xml_folder):
        zip_file = '{}.zip'.format(xml_folder)
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for xml in glob(os.path.join(xml_folder, '*.xml')):
                zf.write(xml, arcname=os.path.basename(xml))

        return zip_file
    

class TrainDataProcessing:
    def __init__(self) -> None:
        pass

    @classmethod
    def get_abspath(cls, path):
        return os.path.abspath(path)

    @classmethod
    def check_zip_file(cls, task_zip_file):
        if zipfile.is_zipfile(task_zip_file):
            zip_file = zipfile.ZipFile(task_zip_file, 'r')
            zip_file.extractall()

            return True
        else:
            return False
        
    @classmethod
    def makedirs(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @classmethod
    def clear_zip_file(cls, zip_file):
        if os.path.exists(zip_file):
            os.remove(zip_file)


class ObjectDetectionTrainDataProcessing(TrainDataProcessing):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_object_detection_train_data_folder(cls, project, task_name):
        return os.path.join(OBJECT_DETECTION_TRAIN_DATASETS_DIR, project, task_name)

    @classmethod
    def get_object_detection_train_data(cls, train_data_folder):
        train_data_dir = 'obj_train_data'
        train_images_folder = os.path.join(train_data_folder, 'images', 'train')
        train_labels_folder = os.path.join(train_data_folder, 'labels', 'train')
        cls().makedirs(train_images_folder)
        cls().makedirs(train_labels_folder)

        for image in glob(os.path.join(train_data_dir, '*.jpg')) + glob(os.path.join(train_data_dir, '*.jpeg')):
            shutil.copyfile(image, os.path.abspath(os.path.join(train_images_folder, os.path.basename(image))))
        for label in glob(os.path.join(train_data_dir, '*.txt')):
            shutil.copyfile(label, os.path.abspath(os.path.join(train_labels_folder, os.path.basename(label))))

        if os.path.isdir(train_data_dir):
            shutil.rmtree(train_data_dir)
        if os.path.isfile('obj.data'):
            os.remove('obj.data')
        if os.path.isfile('obj.names'):
            os.remove('obj.names')
        if os.path.isfile('train.txt'):
            os.remove('train.txt')

        return

    @classmethod
    def merge_object_detection_basicline_data(cls, train_data_folder, project):
        basicline_dataset = os.path.abspath(os.path.join(OBJECT_DETECTION_BASICLINE_DATASETS_DIR, project))
        train_images_folder = os.path.join(train_data_folder, 'images', 'train')
        train_labels_folder = os.path.join(train_data_folder, 'labels', 'train')

        train_images_basicline_dataset = glob(os.path.join(basicline_dataset, 'images', 'train', '*.jpg')) + glob(os.path.join(basicline_dataset, 'images', 'train', '*.jpeg'))
        train_labels_basicline_dataset = glob(os.path.join(basicline_dataset, 'labels', 'train', '*.txt'))

        if train_images_basicline_dataset and train_labels_basicline_dataset:
            for image in train_images_basicline_dataset:
                try:
                    shutil.copyfile(image, os.path.abspath(os.path.join(train_images_folder, os.path.basename(image))))
                except FileNotFoundError:
                    long_src_path = '\\\\?\\' + image
                    long_dst_path = '\\\\?\\' + os.path.abspath(os.path.join(train_images_folder, os.path.basename(image)))
                    shutil.copyfile(long_src_path, long_dst_path)
            for label in train_labels_basicline_dataset:
                try:
                    shutil.copyfile(label, os.path.abspath(os.path.join(train_labels_folder, os.path.basename(label))))
                except FileNotFoundError:
                    long_src_path = '\\\\?\\' + image
                    long_dst_path = '\\\\?\\' + os.path.abspath(os.path.join(train_labels_folder, os.path.basename(label)))
                    shutil.copyfile(long_src_path, long_dst_path)
            
            try:
                shutil.copytree(os.path.join(basicline_dataset, 'images', 'val'), os.path.join(train_data_folder, 'images', 'val'))
            except:
                if os.path.exists(os.path.join(train_data_folder, 'images', 'val')):
                    shutil.rmtree(os.path.join(train_data_folder, 'images', 'val'))
                shutil.copytree('\\\\?\\' + os.path.join(basicline_dataset, 'images', 'val'), '\\\\?\\' + os.path.abspath(os.path.join(train_data_folder, 'images', 'val')))
            try:
                shutil.copytree(os.path.join(basicline_dataset, 'labels', 'val'),  os.path.join(train_data_folder, 'labels', 'val'))
            except:
                if os.path.exists(os.path.join(train_data_folder, 'labels', 'val')):
                    shutil.rmtree(os.path.join(train_data_folder, 'labels', 'val'))
                shutil.copytree('\\\\?\\' + os.path.join(basicline_dataset, 'labels', 'val'), '\\\\?\\' + os.path.abspath(os.path.join(train_data_folder, 'labels', 'val')))

    @classmethod
    def write_data_yaml(cls, project, class_names, train_data_folder):
        train_images_data_path = os.path.join(train_data_folder, 'images', 'train')
        train_labels_data_path = os.path.join(train_data_folder, 'images', 'val')
        data_yaml_path = os.path.join(YOLO_TRAIN_DATA_YAML_DIR, project + '.yaml')

        with open(data_yaml_path, 'w') as f:
            f.writelines(f'names: {class_names}\n')
            f.writelines(f'nc: {len(class_names)}\n')
            f.writelines(f'train: {os.path.abspath(train_images_data_path)}\n')
            f.writelines(f'val: {os.path.abspath(train_labels_data_path)}\n')

        return data_yaml_path
    
    @classmethod
    def get_models_yaml(cls, project):
        return os.path.join(YOLO_TRAIN_MODELS_YAML_DIR, project + '.yaml')
    
    @classmethod
    def get_hyps_yaml(cls, project):
        return os.path.join(YOLO_TRAIN_HYPS_YAML_DIR, project + '.yaml')
    

class ClassificationTrainDataProcessing(TrainDataProcessing):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_classification_train_data_folder(cls, project, task_name):
        return os.path.join(CLASSIFICATION_TRAIN_DATASETS_DIR, project, task_name)
    
    @classmethod
    def get_classification_train_data(cls, train_data_folder):
        train_data_dir = 'default'

        for class_folder in os.listdir(train_data_dir):
            shutil.copytree(os.path.join(train_data_dir, class_folder), os.path.join(train_data_folder, 'train', class_folder))

        if os.path.isdir(train_data_dir):
            shutil.rmtree(train_data_dir)
        if os.path.isdir('bb_landmark'):
            shutil.rmtree('bb_landmark')
        if os.path.isfile('labels.txt'):
            os.remove('labels.txt')

    @classmethod
    def merge_classification_basicline_data(cls, train_data_folder, project):
        basicline_dataset = os.path.abspath(os.path.join(CLASSIFICATION_BASICLINE_DATASETS_DIR, project))
        images_basicline_folder = os.listdir(basicline_dataset, 'images')

        if images_basicline_folder:
            for dataset in os.listdir(images_basicline_folder):
                for folder in os.listdir(images_basicline_folder, dataset):
                    try:
                        shutil.copytree(
                            os.path.join(images_basicline_folder, dataset, folder), 
                            os.path.join(train_data_folder, dataset, folder), 
                            dirs_exist_ok=True
                        )
                    except:
                        long_src_path = '\\\\?\\' + os.path.join(images_basicline_folder, dataset, folder)
                        long_dst_path = '\\\\?\\' + os.path.join(train_data_folder, dataset, folder)
                        shutil.copytree(long_src_path, long_dst_path, dirs_exist_ok=True)


class CategorizeDataProcessing:
    @classmethod
    def get_images(cls, train_data_folder):
        return glob(os.path.join(train_data_folder, 'images', 'train', '*.jpg')) + glob(os.path.join(train_data_folder, 'images', 'train', '*.jpeg'))
    
    @classmethod
    def get_object_detection_basicline_image_names(cls, project):
        return os.listdir(os.path.join(OBJECT_DETECTION_BASICLINE_DATASETS_DIR, project, 'images', 'train'))

    @classmethod
    def get_image_txt_file(cls, image_path, txt_type='.txt'):
        txt_file_path = os.path.splitext(image_path.replace('images', 'labels'))[0] + txt_type
        return txt_file_path

    @classmethod
    def check_image_result(cls, ok_category, class_dict, txt_file):
        with open(txt_file, 'r') as f:
            for line_data in f.readlines():
                class_number = int(line_data.split(' ')[0])
                if list(class_dict.keys())[list(class_dict.values()).index(0)] not in ok_category:
                    return 'NG'
            return 'OK'