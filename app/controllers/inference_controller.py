from app.services.inference_service import YOLOInference
from glob import glob
import shutil
import os
import numpy as np


class InferenceController:
    def __init__(self) -> None:
        pass

    @classmethod
    def inference(cls):...


class YOLOInferenceController(InferenceController):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def inference(cls, data, project):
        org_data_folder = os.path.dirname(data)
        model_file = YOLOInference.get_inference_model_path(project)
        model = YOLOInference.load_model(model_file)
        xml_folder = YOLOInference.check_folder(os.path.join(org_data_folder, 'xml'))

        if os.path.isfile(data):
            images = glob(data)
        elif os.path.isdir(data):
            if glob(os.path.join(data, '*.jpg')):
                images = glob(os.path.join(data, '*.jpg'))
            elif glob(os.path.join(data, '*.jpeg')):
                images = glob(os.path.join(data, '*.jpeg'))

        for image_path in images:
            result = YOLOInference.predict(model, image_path)
            image = YOLOInference.read_image(image_path)
            image_size = image.shape
            image_name = os.path.basename(image_path)
            YOLOInference.output_xml(
                image_size, 
                image_name, 
                list(result['name'].values), 
                list(result[['xmin', 'ymin', 'xmax', 'ymax']].values), 
                xml_folder
            )

        return xml_folder
    
    @classmethod
    def get_train_model_path(cls, project, task_name):
        return YOLOInference.get_train_model_path(project, task_name)

    @classmethod
    def train_dataset_inference(cls, project, task_name, model_file, train_data_folder):
        model = YOLOInference.load_model(model_file)
        images = YOLOInference.get_train_images(train_data_folder)

        for image_path in images:
            result = YOLOInference.predict(model, image_path)
            image_name = YOLOInference.get_image_name(image_path)
            image = YOLOInference.read_image(image_path)
            image_size = YOLOInference.get_image_size(image)
            class_names = YOLOInference.get_class_names(result)
            label_positions = YOLOInference.get_label_positions(result)

            inference_image_folder = YOLOInference.save_inference_image(image_path, project, task_name)
            inferece_xml_folder = YOLOInference.get_inference_xml_folder(project, task_name)
            YOLOInference.output_xml(image_size, image_name, class_names, label_positions, inferece_xml_folder)

        return inference_image_folder, inferece_xml_folder


class CHIPRCInferenceController(InferenceController):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def yolo_inference(cls, results, threshold=0.5, target='ChipRC'):
        classes_list = results['name'].unique()
        chiprc_count = 0
        lcl_chiprc = list(filter(
            lambda x: x < threshold, 
            list(results[results['name'] == target]['confidence'])
        ))
        
        if target in classes_list:
            chiprc_count = results['name'].value_counts()[target]
        
        if target not in classes_list or len(classes_list) > 1:
            return 'NG'
        elif type(chiprc_count) == np.int64 and chiprc_count > 1:
            return 'NG'
        elif len(lcl_chiprc) > 0:
            return 'NG'
        else:
            return 'OK'
                    
    @classmethod
    def predict(cls, model, image_path, transform, generator, discriminator, encoder, criterion, target='ChipRC'):
        results = cls().yolo_inference(model, image_path)
        pred = cls().yolo_predict(results)
        if pred == 'OK':
            chiprcs = results[results['name'] == target]
            image = cls().read_image(image_path)
            pred = cls().vae_predict(image, chiprcs, transform, generator, discriminator, encoder, criterion)
        
        return pred