import shutil
from app.services.model_service import Discriminator, Encoder, Generator
from app.config import GAN_INFERENCE_MODEL_DIR, YOLO_INFERENCE_MODEL_DIR, YOLO_TRAIN_MODEL_DIR, YOLOV5_DIR
from xml.dom.minidom import Document
from glob import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import torch
import torch.nn as nn
import cv2
import numpy as np

from data.config import OBJECT_DETECTION_INFERENCE_DATASETS_DIR, OBJECT_DETECTION_UNDERKILL_DATASETS_DIR, OBJECT_DETECTION_VALIDATION_DATASETS_DIR


class YOLOInference:
    @classmethod
    def get_config(cls, project, config_name='config.ini'):
        config_file = os.path.join(YOLO_INFERENCE_MODEL_DIR, project, config_name)
        if os.path.exists(config_file):
            return config_file
        else:
            config_name = '*.ini'
            config_files = glob(os.path.join(YOLO_INFERENCE_MODEL_DIR, project, config_name))
            if len(config_files) == 1:
                return config_files[0]
            elif len(config_files) > 1:
                raise Exception('No only one config file!')
            else:
                raise Exception('No config file!')
            
    @classmethod
    def check_folder(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    @classmethod
    def output_xml(cls, image_size, image_name, defect_name, defect_position, save_xml_path):
        # Create empty root
        doc = Document()
        # Create root node
        root = doc.createElement('annotation')
        doc.appendChild(root)
        # Create folder node
        folder = doc.createElement('folder')
        folder_text = doc.createTextNode(save_xml_path)
        folder.appendChild(folder_text)
        root.appendChild(folder)
        # Create filename node
        filename = doc.createElement('filename')
        filename_text = doc.createTextNode(image_name)
        filename.appendChild(filename_text)
        root.appendChild(filename)
        # Create path node
        path = doc.createElement('path')
        path_text = doc.createTextNode(save_xml_path + image_name)
        path.appendChild(path_text)
        root.appendChild(path)
        # Create image size node
        size = doc.createElement('size')
        width = doc.createElement('width')
        width_text = doc.createTextNode(str(image_size[1]))
        width.appendChild(width_text)
        size.appendChild(width)
        height = doc.createElement('height')
        height_text = doc.createTextNode(str(image_size[0]))
        height.appendChild(height_text)
        size.appendChild(height)
        depth = doc.createElement('depth')
        depth_text = doc.createTextNode(str(image_size[2]))
        depth.appendChild(depth_text)
        size.appendChild(depth)
        root.appendChild(size)
        # Create object node
        if defect_name != None or defect_position != None:
            #import pdb;pdb.set_trace()
            for name_list, box_list in zip(defect_name, defect_position):
                xml_object = doc.createElement('object')
                # defect name
                name = doc.createElement('name')
                name_text = doc.createTextNode(name_list)
                name.appendChild(name_text)
                xml_object.appendChild(name)
                # bndbox
                bndbox = doc.createElement('bndbox')
                # xmin
                xmin = doc.createElement('xmin')
                xmin_text = doc.createTextNode(str(int(box_list[0])))
                xmin.appendChild(xmin_text)
                bndbox.appendChild(xmin)
                # ymin
                ymin = doc.createElement('ymin')
                ymin_text = doc.createTextNode(str(int(box_list[1])))
                ymin.appendChild(ymin_text)
                bndbox.appendChild(ymin)
                # xmax
                xmax = doc.createElement('xmax')
                xmax_text = doc.createTextNode(str(int(box_list[2])))
                xmax.appendChild(xmax_text)
                bndbox.appendChild(xmax)
                # ymax
                ymax = doc.createElement('ymax')
                ymax_text = doc.createTextNode(str(int(box_list[3])))
                ymax.appendChild(ymax_text)
                bndbox.appendChild(ymax)
                xml_object.appendChild(bndbox)
                root.appendChild(xml_object)
        xml_name = os.path.splitext(image_name)[0] + '.xml'
        with open(os.path.join(save_xml_path, xml_name), 'w') as xml:
            doc.writexml(xml, indent='\t', newl='\n', addindent='\t', encoding='utf-8')

    @classmethod
    def get_inference_model_path(cls, project, model_name='yolo_model.pt'):
        model_file = os.path.join(YOLO_INFERENCE_MODEL_DIR, project, model_name)
        if not os.path.exists(model_file):
            raise Exception('No model file or model file name is not project name (e.x. yolo_model.pt)!')
        else:
            return model_file
        
    @classmethod
    def get_train_model_path(cls, project, task_name):
        return os.path.join(YOLO_TRAIN_MODEL_DIR, project, task_name, 'weights', 'best.pt')

    @property
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def load_model(cls, path, conf=0.25):
        model = torch.hub.load(YOLOV5_DIR, 'custom', path=path, source='local')
        model.conf = conf
        model.cuda()
        return model
    
    @classmethod
    def get_validation_images(cls, project):
        return glob(os.path.join(OBJECT_DETECTION_VALIDATION_DATASETS_DIR, project, '**', '*.jpg'), recursive=True)
    
    @classmethod
    def get_train_images(cls, train_data_folder):
        return glob(os.path.join(train_data_folder, '**', '*.jpg'), recursive=True)

    @classmethod
    def predict(cls, model, image_path):
        return model(image_path).pandas().xyxy[0]
    
    @classmethod
    def read_image(cls, image_path):
        return cv2.imread(image_path)

    @classmethod
    def get_image_name(cls, image_path):
        return os.path.basename(image_path)

    @classmethod
    def get_image_size(cls, image):
        return image.shape
    
    @classmethod
    def get_class_names(cls, result):
        return list(result['name'].values)
    
    @classmethod
    def get_label_positions(cls, result):
        return list(result[['xmin', 'ymin', 'xmax', 'ymax']].values)
    
    @classmethod
    def save_inference_image(cls, image_path, project, task_name):
        image_name = os.path.basename(image_path)
        dst_folder = os.path.join(OBJECT_DETECTION_INFERENCE_DATASETS_DIR, project, task_name, 'images')
        cls().check_folder(dst_folder)
        shutil.copyfile(image_path, os.path.join(dst_folder, image_name))

        return dst_folder
    
    @classmethod
    def get_inference_folder(cls, project, task_name):
        dst_folder = os.path.join(OBJECT_DETECTION_INFERENCE_DATASETS_DIR, project, task_name)
        cls().check_folder(dst_folder)
        
        return dst_folder

    @classmethod
    def get_inference_xml_folder(cls, project, task_name):
        dst_folder = os.path.join(OBJECT_DETECTION_INFERENCE_DATASETS_DIR, project, task_name, 'xmls')
        cls().check_folder(dst_folder)

        return dst_folder


class CHIPRCInference(YOLOInference):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_generator_model_path(cls, project):
        return os.path.join(GAN_INFERENCE_MODEL_DIR, project, 'generator')

    @classmethod
    def get_discriminator_model_path(cls, project):
        return os.path.join(GAN_INFERENCE_MODEL_DIR, project, 'discriminator')
    
    @classmethod
    def get_encoder_model_path(cls, project):
        return os.path.join(GAN_INFERENCE_MODEL_DIR, project, 'encoder')

    @classmethod
    def get_generator_model(cls, path, img_size=256, latent_dim=100, channels=3):
        generator = Generator(img_size, latent_dim, channels)
        generator.load_state_dict(torch.load(path))
        generator.to(cls().device()).eval()

        return generator

    @classmethod
    def get_discriminator_model(cls, path, img_size=256, channels=3):
        discriminator = Discriminator(img_size, channels)
        discriminator.load_state_dict(torch.load(path))
        discriminator.to(cls().device()).eval()

        return discriminator
    
    @classmethod
    def get_encoder_model(cls, path, img_size=256, latent_dim=100, channels=3):
        encoder = Encoder(img_size, latent_dim, channels)
        encoder.load_state_dict(torch.load(path))
        encoder.to(cls().device()).eval()

        return encoder
    
    @classmethod
    def get_criterion(cls):
        return nn.MSELoss()
    
    @classmethod
    def get_transform(cls, img_size=256, channels=3):
        pipeline = [
            transforms.Resize([img_size]*2),
            transforms.RandomHorizontalFlip()
        ]
        if channels == 1:
            pipeline.append(transforms.Grayscale())
        pipeline.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*channels, [0.5]*channels)
        ])
        return transforms.Compose(pipeline)
    
    @classmethod
    def yolo_predict(cls, model, image_path, chiprc_threshold=0.5, target='ChipRC'):
        result = model(image_path).pandas().xyxy[0]
        class_names = result['name'].unique()
        chiprcs = result[result['name'] == target]
        lcl_chiprc = list(filter(lambda x: x < chiprc_threshold ,list(result[result['name']=='ChipRC']['confidence'])))
        chiprc_count = 0

        if target in class_names:
            chiprc_count = result['name'].value_counts()[target]

        if target not in class_names or len(class_names) > 1:   
            return False, chiprcs   
        elif type(chiprc_count) == np.int64 and chiprc_count > 1:
            return False, chiprcs
        elif len(lcl_chiprc) > 0:
            return False, chiprcs
        else:
            return True, chiprcs
    
    @classmethod
    def vae_predict(cls, image_path, chiprcs, transform, generator, discriminator, encoder, criterion, kappa=1.0):
        image = cv2.imread(image_path)
        for i in range(chiprcs.shape[0]):
            chiprc = chiprcs.iloc[i:i+1, :]
            xmin = list(chiprc['xmin'])[0]
            ymin = list(chiprc['ymin'])[0]
            xmax = list(chiprc['xmax'])[0]
            ymax = list(chiprc['ymax'])[0]
            crop = image[int(ymin):int(ymax), int(xmin):int(xmax), ::(1)]
            crop = crop[:, :, ::-1]

            if crop.shape[0] > crop.shape[1]:
                crop = np.rot90(crop, 1)
            
            crop = Image.fromarray(crop, 'RGB')
            image_tensor = transform(crop)
            image_tensor = image_tensor.unsqueeze(0)
            test_dataloader = DataLoader(
                image_tensor,
                batch_size=1,
                shuffle=True
            )

            for img in test_dataloader:
                real_img = img.to(cls().device())

                real_z = encoder(real_img)
                fake_img = generator(real_z)

                real_feature = discriminator.forward_features(real_img)
                fake_feature = discriminator.forward_features(fake_img)

                img_distance = criterion(fake_img, real_img)
                loss_feature = criterion(fake_feature, real_feature)
                anomaly_score = img_distance + kappa * loss_feature
                anomaly_score = float(anomaly_score)

                if anomaly_score > 0.2:
                    return False
                else:
                    return True
                
    @classmethod
    def output_underkill_image(cls, image_path, project, task_name):
        image_name = os.path.basename(image_path)
        dst_path = os.path.join(OBJECT_DETECTION_UNDERKILL_DATASETS_DIR, project, task_name, image_name)
        shutil.copyfile(image_path, dst_path)