import pickle
import shutil
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from app.services.model_service import Discriminator, Encoder, Generator
from xml.dom.minidom import Document
from glob import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.nn import functional as F
from app.config import CLASSIFICATION_INFERNCE_MODEL_DIR, GAN_INFERENCE_MODEL_DIR, OBJECT_DETECTION_PCIE_BODY_THRESHOLD, OBJECT_DETECTION_PCIE_CLASSIFICATION_CLASS_NAMES, OBJECT_DETECTION_PCIE_CLASSIFICATION_NGS, OBJECT_DETECTION_PCIE_PART_NUMBER, OBJECT_DETECTION_PCIE_PCIE_THRESHOLD, OBJECT_DETECTION_PCIE_PICKLE_MODEL_NAME, OBJECT_DETECTION_PCIE_WAYS, YOLO_INFERENCE_MODEL_DIR, YOLO_TRAIN_MODEL_DIR, YOLOV5_DIR, OBJECT_DETECTION_PCIE_CLASSIFICATION_MEAN, OBJECT_DETECTION_PCIE_CLASSIFICATION_STD
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

    @classmethod
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
        shutil.copyfile(image_path, os.path.abspath(os.path.join(dst_folder, image_name)))

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
    
    @classmethod
    def get_underkill_folder(cls, project, task_name):
        dst_folder = os.path.join(OBJECT_DETECTION_UNDERKILL_DATASETS_DIR, project, task_name)
        cls().check_folder(dst_folder)
        
        return dst_folder
    
    @classmethod
    def output_underkill_image(cls, image_path, underkill_folder):
        image_name = os.path.basename(image_path)
        dst_path = os.path.abspath(os.path.join(underkill_folder, image_name))
        shutil.copyfile(image_path, dst_path)

    @classmethod
    def check_validation_count(cls, images):
        return len(images)

    @classmethod
    def check_validation_result(cls, underkill_count, validation_count, underkill_rate=0.001):
        if underkill_count / validation_count > underkill_rate:
            return False
        else:
            return True


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


class PCIEInference(YOLOInference):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_classification_model_path(cls, project):
        return os.path.join(CLASSIFICATION_INFERNCE_MODEL_DIR, project, 'classification_model.pt')

    @classmethod
    def get_pinlocation_model_path(cls, project):
        return os.path.join(YOLO_INFERENCE_MODEL_DIR, project, OBJECT_DETECTION_PCIE_PICKLE_MODEL_NAME)

    @classmethod
    def get_classification_model(cls, model_path):
        return torch.load(model_path, map_location=cls().device())
    
    @classmethod
    def get_pinlocation_model(cls, model_path):
        return pickle.load(open(model_path, 'rb'))
    
    @classmethod
    def classification_inference(cls, image_path, model):
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=OBJECT_DETECTION_PCIE_CLASSIFICATION_MEAN, 
                std=OBJECT_DETECTION_PCIE_CLASSIFICATION_STD
            )
        ])
        # Read image and run prepro
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image = DataLoader(
            image_tensor,
            batch_size=1,
            shuffle=True
        )
        for inputs in image:
            inputs = inputs.to(cls().device())
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
        if OBJECT_DETECTION_PCIE_CLASSIFICATION_CLASS_NAMES[preds[0]] == 'OK':
            return True
        else:
            return False
        
    @classmethod
    def object_detection_inference(cls, model, pinlocation_model, image_path):
        result = model(image_path).pandas().xyxy[0]
        img_size = cv2.imread(image_path, 1).shape
        class_names = result['name'].unique()
        body_xmin = 0
        body_ymin = 0
        body_xmax = 0
        body_ymax = 0

        for way in OBJECT_DETECTION_PCIE_WAYS.keys():
            if way in os.path.basename(image_path):
                img_way = OBJECT_DETECTION_PCIE_WAYS[way]
        for comp in OBJECT_DETECTION_PCIE_PART_NUMBER.keys():
            if comp in os.path.basename(image_path):
                img_comp = OBJECT_DETECTION_PCIE_PART_NUMBER[comp]

        lcl_pcie = []
        lcl_body = []

        PCIE = result[result['name'] == 'PCIE']['confidence']
        for i, conf in enumerate(PCIE):
            if conf < OBJECT_DETECTION_PCIE_PCIE_THRESHOLD:
                lcl_pcie.append(PCIE.index[i])
        result = result.drop(index=lcl_pcie)

        BODY = result[result['name'] == 'BODY']['confidence']
        for i, conf in enumerate(BODY):
            if conf < OBJECT_DETECTION_PCIE_BODY_THRESHOLD:
                lcl_body.append(BODY.index[i])
        result = result.drop(index=lcl_body)

        for NG_type in OBJECT_DETECTION_PCIE_CLASSIFICATION_NGS:
            if NG_type in class_names:
                return False

        if 'PCIE' not in class_names:
            return False
        if 'BODY' not in class_names:
            return False

        BODY = result[result['name']=='BODY']
        if BODY.shape[0] >= 1:
            area_max = 1000000
            area_index = -1
            for i in range(BODY.shape[0]):
                item = BODY.iloc[i,:]
                area = (item['xmax'] - item['xmin']) * (item['ymax'] - item['ymin'])
                if area <= area_max:
                    area_max = area
                    area_index = BODY.index[i]
            sub_BODY = BODY.loc[area_index, :]

            body_xmin = sub_BODY['xmin']
            body_ymin = sub_BODY['ymin']
            body_xmax = sub_BODY['xmax']
            body_ymax = sub_BODY['ymax']

            if area_index == -1:
                return False
            elif (body_xmin < 10) or (body_ymin < 10) or (body_xmax > img_size[1]-5) or ((body_ymax > img_size[0]-5)):
                return False
        
        Pin_OK = result[result['name']=='PIN_OK']
        Pin_index = -1
        if Pin_OK.shape[0] >= 1:
            for i in Pin_OK.index:
                item = Pin_OK.loc[i,:]
                if (body_xmin - 5 < item['xmin']) & (body_ymin - 5 < item['ymin']) & (body_xmax + 5 > item['xmax']) & (body_ymax + 5 > item['ymax']):
                    Pin_index = i
                    break
        if Pin_index == -1:
            return False
        else:
            sub_Pin = Pin_OK.loc[Pin_index, :]
            Pin_xmin, Pin_ymin, Pin_xmax, Pin_ymax = sub_Pin['xmin'], sub_Pin['ymin'], sub_Pin['xmax'], sub_Pin['ymax']
            Pin_data = [[body_xmin, body_ymin, body_xmax, body_ymax, Pin_xmin, Pin_ymin, Pin_xmax, Pin_ymax, img_way, img_comp]]
            Pin_result = pinlocation_model.predict(Pin_data)[0]

            if Pin_result == -1:
                return 'NG'

        if 'W' in class_names:
            return True
        elif 'EXPOSURE' in class_names:
            return True
        else:
            return True
