import pickle
import re
import shutil
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from app.metric_learning.inference import extract_query_features
from app.metric_learning.utils import read_mean_std, setup_seed
from app.services.model_service import Discriminator, Encoder, Generator
from xml.dom.minidom import Document
from glob import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.nn import functional as F
from app.config import MODEL_DIRS, OBJECT_DETECTION_PCIE, UNDERKILL_RATE, YOLOV5
from data.config import CLASSIFICATION_TRAIN_DATASETS_DIR, CLASSIFICATION_UNDERKILL_DATASETS_DIR, CLASSIFICATION_VALIDATION_DATASETS_DIR, OBJECT_DETECTION_INFERENCE_DATASETS_DIR, OBJECT_DETECTION_UNDERKILL_DATASETS_DIR, OBJECT_DETECTION_VALIDATION_DATASETS_DIR


class YOLOInference:
    @classmethod
    def get_config(cls, project, config_name='config.ini'):
        config_file = os.path.join(MODEL_DIRS['YOLO_INFERENCE_MODEL_DIR'], project, config_name)
        if os.path.exists(config_file):
            return config_file
        else:
            config_name = '*.ini'
            config_files = glob(os.path.join(MODEL_DIRS['YOLO_INFERENCE_MODEL_DIR'], project, config_name))
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
        model_file = os.path.join(MODEL_DIRS['YOLO_INFERENCE_MODEL_DIR'], project, model_name)
        if not os.path.exists(model_file):
            return 
        else:
            return model_file
        
    @classmethod
    def get_train_model_path(cls, project, task_name):
        return os.path.join(MODEL_DIRS['YOLO_TRAIN_MODEL_DIR'], project, task_name, 'weights', 'best.pt')

    @classmethod
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def load_model(cls, path, conf=0.25):
        model = torch.hub.load(YOLOV5['YOLOV5_DIR'], 'custom', path=path, source='local')
        model.conf = conf
        model.cuda()
        return model
    
    @classmethod
    def get_validation_images(cls, project):
        return glob(os.path.join(OBJECT_DETECTION_VALIDATION_DATASETS_DIR, project, '**', '*.jpg'), recursive=True) + glob(os.path.join(OBJECT_DETECTION_VALIDATION_DATASETS_DIR, project, '**', '*.jpeg'), recursive=True)
    
    @classmethod
    def get_train_images(cls, train_data_folder):
        return glob(os.path.join(train_data_folder, '**', '*.jpg'), recursive=True) + glob(os.path.join(train_data_folder, '**', '*.jpeg'), recursive=True)

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
    def check_validation_result(cls, underkill_count, validation_count, underkill_rate=0.01):
        if underkill_count / validation_count > underkill_rate:
            return False
        else:
            return True


class MobileNetGANInference:
    def __init__(self) -> None:
        pass

    @classmethod
    def check_folder(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    @classmethod
    def get_generator_model_path(cls, project):
        return os.path.join(MODEL_DIRS['GAN_INFERENCE_MODEL_DIR'], project, 'generator')

    @classmethod
    def get_discriminator_model_path(cls, project):
        return os.path.join(MODEL_DIRS['GAN_INFERENCE_MODEL_DIR'], project, 'discriminator')
    
    @classmethod
    def get_encoder_model_path(cls, project):
        return os.path.join(MODEL_DIRS['GAN_INFERENCE_MODEL_DIR'], project, 'encoder')
    
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
    def load_model(cls, model_path):
        return torch.load(model_path, map_location=cls().device())
    
    @classmethod
    def get_criterion(cls):
        return nn.MSELoss()

    @classmethod
    def device(cls):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def get_train_model_path(cls, project, task_name):
        return os.path.join(MODEL_DIRS['MOBILENET_TRAIN_MODEL_DIR'], project, task_name, 'best.pt')

    @classmethod
    def get_transform(cls, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        return transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ])
    
    @classmethod
    def get_gan_transform(cls, img_size, channels): 
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
    def get_classes(cls, project, task_name):
        classes_file = os.path.join(CLASSIFICATION_TRAIN_DATASETS_DIR, project, task_name, 'classes.txt')
        class_list = []
        with open(classes_file, 'r') as f:
            for class_name in f.readlines():
                class_list.append(class_name.strip())

        return class_list

    @classmethod
    def get_mean_std(cls, project, task_name):
        mean_std_file = os.path.join(CLASSIFICATION_TRAIN_DATASETS_DIR, project, task_name, 'mean_std.txt')
        with open(mean_std_file, 'r') as f:
            mean = eval(re.search('[\[].*[\]]', f.readline())[0])
            std = eval(re.search('[\[].*[\]]', f.readline())[0])

        return mean, std

    @classmethod
    def get_validation_images(cls, project):
        return glob(os.path.join(CLASSIFICATION_VALIDATION_DATASETS_DIR, project, '**', '*.jpg'), recursive=True) + glob(os.path.join(CLASSIFICATION_VALIDATION_DATASETS_DIR, project, '**', '*.jpeg'), recursive=True)
    
    @classmethod
    def check_validation_count(cls, images):
        return len(images)
    
    @classmethod
    def get_underkill_folder(cls, project, task_name):
        dst_folder = os.path.join(CLASSIFICATION_UNDERKILL_DATASETS_DIR, project, task_name)
        cls().check_folder(dst_folder)
        
        return dst_folder
    
    @classmethod
    def output_underkill_image(cls, image_path, underkill_folder):
        image_name = os.path.basename(image_path)
        dst_path = os.path.abspath(os.path.join(underkill_folder, image_name))
        shutil.copyfile(image_path, dst_path)

    @classmethod
    def check_validation_result(cls, underkill_count, validation_count, underkill_rate=UNDERKILL_RATE):
        if underkill_count / validation_count > underkill_rate:
            return False
        else:
            return True
        
    @classmethod
    def classification_inference(cls, image_path, model, class_list, data_transforms, confidence):
        image = Image.open(image_path).convert("RGB")
        input_tensor = data_transforms(image)
        input_batch = input_tensor.unsqueeze(0).to(cls().device())

        with torch.no_grad():
            output = model(input_batch)

        prediction = torch.nn.functional.softmax(output[0], dim=0)
        predicted_score = prediction.amax().item()
        predicted_class = prediction.argmax().item()

        if class_list[predicted_class] == 'OK' and predicted_score > confidence:
            return True
        else:
            return False

    @classmethod
    def gan_inference(cls, image_path, generator, discriminator, encoder, criterion, kappa, anormaly_threshold, img_size=256, channels=3):
        image = cv2.imread(image_path)
        image = Image.fromarray(image, 'RGB')
        transform = cls().get_gan_transform(img_size, channels)
        image_tensor = transform(image)
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

            # Scores for anomaly detection
            img_distance = criterion(fake_img, real_img)
            loss_feature = criterion(fake_feature, real_feature)
            anomaly_score = img_distance + kappa * loss_feature
            anomaly_score = float(f'{anomaly_score}')
                
            if anomaly_score > anormaly_threshold:
                return False
            else:
                return True


class YOLOFanoGANInference(YOLOInference):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_generator_model_path(cls, project):
        return os.path.join(MODEL_DIRS['GAN_INFERENCE_MODEL_DIR'], project, 'generator')

    @classmethod
    def get_discriminator_model_path(cls, project):
        return os.path.join(MODEL_DIRS['GAN_INFERENCE_MODEL_DIR'], project, 'discriminator')
    
    @classmethod
    def get_encoder_model_path(cls, project):
        return os.path.join(MODEL_DIRS['GAN_INFERENCE_MODEL_DIR'], project, 'encoder')

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
    
    def nk_chiprc_2_condition(self, target, class_names, result, lcl_chiprc):
        chiprc_count = 0
        if target in class_names:
            chiprc_count = result['name'].value_counts()[target]

        if target not in class_names or len(class_names) > 1:   
            return False   
        elif type(chiprc_count) == np.int64 and chiprc_count > 1:
            return False
        elif len(lcl_chiprc) > 0:
            return False
        else:
            return True
        
    def hz_chiprc_condition(self, target, class_names, result, lcl_chiprc):
        chiprc_count = 0
        if target in class_names:
            chiprc_count = result['name'].value_counts()[target]

        if target not in class_names or len(class_names) > 1:   
            return False
        elif type(chiprc_count) == np.int64 and chiprc_count > 1:
            return False
        elif len(lcl_chiprc) > 0:
            return False
        else:
            return True
        
    def zj_chiprc_condition(self, class_names, target):
        if len(class_names) == 1 and target in class_names:
            return True
        elif 'Missing' in class_names:
            return False
        elif 'PartofComp' in class_names:
            return False
        elif 'Particle' in class_names:
            return False
        elif 'Shift' in class_names:
            return False
        elif 'Billboard' in class_names:
            return False
        elif 'Flipover' in class_names:
            return False
        else:
            return False
        
    def zj_saw_condition(self, class_names):
        if len(class_names) == 3 and 'SAW_s' in class_names:
            if '5bar_s' in class_names:
                if '3bar_s' in class_names:
                    return True
                else:
                    return False
            else:
                return False
        elif 'Missing' in class_names:
            return False
        elif 'ChipRC' in class_names:
            return False
        elif 'STAN' in class_names:
            return False
        elif 'Shift' in class_names:
            return False
        else:
            return False
    
    def zj_wlcsp567l_condition(self, class_names, target):
        if target not in class_names:
            return False
        elif 'shift' in class_names:
            return False
        elif 'broken' in class_names:
            return False
        return True
    
    def zj_xtal_condition(self, class_names, target):
        if 'MISSINGSOLDER' in class_names:
            return False
        elif 'EMPTY' in class_names:
            return False
        elif 'SHIFT' in class_names:
            return False
        elif 'TOUCH' in class_names:
            return False
        elif 'STAN' in class_names:
            return False
        elif len(class_names) == 3 and target in class_names:
            return True
        else:
            return False

    def zj_ic_condition(self, class_names, target):
        if len(class_names) == 2:
            if 'POL_DOWN' in class_names and target in class_names:
                return True
            elif 'POL_RIGHT' in class_names and target in class_names:
                return True
            else:
                return False
        else:
            return False
        
    def zj_mc_condition(self, class_names, target):
        if len(class_names) == 1 and target in class_names:
            return True
        elif 'MISSING' in class_names:
            return False
        elif 'TOUCH' in class_names:
            return False
        elif 'STAN' in class_names:
            return False
        elif 'SHIFT' in class_names:
            return False
        elif 'MOVING' in class_names:
            return False
        elif 'Flipover' in class_names:
            return False
        else:
            return False
        
    @classmethod
    def jq_4pins_condition(self, class_names, target):
        if len(class_names) == 5 and target in class_names:
            if 'PADFIT' in class_names:
                return True
            else:
                return False 
        elif 'MISSINGSOLDER' in class_names:
            return False
        elif 'SHIFT' in class_names:
            return False
        elif 'PADSHT' in class_names:
            return False        
        elif 'MOVING' in class_names:
            return False       
        elif 'TOUCH' in class_names:
            return False           
        elif 'EMPTY' in class_names:
            return False
        elif 'STAN' in class_names:
            return False
        else:
            return False

    @classmethod
    def jq_chiprc_condition(self, class_names, target):
        if len(class_names) == 1 and target in class_names:
            return True
        elif 'MISSING' in class_names:
            return False
        elif 'TOUCH' in class_names:
            return False
        elif 'STAN' in class_names:
            return False
        elif 'SHIFT' in class_names:
            return False
        elif 'TPD' in class_names:
            return False
        elif 'MOVING' in class_names:
            return False
        elif 'EMPTY' in class_names:
            return False
        elif 'INVERSED' in class_names:
            return False      
        elif 'BROKEN' in class_names:
            return False       
        elif 'GAP' in class_names:
            if list(class_names).count(target) == 1:
                return True
            else:
                return False
        elif 'LYTBRI' in class_names:
            if list(class_names).count(target) == 1:
                return True
            else:
                return False            
        else:
            return False

    @classmethod
    def jq_icbga_condition(self, class_names, target):
        if len(class_names) == 1 and target in class_names:
            return True
        elif 'SHIFT' in class_names:
            return False
        elif 'MOVING' in class_names:
            return False     
        elif 'MISSINGSOLDER' in class_names:
            return False
        elif 'TOUCH' in class_names:
            return False            
        elif 'STAN' in class_names:
            return False
        elif 'POL' in class_names:
            if list(class_names).count('BODY') == 1:
                return True
            else:
                return False       
        else:
            return False

    @classmethod
    def jq_filter_condition(self, class_names, target):
        if len(class_names) == 1 and target in class_names:
            return True
        elif 'MISSING' in class_names:
            return False
        elif 'SHIFT' in class_names:
            return False
        elif 'MOVING' in class_names:
            return False        
        elif 'BROKEN' in class_names:
            return False
        elif 'TOUCH' in class_names:
            if 'METEL' in class_names:
                return True
            else:
                return False            
        elif 'EMPTY' in class_names:
            return False
        elif 'STAN' in class_names:
            return False
        else:
            return False

    @classmethod
    def jq_nefang_condition(self, class_names, target):
        if len(class_names) == 4 and 'ALIGN' in class_names and 'SHIFT' not in class_names:
            if target in class_names:
                return True
            else:
                return False
        elif 'CORSHT' in class_names:
            return False
        elif 'SHIFT' in class_names:
            return False
        else:
            return False

    @classmethod
    def jq_xtal_condition(self, class_names, target):
        if 'MISSINGSOLDER' in class_names:
            return False
        elif 'EMPTY' in class_names:
            return False
        elif 'SHIFT' in class_names:
            return False
        elif 'TOUCH' in class_names:
            return False
        elif 'STAN' in class_names:
            return False
        elif len(class_names) == 3 and target in class_names:
            return True
        else:
            return False

    @classmethod
    def jq_sot_condition(self, class_names, target):
        if len(class_names) == 2 and 'BODY' in class_names:
            if target in class_names:
                return True
            else:
                return False
        elif 'MISSINGSOLDER' in class_names:
            return False
        elif 'SHIFT' in class_names:
            return False
        elif 'MOVING' in class_names:
            return False      
        elif 'BROKEN' in class_names:
            return False
        elif 'TOUCH' in class_names:
            return False          
        elif 'EMPTY' in class_names:
            return False
        elif 'STAN' in class_names:
            return False
        else:
            return False

    @classmethod
    def yolo_predict(cls, model, image_path, project, chiprc_threshold=0.5):
        result = model(image_path).pandas().xyxy[0]
        class_names = result['name'].unique()

        if project == 'NK_DAOI_CHIPRC_2':
            target = 'ChipRC'
            target_df = result[result['name'] == target]
            lcl_chiprc = list(filter(lambda x: x < chiprc_threshold ,list(result[result['name' ]== target]['confidence'])))

            return cls().nk_chiprc_2_condition(target, class_names, result, lcl_chiprc), target_df
        
        elif project == 'HZ_CHIPRC':
            target = 'ChipRC'
            lcl_chiprc = list(filter(lambda x: x < chiprc_threshold ,list(result[result['name'] == target]['confidence'])))

            return cls().hz_chiprc_condition(target, class_names, result, lcl_chiprc)
        
        elif project == 'ZJ_CHIPRC':
            target = 'Comp'

            return cls().zj_chiprc_condition(class_names, target)
        
        elif project == 'ZJ_SAW':
            target = 'SAW_t'

            return cls().zj_saw_condition(class_names)
        
        elif project == 'ZJ_WLCSP567L':
            target = 'BGA'
            target_df = result[result['name'] == target]

            return cls().zj_wlcsp567l_condition(class_names, target), target_df
        
        elif project == 'ZJ_XTAL':
            target = 'BODY'

            return cls().zj_xtal_condition(class_names, target)

        elif project == 'ZJ_IC':
            target = 'POL'

            return cls().zj_ic_condition(class_names, target)
        
        elif project == 'ZJ_MC':
            target = 'Comp'

            return cls().zj_mc_condition(class_names, target)
        
        elif project == 'JQ_4PINS':
            class_names = result['name'].values
            target = 'BODY'

            return cls().jq_4pins_condition(class_names, target)
        
        elif project == 'JQ_CHIPRC':
            target = 'COMP'

            return cls().jq_chiprc_condition(class_names, target)
        
        elif project == 'JQ_ICBGA':
            target = 'BODY'

            return cls().jq_icbga_condition(class_names, target)
        
        elif project == 'JQ_FILTER':
            target = 'COMP'

            return cls().jq_filter_condition(class_names, target)
        
        elif project == 'JQ_NEFANG':
            class_names = result['name'].values
            target = 'COR'

            return cls().jq_nefang_condition(class_names, target)
        
        elif project == 'JQ_XTAL':
            target = 'BODY'
            target_df = result[result['name'] == target]

            return cls().jq_xtal_condition(class_names, target)
        
        elif project == 'JQ_SOT':
            target = 'POL'

            return cls().jq_sot_condition(class_names, target)
    
    @classmethod
    def vae_predict(cls, image_path, target_df, transform, generator, discriminator, encoder, criterion, kappa=1.0, anormaly_threshold=0.2):
        image = cv2.imread(image_path)
        for i in range(target_df.shape[0]):
            chiprc = target_df.iloc[i:i+1, :]
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

                if anomaly_score > anormaly_threshold:
                    return False
                else:
                    return True


class MobileNetYOLOIForestInference(YOLOInference):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_classification_model_path(cls, project):
        return os.path.join(MODEL_DIRS['CLASSIFICATION_INFERNCE_MODEL_DIR'], project, 'classification_model.pt')

    @classmethod
    def get_pinlocation_model_path(cls, project):
        return os.path.join(MODEL_DIRS['YOLO_INFERENCE_MODEL_DIR'], project, OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_PICKLE_MODEL_NAME'])

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
                mean=OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_CLASSIFICATION_MEAN'], 
                std=OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_CLASSIFICATION_STD']
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
        if OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_CLASSIFICATION_CLASS_NAMES'][preds[0]] == 'OK':
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

        for way in OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_WAYS'].keys():
            if way in os.path.basename(image_path):
                img_way = OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_WAYS'][way]
        for comp in OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_PART_NUMBER'].keys():
            if comp in os.path.basename(image_path):
                img_comp = OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_PART_NUMBER'][comp]

        lcl_pcie = []
        lcl_body = []

        PCIE = result[result['name'] == 'PCIE']['confidence']
        for i, conf in enumerate(PCIE):
            if conf < OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_PCIE_THRESHOLD']:
                lcl_pcie.append(PCIE.index[i])
        result = result.drop(index=lcl_pcie)

        BODY = result[result['name'] == 'BODY']['confidence']
        for i, conf in enumerate(BODY):
            if conf < OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_BODY_THRESHOLD']:
                lcl_body.append(BODY.index[i])
        result = result.drop(index=lcl_body)

        for NG_type in OBJECT_DETECTION_PCIE['OBJECT_DETECTION_PCIE_CLASSIFICATION_NGS']:
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


class EfficientNetEmbeddingInference:
    def __init__(self) -> None:
        pass

    @classmethod
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_model_path(cls, project):
        return os.path.join(MODEL_DIRS['CLASSIFICATION_INFERNCE_MODEL_DIR'], project, 'classification_model.pt')
    
    @classmethod
    def load_model(cls, model_path):
        return torch.load(model_path, map_location=cls().device())

    @classmethod
    def get_train_model_path(cls, project, task_name):
        return os.path.join(MODEL_DIRS['METRIC_LEARNING_TRAIN_MODEL_DIR'], project, task_name, 'best.pt')
    
    @classmethod
    def get_mean_std(cls, project, task_name):
        mean_std_file = os.path.join(CLASSIFICATION_TRAIN_DATASETS_DIR, project, task_name, 'mean_std.txt')
        with open(mean_std_file, 'r') as f:
            mean = eval(re.search('[\[].*[\]]', f.readline())[0])
            std = eval(re.search('[\[].*[\]]', f.readline())[0])

        return mean, std

    @classmethod
    def get_validation_images(cls, project):
        return glob(os.path.join(CLASSIFICATION_VALIDATION_DATASETS_DIR, project, '**', '*.jpg'), recursive=True) + glob(os.path.join(CLASSIFICATION_VALIDATION_DATASETS_DIR, project, '**', '*.jpeg'), recursive=True)
    
    @classmethod
    def check_validation_count(cls, images):
        return len(images)
    
    @classmethod
    def check_folder(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

        return path
    
    @classmethod
    def get_underkill_folder(cls, project, task_name):
        dst_folder = os.path.join(CLASSIFICATION_UNDERKILL_DATASETS_DIR, project, task_name)
        cls().check_folder(dst_folder)
        
        return dst_folder

    @classmethod
    def inference(cls, model, mean, std, image_path, query_image, seed, confidence):
        setup_seed(seed)

        if os.path.isfile(image_path):
            query_features = extract_query_features(model, query_image, cls().device(), mean, std)
            output = extract_query_features(model, image_path, cls().device(), mean, std)
            score = torch.cosine_similarity(output, query_features)

        if score > confidence:
            return True
        else:
            return False

    @classmethod
    def output_underkill_image(cls, image_path, underkill_folder):
        image_name = os.path.basename(image_path)
        dst_path = os.path.abspath(os.path.join(underkill_folder, image_name))
        shutil.copyfile(image_path, dst_path)
        
    @classmethod
    def check_validation_result(cls, underkill_count, validation_count):
        if underkill_count / validation_count > UNDERKILL_RATE:
            return False
        else:
            return True