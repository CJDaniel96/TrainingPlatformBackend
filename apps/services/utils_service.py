from datetime import datetime
from flask_restx import marshal_with
from xml.dom.minidom import Document
from pathlib import Path
from apps.serializers.utils_serializer import *
from config import DOWNLOADS_DATA_DIR
import xml.etree.ElementTree as et
import zipfile
import os


class DateTimeService:
    @classmethod
    @marshal_with(datetime_serializer)
    def get_datetime_barcode(cls):
        return {'datetime_barcode': datetime.now()}
    
    
class UnzipDataService:
    @classmethod
    @marshal_with(unzip_data_serializer)
    def unzip_data(cls, zip_paths, name):
        unzip_data_folder = Path(DOWNLOADS_DATA_DIR, name).resolve()
        
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                zip_file.extractall(str(unzip_data_folder))
            os.remove(zip_path)
        
        return {'unzip_data_folder': str(unzip_data_folder)}
    

class XMLService:
    def _create_folder_node(self, doc, image_path):
        folder = doc.createElement('folder')
        folder_text = doc.createTextNode(str(Path(image_path).parent))
        folder.appendChild(folder_text)
        
        return folder
    
    def _create_filename_node(self, doc, image_path):
        filename = doc.createElement('filename')
        filename_text = doc.createTextNode(Path(image_path).name)
        filename.appendChild(filename_text)
        
        return filename

    def _create_path_node(self, doc, image_path):
        path = doc.createElement('path')
        path_text = doc.createTextNode(image_path)
        path.appendChild(path_text)
        
        return path

    def _create_size_node(self, doc, image_size):
        size = doc.createElement('size')
        width, height, depth = image_size
        size.appendChild(self._create_dimension_node(doc, 'width', width))
        size.appendChild(self._create_dimension_node(doc, 'height', height))
        size.appendChild(self._create_dimension_node(doc, 'depth', depth))
        
        return size

    def _create_dimension_node(self, doc, tag, value):
        dimension = doc.createElement(tag)
        dimension_text = doc.createTextNode(str(value))
        dimension.appendChild(dimension_text)
        
        return dimension

    def _create_object_node(self, doc, name, box_list):
        xml_object = doc.createElement('object')
        xml_object.appendChild(self._create_name_node(doc, name))
        xml_object.appendChild(self._create_bndbox_node(doc, box_list))
        
        return xml_object

    def _create_name_node(self, doc, name):
        name_node = doc.createElement('name')
        name_text = doc.createTextNode(name)
        name_node.appendChild(name_text)
        
        return name_node

    def _create_bndbox_node(self, doc, box_list):
        bndbox = doc.createElement('bndbox')
        for tag, value in zip(['xmin', 'ymin', 'xmax', 'ymax'], box_list):
            bndbox.appendChild(self._create_dimension_node(doc, tag, int(value)))

        return bndbox
    
    def _output_xml(self, image_size, image_path, defect_name, defect_position):
        image_size = eval(image_size)
        doc = Document()
        root = doc.createElement('annotation')
        doc.appendChild(root)

        root.appendChild(self._create_folder_node(doc, image_path))
        root.appendChild(self._create_filename_node(doc, image_path))
        root.appendChild(self._create_path_node(doc, image_path))
        root.appendChild(self._create_size_node(doc, image_size))

        if defect_name is not None or defect_position is not None:
            for name_list, box_list in zip(defect_name, defect_position):
                root.appendChild(self._create_object_node(doc, name_list, box_list))

        xml_path = str(Path(image_path).with_suffix('.xml'))
        with Path(xml_path).open('w') as xml:
            doc.writexml(xml, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
            
    @classmethod
    def output_xmls(cls, results):
        output_list = []
        
        for result in results:
            try:
                cls()._output_xml(result['image_size'], result['image_path'], result['defect_name'], result['defect_position'])
                output_list.append({
                    'image_path': result['image_path'],
                    'status': 'success', 
                    'message': 'XML file created successfully'
                })
            except Exception as e:
                output_list.append({
                    'image_path': result['image_path'],
                    'status': 'error', 
                    'message': f'Error creating XML file: {str(e)}'
                })
        
        return output_list
    
    
class XMLtoYOLOService:
    def _xml_to_yolo_bbox(self, bbox, w, h):
        # xmin, ymin, xmax, ymax
        x_center = ((bbox[2] + bbox[0]) / 2) / w
        y_center = ((bbox[3] + bbox[1]) / 2) / h
        width = (bbox[2] - bbox[0]) / w
        height = (bbox[3] - bbox[1]) / h
        
        return [x_center, y_center, width, height]

    def _yolo_to_xml_bbox(self, bbox, w, h):
        # x_center, y_center width heigth
        w_half_len = (bbox[2] * w) / 2
        h_half_len = (bbox[3] * h) / 2
        xmin = int((bbox[0] * w) - w_half_len)
        ymin = int((bbox[1] * h) - h_half_len)
        xmax = int((bbox[0] * w) + w_half_len)
        ymax = int((bbox[1] * h) + h_half_len)
        
        return [xmin, ymin, xmax, ymax]
    
    @classmethod
    @marshal_with(xml_to_yolo_serializer)
    def xml_to_yolo(cls, classes:dict, xml_dir:str):
        xml_dir_obj = Path(xml_dir)
        txt_dir_obj = xml_dir_obj.parent.joinpath('labels')
        txt_dir_obj.mkdir(parents=True, exist_ok=True)
        xml_files = xml_dir_obj.glob('*.xml')
        
        for xml_file in xml_files:
            filename = xml_file.stem
            
            result = []
            
            # parse the content of the xml file
            tree = et.parse(str(xml_file))
            root = tree.getroot()
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            
            for obj in root.findall('object'):
                label = obj.find("name").text
                # check for new classes and append to list
                index = classes[label]
                pil_bbox = [int(float(x.text)) for x in obj.find("bndbox")]
                yolo_bbox = cls()._xml_to_yolo_bbox(pil_bbox, width, height)
                # convert data to string
                bbox_string = " ".join([str(x) for x in yolo_bbox])
                result.append(f"{index} {bbox_string}")
                
            # generate a YOLO format text file for each xml file
            txt_file = txt_dir_obj.joinpath(f'{filename}.txt')
            with txt_file.open('w', encoding='utf-8') as f:
                f.write('\n'.join(result))
                
                    
        return {'labels_path': str(txt_dir_obj)}