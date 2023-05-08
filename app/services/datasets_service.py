from data.config import ORIGIN_DATASETS_FOLDER_PROFIX, ORIGIN_DATASETS_DIR
from datetime import datetime
from glob import glob
import os
import zipfile


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