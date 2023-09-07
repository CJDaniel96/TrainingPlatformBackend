from glob import glob
import os
import uuid

from data.config import OBJECT_DETECTION_VALIDATION_DATASETS_DIR


class ManageService:
    @classmethod
    def check_image_type(cls, image, target_image_type='jpg'):
        image_name, image_type = os.path.splitext(image)
        if image_type[1:] != target_image_type:
            os.rename(image, f'{image_name}.{target_image_type}')
    
    @classmethod
    def parse_chinses_to_english(cls, path, image_type='jpg'):
        for img_path in glob(os.path.join(path, '**', f'.{image_type}'), recursive=True):
            new_img_path = ''
            if '缺件' in new_img_path:
                new_img_path = img_path.replace('缺件', 'Lack')
            if '多件' in new_img_path:
                new_img_path = img_path.replace('多件', 'Multi')
            if '補正' in new_img_path:
                new_img_path = img_path.replace('補正', 'Repositive')
            if '異物' in new_img_path:
                new_img_path = img_path.replace('異物', 'Foreign')
            if '良品' in new_img_path:
                new_img_path = img_path.replace('良品', 'GoodProduct')
            if '變形' in new_img_path:
                new_img_path = img_path.replace('變形', 'Distortion')
            if '偏移' in new_img_path:
                new_img_path = img_path.replace('偏移', 'Shift')
            if '短路' in new_img_path:
                new_img_path = img_path.replace('短路', 'ShortCircuit')
            if '少錫' in new_img_path:
                new_img_path = img_path.replace('少錫', 'UnderTinning')
            if '殘膠' in new_img_path:
                new_img_path = img_path.replace('殘膠', 'Residue')
            if '條碼' in new_img_path:
                new_img_path = img_path.replace('條碼', 'Barcode')
            if '瑕疵' in new_img_path:
                new_img_path = img_path.replace('瑕疵', 'Defect')
            if '撞傷' in new_img_path:
                new_img_path = img_path.replace('撞傷', 'Bruise')
            if '外觀不良' in new_img_path:
                new_img_path = img_path.replace('外觀不良', 'BadApperance')
            if '立碑' in new_img_path:
                new_img_path = img_path.replace('立碑', 'Stand')
            if '溢錫' in new_img_path:
                new_img_path = img_path.replace('溢錫', 'SpilledTin')
            if '翻件' in new_img_path:
                new_img_path = img_path.replace('翻件', 'Flip')
            if '空焊' in new_img_path:
                new_img_path = img_path.replace('空焊', 'EmptyWelding')
            if '蝧颱辣' in new_img_path:
                new_img_path = img_path.replace('蝧颱辣', 'Other')
            if '翹腳' in new_img_path:
                new_img_path = img_path.replace('翹腳', 'CrossLegged')
            if new_img_path:
                os.rename(img_path, new_img_path)
