from pathlib import Path
from config import *
import yaml


class Yolov5Yaml:
    @classmethod
    def get_data_yaml(cls, algorithm, datasets_dir, classes):
        data_yaml_obj = Path(YAMLS_DIR, algorithm, YOLOV5_DATA_YAML)
        with data_yaml_obj.open('r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            
        classes_map = {v: k for k, v in classes.items()}
        
        data_yaml['path'] = str(Path(datasets_dir).resolve())
        data_yaml['names'] = classes_map
        
        with data_yaml_obj.open('w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f)
            
        return str(data_yaml_obj)
            
    @classmethod
    def get_model_yaml(cls, algorithm, nc):
        model_yaml_obj = Path(YAMLS_DIR, algorithm, YOLOV5_MODEL_YAML)
        with model_yaml_obj.open('r', encoding='utf-8') as f:
            model_yaml = yaml.safe_load(f)
        
        model_yaml['nc'] = nc
        
        with model_yaml_obj.open('w', encoding='utf-8') as f:
            yaml.dump(model_yaml, f)
            
        return str(model_yaml_obj)
    
    @classmethod
    def get_hyp_yaml(cls, algorithm, project):
        if project in YOLOV5_HYP_RANDOM_CROP_CLOSE_PROJECT:
            return str(Path(YAMLS_DIR, algorithm, YOLOV5_CLOSE_RANDOM_HYP_YAML))
        else:
            return str(Path(YAMLS_DIR, algorithm, YOLOV5_HYP_YAML))