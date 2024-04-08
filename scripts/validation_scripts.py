import numpy as np


class NK_DAOI_CHIPRC_2:
    @classmethod
    def predict(cls, result, ok_label): 
        class_names = np.unique(result['defect_name'])
        lcl_chiprc = 0
        for name, score in zip(result['defect_name'], result['confidence']):
            if name == 'ChipRC' and score > 0.5:
                lcl_chiprc += 1
        chiprc_count = 0

        if ok_label in class_names:
            chiprc_count = result['defect_name'].value_counts()[ok_label]

        if ok_label not in class_names or len(class_names) > 1:   
            return False  
        elif type(chiprc_count) == np.int64 and chiprc_count > 1:
            return False
        elif lcl_chiprc > 0:
            return False
        else:
            return True
        
class NK_CHIPRC_NA:
    @classmethod
    def predict(cls, result):
        class_names = np.unique(result['defect_name'])
        
        if len(class_names) == 0:
            return 'NG', 'nolabel'
        elif 'ChipRC' in class_names:
            return 'NG', 'ChipRC'
        elif 'STAN_BLACK' in class_names:
            return 'NG', 'STAN_BLACK'
        elif 'STAN_SN' in class_names:
            return 'NG', 'STAN_SN'
        elif 'OVERLAP' in class_names:
            return 'NG', 'OVERLAP'
        elif 'MISSING' in class_names and len(class_names) == 1:
            return 'OK', 'OK'
        else:
            return 'NG', 'else'
