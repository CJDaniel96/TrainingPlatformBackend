import numpy as np


class NK_DAOI_CHIPRC_2:
    @classmethod
    def predict(cls, result, ok_label, chiprc_threshold=0.5): 
        class_names = np.unique(result['defect_name'])
        lcl_chiprc = 0
        for name, score in zip(result['defect_name'], result['confidence']):
            if name == 'ChipRC' and score > chiprc_threshold:
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