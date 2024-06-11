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
        
class ZJ_ChipRC:
    @classmethod
    def predict(cls, result):
        cls_list = result['name'].unique()
        if len(cls_list) == 1 and 'Comp' in cls_list:
            return 'OK', 'OK'
        elif 'Missing' in cls_list:
            return 'NG', 'Missing'
        elif 'PartofComp' in cls_list:
            return 'NG', 'PartofComp'
        elif 'Particle' in cls_list:
            return 'NG', 'Particle'
        elif 'Shift' in cls_list:
            return 'NG', 'Shift'
        elif 'Billboard' in cls_list:
            return 'NG', 'Billboard'
        elif 'Flipover' in cls_list:
            return 'NG', 'Flipover'
        else:
            return 'NG', 'Others'
        
class ZJ_MC:
    @classmethod
    def predict(cls, result):
        cls_list = result['name'].unique()
        if len(cls_list) == 1 and 'COMP' in cls_list:
            return 'OK', 'OK'
        elif 'MISSING' in cls_list:
            return 'NG', 'Missing'
        elif 'TOUCH' in cls_list:
            return 'NG', 'PartofComp'
        elif 'STAN' in cls_list:
            return 'NG', 'Particle'
        elif 'SHIFT' in cls_list:
            return 'NG', 'Shift'
        elif 'MOVING' in cls_list:
            return 'NG', 'SHIFT'
        elif 'Flipover' in cls_list:
            return 'NG', 'Flipover'
        else:
            return 'NG', 'Others'
        
class ZJ_WLCSP567L:
    @classmethod
    def predict(cls, result):
        cls_list = result['name'].unique()
        Broken = result[result['name'] == 'broken']
        WLCSP567L = result[result['name'] == 'BGA']

        if 'BGA' not in cls_list:
            return 'NG', 'Missing', WLCSP567L, Broken
        elif 'shift' in cls_list:
            return 'NG', 'shift', WLCSP567L, Broken
        elif 'broken' in cls_list:
            return 'NG', 'broken', WLCSP567L, Broken
        return 'OK', 'OK', WLCSP567L, Broken
    
class ZJ_XTAL:
    @classmethod
    def predict(cls, result):
        cls_list = result['name'].unique()
 
        if 'MISSINGSOLDER' in cls_list:
            return 'NG', 'Missing'
        elif 'EMPTY' in cls_list:
            return 'NG', 'Flip'
        elif 'SHIFT' in cls_list:
            return 'NG', 'SHIFT'
        elif 'TOUCH' in cls_list:
            return 'NG', 'PartofComp'
        elif 'STAN' in cls_list:
            return 'NG', 'STAN'
        else:
            return 'NG', 'Others'
        
class JQ_4PINS:
    @classmethod
    def predict(cls, result):
        cls_list = result['name']

        if len(cls_list) == 5 and 'BODY' in cls_list.values:
            if 'PADFIT' in cls_list.values:
                return 'OK', 'OK'
            else:
                return 'NG', 'Shift' 
        elif 'MISSINGSOLDER' in cls_list.values:
            return 'NG', 'Missing'
        elif 'SHIFT' in cls_list.values:
            return 'NG', 'Rotation'
        elif 'PADSHT' in cls_list.values:
            return 'NG', 'Shift'        
        elif 'MOVING' in cls_list.values:
            return 'NG', 'Shift'        
        elif 'TOUCH' in cls_list.values:
            return 'NG', 'TOUCH'            
        elif 'EMPTY' in cls_list.values:
            return 'NG', 'Empty'
        elif 'STAN' in cls_list.values:
            return 'NG', 'Stain'
        else:
            return 'NG', 'Others'

class JQ_ChipRC:
    @classmethod
    def predict(cls, result):
        class_names = result['name'].unique()
        if len(class_names) == 1 and 'COMP' in class_names:
            return 'OK', 'OK'
        elif 'GAP' in class_names and 'COMP' in class_names and len(class_names) == 2:
            return 'OK', 'GAP_OK'
        elif 'LYTBRI' in class_names and 'COMP' in class_names and len(class_names) == 2:
            return 'OK', 'LYTBRI_OK'

        if 'MISSING' in class_names:
            return 'NG', 'MISSING'
        elif 'TOUCH' in class_names:
            return 'NG', 'TOUCH'
        elif 'STAN' in class_names:
            return 'NG', 'STAN'
        elif 'SHIFT' in class_names:
            return 'NG', 'SHIFT'
        elif 'TPD' in class_names:
            return 'NG', 'TPD'
        elif 'MOVING' in class_names:
            return 'NG', 'MOVING'
        elif 'EMPTY' in class_names:
            return 'NG', 'EMPTY'
        elif 'INVERSED' in class_names:
            return 'NG', 'INVERSED'     
        elif 'BROKEN' in class_names:
            return 'NG', 'BROKEN'      
        elif 'GAP' in class_names:
            return 'NG', 'GAP'
        elif 'LYTBRI' in class_names:
            return 'NG', 'LYTBRI'           
        else:
            return 'NG', 'Others'
        
class JQ_FILTER:
    @classmethod
    def predict(cls, result):
        cls_list = result['name'].unique()
        
        if len(cls_list) == 1 and 'COMP' in cls_list:
            return 'OK', 'OK'
        elif 'MISSING' in cls_list:
            return 'NG', 'Missing'
        elif 'SHIFT' in cls_list:
            return 'NG', 'Rotation'
        elif 'MOVING' in cls_list:
            return 'NG', 'Shift'        
        elif 'BROKEN' in cls_list:
            return 'NG', 'Broken'
        elif 'TOUCH' in cls_list:
            if 'METEL' in cls_list:
                return 'OK', 'OK'
            else:
                return 'NG', 'TOUCH'            
        elif 'EMPTY' in cls_list:
            return 'NG', 'Empty'
        elif 'STAN' in cls_list:
            return 'NG', 'Stain'
        else:
            return 'NG', 'Others'
        
class JQ_LXX:
    @classmethod
    def predict(cls, result):
        filtered_result = result[result['confidence'] > 0.5]
        class_names = filtered_result['name'].unique()
            
        if 'MISSING' in class_names:
            return 'NG', 'MISSING'
        elif 'TOUCH' in class_names:
            return 'NG', 'TOUCH'
        elif 'STAN' in class_names:
            return 'NG', 'STAN'
        elif 'SHIFT' in class_names:
            return 'NG', 'SHIFT'
        elif 'TPD' in class_names:
            return 'NG', 'TPD'
        elif 'MOVING' in class_names:
            return 'NG', 'MOVING'
        elif 'EMPTY' in class_names:
            return 'NG', 'EMPTY'
        elif 'INVERSED' in class_names:
            return 'NG', 'INVERSED'      
        elif 'BROKEN' in class_names:
            return 'NG', 'BROKEN'
        elif len(class_names) == 1 and 'COMP' in class_names:
            return 'OK', 'COMP'
        elif 'GAP' in class_names:
            if list(class_names).count('COMP') == 1:
                return 'OK', 'COMP'
            else:
                return 'NG', 'GAP'
        elif 'LYTBRI' in class_names:
            if list(class_names).count('COMP') == 1:
                return 'OK', 'COMP'
            else:
                return 'NG', 'LYTBRI' 
        elif 'BODY' in class_names:
            bodyObj = result[result['confidence'] > 0.8]
            print(bodyObj)
            if not bodyObj.empty:
                return 'OK', 'BODY'  
            else:
                return 'NG', 'UNKNOWN'
        else:
            return 'NG', 'UNKNOWN'
        
class JQ_SOT:
    @classmethod
    def predict(cls, result):
        cls_list = result['name'].unique()

        if len(cls_list) == 2 and 'BODY' in cls_list:
            if 'POL' in cls_list:
                return 'OK', 'OK'
            else:
                return 'NG', 'NoPOL' 
        elif 'MISSINGSOLDER' in cls_list:
            return 'NG', 'Missing'
        elif 'SHIFT' in cls_list:
            return 'NG', 'Rotation'
        elif 'MOVING' in cls_list:
            return 'NG', 'Shift'        
        elif 'BROKEN' in cls_list:
            return 'NG', 'Broken'
        elif 'TOUCH' in cls_list:
            return 'NG', 'TOUCH'            
        elif 'EMPTY' in cls_list:
            return 'NG', 'Empty'
        elif 'STAN' in cls_list:
            return 'NG', 'Stain'
        else:
            return 'NG', 'Others' 