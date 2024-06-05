from pathlib import Path
from flask_restx import marshal_with
from apps.databases.ai import AiModelInfo, AiModelPerf, OdTrainingInfo, ClsTrainingInfo
from apps.databases import db
from apps.serializers.information_serializer import *


class TrainingInfoService:
    table: OdTrainingInfo = None
    
    def _get_model_version(self, comp_type, val_status='APPROVE'):
        training_info = self.table.query.filter(
            self.table.comp_type == comp_type,
            self.table.val_status == val_status
        ).order_by(self.table.model_version.desc()).first()
        
        if training_info:
            return int(training_info.model_version) + 1
        else:
            return 0
    
    @classmethod
    @marshal_with(get_training_info_val_status_serializer)
    def get_training_info_val_status(self, task_id):
        training_info = self.table.query.filter(
            self.table.task_id == task_id
        ).first()
        
        if training_info:
            return training_info
        else:
            return {'val_status': 'APPROVE'}
    
    @classmethod
    @marshal_with(update_new_info_serializer)
    def update_new_info(cls, task_id, comp_type, validate_result):
        if validate_result:
            val_status = 'APPROVE'
            model_version = cls()._get_model_version(comp_type, val_status)
        else:
            val_status = 'FAIL'
            model_version = None
            
        row = cls.table(
            task_id=task_id,
            comp_type=comp_type,
            val_status=val_status,
            model_version=model_version
        )
        db.session.add(row)
        db.session.commit()
        
        return {'status': f'update new info success in {cls.table.__tablename__}'}
        
        
class ODTrainingInfoService(TrainingInfoService):
    table = OdTrainingInfo
    

class CLSTrainingInfoService(TrainingInfoService):
    table = ClsTrainingInfo
    
    
class AIModelInformationService:
    @classmethod
    @marshal_with(update_new_ai_model_info_serializer)
    def update_ai_model_info(cls, model_type, model_path, ip_address, finetune_id, finetune_type, verified_status):
        row = AiModelInfo(
            model_type=model_type,
            model_name='ORG',
            model_path=f'{ip_address}//{Path(model_path).resolve()}',
            verified_status=verified_status,
            finetune_id=finetune_id,
            finetune_type=finetune_type
        ) 
        db.session.add(row)
        db.session.commit()
        
        return AiModelInfo.query.order_by(AiModelInfo.model_id.desc()).first()
    

class AIModelPerformanceSerivce:
    @classmethod
    @marshal_with(update_new_info_serializer)
    def update_ai_model_perf(cls, model_id, metrics_result, false_negative_imgs, false_positive_imgs):
        row = AiModelPerf(
            model_id=model_id,
            metrics_result=repr(metrics_result), 
            false_negative_imgs=repr(false_negative_imgs), 
            false_positive_imgs=repr(false_positive_imgs)
        )
        db.session.add(row)
        db.session.commit()
        
        return {'status': f'update new info success in {AiModelPerf.__tablename__}'}