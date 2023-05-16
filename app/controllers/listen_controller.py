import time
from app.config import LISTEN_DATABASE_TIME_SLEEP
from app.services.database_service import IRIRecordService, TrainingInfoService, URDRecordService
from app.services.logging_service import Logger


class Listener:
    @classmethod
    def listen(cls):
        while True:
            Logger.info('Check Database Record')
            iri_record = IRIRecordService.status()
            urd_record = URDRecordService.status()
            if iri_record and urd_record:
                if iri_record.update_time > urd_record.update_time:
                    return urd_record
                else:
                    return iri_record
            elif not iri_record and urd_record:
                return urd_record
            elif iri_record and not urd_record:
                return iri_record
            time.sleep(LISTEN_DATABASE_TIME_SLEEP)

    @classmethod
    def update_record_status(cls, tablename, id, status):
        Logger.info('Update Database Record Status')
        if tablename == 'iri_record':
            IRIRecordService.update_status(id, status)
        elif tablename == 'urd_record':
            URDRecordService.update_status(id, status)

    @classmethod
    def update_record_task_id(cls, tablename, id, task_id):
        Logger.info('Update Database Record Task ID')
        if tablename == 'iri_record':
            IRIRecordService.update_task_id(id, task_id)
        elif tablename == 'urd_record':
            URDRecordService.update_task_id(id, task_id)

    @classmethod
    def update_record_task_name(cls, tablename, id, task_name):
        Logger.info('Update Database Record Task Name')
        if tablename == 'iri_record':
            IRIRecordService.update_task_name(id, task_name)
        elif tablename == 'urd_record':
            URDRecordService.update_task_name(id, task_name)

    @classmethod
    def update_record_object_detection_training_status(cls, tablename, id, training_status):
        Logger.info('Update Database Record Object Detection Training Status')
        if tablename == 'iri_record':
            IRIRecordService.update_od_training_status(id, training_status)
        elif tablename == 'urd_record':
            URDRecordService.update_od_training_status(id, training_status)

    @classmethod
    def update_record_object_detection_training_info(cls, result, task_id, comp_type):
        Logger.info('Update Database Record Object Detection Training Information')
        if result:
            val_status = 'APPROVE'
        else:
            val_status = 'FAIL'
        model_version = TrainingInfoService.get_object_detection_model_version(comp_type, val_status)

        if result:
            if model_version:
                model_version = int(model_version) + 1
            else:
                model_version = 1
        TrainingInfoService.insert_object_detection_result(task_id, comp_type, val_status, model_version)
