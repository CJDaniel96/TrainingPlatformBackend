import time
from app.config import LISTEN_DATABASE_TIME_SLEEP
from app.services.database_service import IRIRecord, URDRecord
from app.services.logging_service import Logger


class Listener:
    @classmethod
    def listen(cls):
        Logger.info('Check Database Record')
        while True:
            iri_record = IRIRecord.status()
            urd_record = URDRecord.status()
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
        if tablename == 'iri_record':
            IRIRecord.update_status(id, status)
        elif tablename == 'urd_record':
            URDRecord.update_status(id, status)

    @classmethod
    def update_record_task_id(cls, tablename, id, task_id):
        if tablename == 'iri_record':
            IRIRecord.update_task_id(id, task_id)
        elif tablename == 'urd_record':
            URDRecord.update_task_id(id, task_id)

    @classmethod
    def update_record_task_name(cls, tablename, id, task_name):
        if tablename == 'iri_record':
            IRIRecord.update_task_name(id, task_name)
        elif tablename == 'urd_record':
            URDRecord.update_task_name(id, task_name)