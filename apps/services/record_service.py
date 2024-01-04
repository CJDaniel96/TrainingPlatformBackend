from flask_restx import marshal_with
from apps.serializers.record_serializer import *
from apps.databases.ai import IriRecord, UrdRecord
from apps.databases import db
from config import RECORD_STATUSES


class BaseRecordService:
    table = None
    
    @marshal_with(record_serializer)
    def get_status(self, record_id):
        if record_id:
            return self.table.query.filter(self.table.id == record_id).first()
        else:
            return self.table.query.filter(
                self.table.status.in_(RECORD_STATUSES[self.table.__tablename__])
            ).order_by(self.table.update_time).first()
        
    @classmethod
    def update(self, record_id, column, value):
        record = self.table.query.get(record_id)
        setattr(record, column, value)
        db.session.commit()


class IRIRecordService(BaseRecordService):
    table = IriRecord


class URDRecordService(BaseRecordService):
    table = UrdRecord
