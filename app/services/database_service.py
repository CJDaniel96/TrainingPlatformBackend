from datetime import timedelta
import random
from app.config import IRI_RECORD_STATUS, URD_RECORD_STATUS
from data.config import DATABASES
from data.database.ai import ImagePool, IriRecord, UrdRecord
from data.database.amr_nifi_test import AmrRawData
from data.database.sessions import create_session


AI = DATABASES['ai']['NAME']
AMR_NIFI_TEST = DATABASES['amr_nifi_test']['NAME']
CVAT = DATABASES['cvat']['NAME']


class IRIRecord:
    @classmethod
    def status(cls):
        with create_session(AI) as session:
            return session.query(IriRecord).filter(
                IriRecord.status.in_(IRI_RECORD_STATUS)
            ).order_by(IriRecord.update_time.desc()).first()

    @classmethod
    def update_line(cls, id, lines):
        with create_session(AI) as session:
            session.query(IriRecord).filter(IriRecord.id == id).update({
                'line': repr(lines).replace('\'', '"')
            })
            session.commit()

    @classmethod
    def update_status(cls, id, status):
        with create_session(AI) as session:
            session.query(IriRecord).filter(IriRecord.id == id).update({
                "status": status
            })
            session.commit()

    @classmethod
    def update_task_id(cls, id, task_id):
        with create_session(AI) as session:
            session.query(IriRecord).filter(IriRecord.id == id).update({
                "task_id": task_id
            })
            session.commit()

    @classmethod
    def update_task_name(cls, id, task_name):
        with create_session(AI) as session:
            session.query(IriRecord).filter(IriRecord.id == id).update({
                "task": task_name
            })
            session.commit()

    @classmethod
    def update_od_training_status(cls, id, od_training_status):
        with create_session(AI) as session:
            session.query(IriRecord).filter(IriRecord.id == id).update({
                "od_training": od_training_status
            })
            session.commit()

    @classmethod
    def update_cls_training_status(cls, id, cls_training_status):
        with create_session(AI) as session:
            session.query(IriRecord).filter(IriRecord.id == id).update({
                "cls_training": cls_training_status
            })
            session.commit()


class URDRecord:
    @classmethod
    def status(cls):
        with create_session(AI) as session:
            return session.query(UrdRecord).filter(
                UrdRecord.status.in_(URD_RECORD_STATUS)
            ).order_by(UrdRecord.update_time.desc()).first()

    @classmethod
    def update_line(cls, id, lines):
        with create_session(AI) as session:
            session.query(UrdRecord).filter(UrdRecord.id == id).update({
                'line': repr(lines).replace('\'', '"')
            })
            session.commit()

    @classmethod
    def update_status(cls, id, status):
        with create_session(AI) as session:
            session.query(UrdRecord).filter(UrdRecord.id == id).update({
                "status": status
            })
            session.commit()

    @classmethod
    def update_task_id(cls, id, task_id):
        with create_session(AI) as session:
            session.query(UrdRecord).filter(UrdRecord.id == id).update({
                "task_id": task_id
            })
            session.commit()

    @classmethod
    def update_task_name(cls, id, task_name):
        with create_session(AI) as session:
            session.query(UrdRecord).filter(UrdRecord.id == id).update({
                "task": task_name
            })
            session.commit()

    @classmethod
    def update_od_training_status(cls, id, od_training_status):
        with create_session(AI) as session:
            session.query(UrdRecord).filter(UrdRecord.id == id).update({
                "od_training": od_training_status
            })
            session.commit()

    @classmethod
    def update_cls_training_status(cls, id, cls_training_status):
        with create_session(AI) as session:
            session.query(UrdRecord).filter(UrdRecord.id == id).update({
                "cls_training": cls_training_status
            })
            session.commit()


class ImagePoolDatabase:
    def smart_filter_images(self, images):
        buffer = []
        dataset = {}
        for line in images:
            for data in images[line]:
                buffer.append([line, data])

        random.shuffle(buffer)

        for data in buffer[:100]:
            if data[0] not in dataset:
                dataset[data[0]] = [data[1]]
            else:
                dataset[data[0]].append(data[1])

        return dataset

    @classmethod
    def get_images(cls):...

    @classmethod
    def get_image_pool(cls):
        with create_session(AI) as session:
            return session.query(ImagePool).all()


class ImageData(ImagePoolDatabase):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_images(cls, site, line, group_type, start_date, end_date, smart_filter=False, is_covered=True, ai_result='0'):
        images = {}
        with create_session(AMR_NIFI_TEST) as session:
            end_date += timedelta(days=1)
            for each_line in eval(line):
                data = session.query(AmrRawData.image_path).filter(
                    AmrRawData.site == site,
                    AmrRawData.line_id == each_line,
                    AmrRawData.group_type == group_type,
                    AmrRawData.create_time.between(start_date, end_date),
                    AmrRawData.is_covered == is_covered,
                    AmrRawData.ai_result == ai_result
                ).all()

                images[each_line] = [obj.image_path for obj in data]

            if smart_filter:
                images = cls().smart_filter_images(images)

            return images


class UploadData(ImagePoolDatabase):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_images(self, uuids):
        images = {}
        with create_session(AMR_NIFI_TEST) as session:
            data = session.query(AmrRawData).filter(AmrRawData.uuid.in_(eval(uuids))).all()
            for obj in data:
                if obj.line_id not in images:
                    images[obj.line_id] = [obj.image_path]
                else:
                    images[obj.line_id].append(obj.image_path)

            return images


class UploadImageData(ImagePoolDatabase):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_images(self, uuids):
        images = {}
        with create_session(AI) as session:
            data = session.query(UploadData).filter(UploadData.uuid.in_(eval(uuids))).all()
            for obj in data:
                if obj.line_id not in images:
                    images[obj.line_id] = [obj.image_path]
                else:
                    images[obj.line_id].append(obj.image_path)

            return images