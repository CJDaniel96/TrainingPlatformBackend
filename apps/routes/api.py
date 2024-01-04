from apps.controllers.model_controller import *
from apps.controllers.record_controller import *
from apps.controllers.amr_raw_data_controller import *
from apps.controllers.image_pool_controller import *
from apps.controllers.utils_controller import *
from apps.controllers.category_controller import *
from apps.docs.category_doc import *
from apps.docs.amr_raw_data_doc import *
from apps.docs.record_doc import *
from apps.docs.inference_doc import *
from apps.docs.image_pool_doc import *
from apps.docs.utils_doc import *
from apps.docs.category_doc import *


# Routes Settings

status_routes = [
    {
        'resource': IRIRecordStatusController,
        'urls': '/iri_record',
        'doc': iri_record_status_doc
    },
    {
        'resource': URDRecordStatusController,
        'urls': '/urd_record',
        'doc': urd_record_status_doc
    }
]

data_routes = [
    {
        'resource': AssignLightImageController,
        'urls': '/images/assignlight',
        'doc': image_controller_doc
    },
    {
        'resource': ImageController,
        'urls': '/images',
        'doc': image_controller_doc
    },
    {
        'resource': UuidsImageController,
        'urls': '/images/uuids',
        'doc': uuids_image_controller_doc
    }
]

category_mapping_routes = [
    {
        'resource': CategoryMappingLabelsController,
        'urls': '/labels',
        'doc': category_mapping_labels_controller_doc
    },
    {
        'resource': CategoryMappingOKLabelsController,
        'urls': '/ok_labels',
        'doc': category_mapping_ok_labels_controller_doc
    }
]

utils_routes = [
    {
        'resource': DateTimeBarcodeController,
        'urls': '/barcode',
        'doc': datatime_barcode_doc
    },
    {
        'resource': UnzipDataController,
        'urls': '/unzip_data',
        'doc': unzip_data_doc
    },
    {
        'resource': OutputXMLController,
        'urls': '/output_xmls',
        'doc': output_xmls_doc
    },
    {
        'resource': XMLToYOLOFormatController,
        'urls': '/xml2yolo',
        'doc': xml_to_yolo_doc
    }
]

models_routes = [
    {
        'resource': Yolov5InferenceController,
        'urls': '/yolov5/inference',
        'doc': yolov5_inference_doc
    },
    {
        'resource': Yolov5TrainingController,
        'urls': '/yolov5/train',
        'doc': yolov5_train_doc
    }
]

image_pool_routes = [
    {
        'resource': ImagePoolInfoController,
        'urls': '/info',
        'doc': image_pool_doc
    }
]