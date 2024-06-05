from apps.serializers import DateBarcodeFormat
from flask_restx import fields


datetime_serializer = {
    'datetime_barcode': DateBarcodeFormat
}

unzip_data_serializer = {
    'unzip_data_folder': fields.String
}

xml_to_yolo_serializer = {
    'labels_path': fields.String
}