from flask_restx import fields


class Eval(fields.Raw):
    def format(self, value):
        return eval(value) if type(value) is str else None
    

class DateBarcodeFormat(fields.Raw):
    def format(self, value):
        return value.strftime('%Y%m%d%H%M%S')