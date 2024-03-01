from flask import Flask, Blueprint
from flask_restx import Api, Namespace
from apps.routes.api import *


FLASK_API_PREFIX = os.getenv('FLASK_API_PREFIX')

# Api Init
api_bp = Blueprint('api_bp', __name__, url_prefix=FLASK_API_PREFIX)
api = Api(api_bp, version='1.0', title='Training Platform Backend APIs', doc='/docs')


def register_routes(name, path, routes):
    namespace = Namespace(name, path=path)
    for router in routes:
        namespace.add_resource(router['resource'], router['urls'])
        if 'doc' in router.keys() and isinstance(router['doc'], dict):
            for key, value in router['doc'].items():
                if key in router['resource']().__dir__():
                    method = getattr(router['resource'], key)
                    if 'params' in value.keys() and isinstance(value['params'], dict):
                        namespace.doc(params=value['params'])(method)
                    if 'payload' in value.keys() and isinstance(value['payload'], dict):
                        namespace.expect(namespace.model(router['resource']().__class__.__name__ + method.__name__, value['payload']))(method)
            
    api.add_namespace(namespace)

def init_app(app: Flask):
    app.register_blueprint(api_bp)
    

# Register Routes

register_routes('Status', '/status', status_routes)
register_routes('Data', '/data', data_routes)
register_routes('Utils', '/utils', utils_routes)
register_routes('Models', '/models', models_routes)
register_routes('Info', '/info', info_routes)
register_routes('Category', '/category', category_routes)