from os import getenv
from flask import Flask
from apps import routes, databases
from config import ENV_CONFIG


def create_app():
    app = Flask(__name__)
    app.config.update(ENV_CONFIG[getenv('FLASK_ENV')].__dict__)
    
    # App Initial
    databases.init_app(app)
    routes.init_app(app)
    
    return app