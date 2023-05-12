import argparse
from app.controllers.check_environment_controller import CheckEnvironmentController
from app.main import app_run
from app.services.logging_service import Logger


def checkenv():
    Logger.info('Check Datasets Environment')
    # Check Enviroment
    CheckEnvironmentController.check_datasets_environment()

def runserver():
    Logger.info('Training Platform serving...')

    while True:
        app_run()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['runserver', 'checkenv'], help='the command to run')
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = parse_opt()
    if opt.command == 'runserver':
        runserver()
    elif opt.command == 'checkenv':
        checkenv()
