import argparse
from app.controllers.check_environment_controller import CheckEnvironmentController
from app.controllers.manage_controller import ManageController
from app.main import app_run
from app.services.logging_service import Logger


def insert_validated_images_to_db(project, group_type, image_type):
    Logger.info('Insert Validated Images To Database')
    ManageController.insert_validated_images_to_db(project, group_type, image_type)

def checkenv():
    Logger.info('Check Datasets Environment')
    # Check Enviroment
    CheckEnvironmentController.check_data_environment()
    CheckEnvironmentController.check_model_environment()

def runserver():
    Logger.info('Training Platform serving...')

    while True:
        app_run()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['runserver', 'checkenv', 'insertvalimgs'], help='the command to run')
    parser.add_argument('--project', type=str, default='', help='provide project to insert validated images')
    parser.add_argument('--group-type', type=str, default='', help='provide group type to insert validated images')
    parser.add_argument('--image-type', type=str, default='jpg', help='provide image type to insert validated images')
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = parse_opt()
    if opt.command == 'runserver':
        runserver()
    elif opt.command == 'checkenv':
        checkenv()
    elif opt.command == 'insertvalimgs':
        insert_validated_images_to_db(opt.project, opt.group_type, opt.image_type)
