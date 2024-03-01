from scripts.monitor_script import Monitor
from config import *
from pathlib import Path
import requests
import argparse

from torchvision.models import MobileNetV2
def checkenv():
    # Check Models Folder
    for project, _ in TRAINING_FLOW[SITE].items():
        for folder in ['inference', 'train']:
            Path(MODELS_DIR, project, folder).mkdir(parents=True, exist_ok=True)
            
def runscript():
    monitor = Monitor()
    # record:dict = requests.get('http://localhost:5000/api/v1/status/iri_record', params={'id': 100}).json()
    # monitor.cls_initialized(**record['data'])
    monitor.run()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['runscript', 'checkenv', 'insertvalimgs'], help='the command to run')
    parser.add_argument('--project', type=str, default='', help='provide project to insert validated images')
    parser.add_argument('--group-type', type=str, default='', help='provide group type to insert validated images')
    parser.add_argument('--image-type', type=str, default='jpg', help='provide image type to insert validated images')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt()
    if opt.command == 'runscript':
        runscript()
    elif opt.command == 'checkenv':
        checkenv()