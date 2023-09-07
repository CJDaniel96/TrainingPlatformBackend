import http
import os
import requests
from app.services.logging_service import Logger
from data.config import ORIGIN_DATASETS_DIR



class ImagePool:
    @classmethod
    def download(cls, image_pool, image_list):
        download_url = f'http://{image_pool.ip}/imagesinzip'
        download_proxies = {'http': download_url}
        resp = requests.post(download_url, proxies=download_proxies, json={
            "paths": image_list
        })
        if resp.status_code == 200:
            Logger.info(f'Downloading Images from {image_pool.prefix} Image Pool...')
            with open(os.path.join(ORIGIN_DATASETS_DIR, image_pool.line + '.zip'), 'wb') as zip_file:
                zip_file.write(resp.content)
            Logger.info(f'Download Images from {image_pool.prefix} image pool finish!')
        elif resp.status_code == 403:
            Logger.error(http.HTTPStatus.FORBIDDEN)