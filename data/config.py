# Database Settings

DATABASES = {
    'ai': {
        'ENGINE': 'postgresql',
        'NAME': 'ai',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': '10.0.4.188',
        'PORT': '5432',
    }, 
    'amr_nifi_test': {
        'ENGINE': 'postgresql',
        'NAME': 'amr_nifi_test',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': '10.0.4.188',
        'PORT': '5432',
    }, 
    'cvat': {
        'ENGINE': 'postgresql',
        'NAME': 'cvat',
        'USER': 'root',
        'PASSWORD': 'password',
        'HOST': '10.0.13.80',
        'PORT': '5432',
    }
}

# Datasets Environment Settings

ORIGIN_DATASETS_DIR = './data/datasets/origin_datasets'

ORIGIN_DATASETS_FOLDER_PROFIX = 'org_data'

RAW_DATASETS_DIR = './data/datasets/raw_datasets'