import logging
import os
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
DATASET_NAME = 'cil-road-segmentation-2022'

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    logger.info('downloading dataset from kaggle')
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files('cil-road-segmentation-2022', path=RAW_DATA_PATH)

    logger.info('unzip dataset')
    zip_file = os.path.join(RAW_DATA_PATH, DATASET_NAME + '.zip')
    with zipfile.ZipFile(zip_file, 'r') as f:
        f.extractall(PROCESSED_DATA_PATH)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
