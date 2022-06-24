from ast import Str
import logging
import os
import zipfile
from pathlib import Path
from PIL import Image

from kaggle.api.kaggle_api_extended import KaggleApi

RAW_DATA_PATH = 'data/external'
PROCESSED_DATA_PATH = 'data/processed'
DATASET_NAME = 'massachusetts-roads-dataset'
RAW_IMAGE_PATH = 'data/processed/tiff'
IMAGE_PATH = 'data/processed/tiff/images'
GROUNDTRUTH_PATH = 'data/processed/tiff/groundtruth'
FINAL_IMAGE_PATH = 'data/processed/training/images'
FINAL_GROUNDTRUTH_PATH = 'data/processed/training/groundtruth'

def convert_images():
    Path(IMAGE_PATH).mkdir(parents=True, exist_ok=True)
    Path(GROUNDTRUTH_PATH).mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(RAW_IMAGE_PATH, topdown=False):
        for name in files:
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff" or os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
                if not os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".png"):
                    if(root.endswith("labels")):
                        path_out = os.path.join(GROUNDTRUTH_PATH, name)
                    else:
                        path_out = os.path.join(IMAGE_PATH, name)
                        
                    outfile = os.path.splitext(path_out)[0] + ".png"
                    im = Image.open(os.path.join(root, name))
                    im.save(outfile, "png", quality=100)

                os.remove(os.path.join(root, name))

def crop_images():
    index = (len([name for name in os.listdir(FINAL_IMAGE_PATH)]))

    image_path = os.path.join(FINAL_IMAGE_PATH, "satimage_")
    groundtruth_path = os.path.join(FINAL_GROUNDTRUTH_PATH, "satimage_")
    for root, dirs, files in os.walk(IMAGE_PATH, topdown=False):
        for name in files:
            image_in = os.path.join(IMAGE_PATH, name)
            image = Image.open(image_in)
            image = image.crop((150, 150, 1350, 1350))

            image.crop((0, 0, 400, 400)).save(image_path + str(index) + ".png", "png", quality=100)
            image.crop((400, 0, 800, 400)).save(image_path + str(index+1) + ".png", "png", quality=100)
            image.crop((800, 0, 1200, 400)).save(image_path + str(index+2) + ".png", "png", quality=100)
            image.crop((0, 400, 400, 800)).save(image_path + str(index+3) + ".png", "png", quality=100)
            image.crop((400, 400, 800, 800)).save(image_path + str(index+4) + ".png", "png", quality=100)
            image.crop((800, 400, 1200, 800)).save(image_path + str(index+5) + ".png", "png", quality=100)
            image.crop((0, 800, 400, 1200)).save(image_path + str(index+6) + ".png", "png", quality=100)
            image.crop((400, 800, 800, 1200)).save(image_path + str(index+7) + ".png", "png", quality=100)
            image.crop((800, 800, 1200, 1200)).save(image_path + str(index+8) + ".png", "png", quality=100)

            groundtruth_in = os.path.join(GROUNDTRUTH_PATH, name)
            groundtruth = Image.open(groundtruth_in)
            groundtruth = groundtruth.crop((150, 150, 1350, 1350))

            groundtruth.crop((0, 0, 400, 400)).save(groundtruth_path + str(index) + ".png", "png", quality=100)
            groundtruth.crop((400, 0, 800, 400)).save(groundtruth_path + str(index+1) + ".png", "png", quality=100)
            groundtruth.crop((800, 0, 1200, 400)).save(groundtruth_path + str(index+2) + ".png", "png", quality=100)
            groundtruth.crop((0, 400, 400, 800)).save(groundtruth_path + str(index+3) + ".png", "png", quality=100)
            groundtruth.crop((400, 400, 800, 800)).save(groundtruth_path + str(index+4) + ".png", "png", quality=100)
            groundtruth.crop((800, 400, 1200, 800)).save(groundtruth_path + str(index+5) + ".png", "png", quality=100)
            groundtruth.crop((0, 800, 400, 1200)).save(groundtruth_path + str(index+6) + ".png", "png", quality=100)
            groundtruth.crop((400, 800, 800, 1200)).save(groundtruth_path + str(index+7) + ".png", "png", quality=100)
            groundtruth.crop((800, 800, 1200, 1200)).save(groundtruth_path + str(index+8) + ".png", "png", quality=100)

            index += 9


    # image_path = os.path.join(FINAL_IMAGE_PATH, "satimage_")
    # for root, dirs, files in os.walk(IMAGE_PATH, topdown=False):
    #     for name in files:
    #         path_in = os.path.join(root, name)

    #         im = Image.open(path_in)
    #         im = im.crop((150, 150, 1350, 1350))

    #         im.crop((0, 0, 400, 400)).save(image_path + str(index) + ".png", "png", quality=100)
    #         im.crop((400, 0, 800, 400)).save(image_path + str(index+1) + ".png", "png", quality=100)
    #         im.crop((800, 0, 1200, 400)).save(image_path + str(index+2) + ".png", "png", quality=100)
    #         im.crop((0, 400, 400, 800)).save(image_path + str(index+3) + ".png", "png", quality=100)
    #         im.crop((400, 400, 800, 800)).save(image_path + str(index+4) + ".png", "png", quality=100)
    #         im.crop((800, 400, 1200, 800)).save(image_path + str(index+5) + ".png", "png", quality=100)
    #         im.crop((0, 800, 400, 1200)).save(image_path + str(index+6) + ".png", "png", quality=100)
    #         im.crop((400, 800, 800, 1200)).save(image_path + str(index+7) + ".png", "png", quality=100)
    #         im.crop((800, 800, 1200, 1200)).save(image_path + str(index+8) + ".png", "png", quality=100)

    #         index += 9

    # index = (len([name for name in os.listdir(FINAL_GROUNDTRUTH_PATH)]))

    # image_path = os.path.join(FINAL_GROUNDTRUTH_PATH, "satimage_")
    # for root, dirs, files in os.walk(GROUNDTRUTH_PATH, topdown=False):
    #     for name in files:
    #         path_in = os.path.join(root, name)

    #         im = Image.open(path_in)
    #         im = im.crop((150, 150, 1350, 1350))

    #         im.crop((0, 0, 400, 400)).save(image_path + str(index) + ".png", "png", quality=100)
    #         im.crop((400, 0, 800, 400)).save(image_path + str(index+1) + ".png", "png", quality=100)
    #         im.crop((800, 0, 1200, 400)).save(image_path + str(index+2) + ".png", "png", quality=100)
    #         im.crop((0, 400, 400, 800)).save(image_path + str(index+3) + ".png", "png", quality=100)
    #         im.crop((400, 400, 800, 800)).save(image_path + str(index+4) + ".png", "png", quality=100)
    #         im.crop((800, 400, 1200, 800)).save(image_path + str(index+5) + ".png", "png", quality=100)
    #         im.crop((0, 800, 400, 1200)).save(image_path + str(index+6) + ".png", "png", quality=100)
    #         im.crop((400, 800, 800, 1200)).save(image_path + str(index+7) + ".png", "png", quality=100)
    #         im.crop((800, 800, 1200, 1200)).save(image_path + str(index+8) + ".png", "png", quality=100)

    #         index += 9

def crop_and_save(path, folder, number):
    im = Image.open(path)
    if(im.size != (1500,1500)): 
        return

    im = im.crop((150, 150, 1350, 1350))
    print(os.path.splitext(path)[0] + "_1" + ".png")
    im.crop((0, 0, 400, 400)).save(os.path.splitext(path)[0] + "_1" + ".png", "png", quality=100)
    im.crop((400, 0, 800, 400)).save(os.path.splitext(path)[0] + "_2" + ".png", "png", quality=100)
    im.crop((800, 0, 1200, 400)).save(os.path.splitext(path)[0] + "_3" + ".png", "png", quality=100)
    im.crop((0, 400, 400, 800)).save(os.path.splitext(path)[0] + "_4" + ".png", "png", quality=100)
    im.crop((400, 400, 800, 800)).save(os.path.splitext(path)[0] + "_5" + ".png", "png", quality=100)
    im.crop((800, 400, 1200, 800)).save(os.path.splitext(path)[0] + "_6" + ".png", "png", quality=100)
    im.crop((0, 800, 400, 1200)).save(os.path.splitext(path)[0] + "_7" + ".png", "png", quality=100)
    im.crop((400, 800, 800, 1200)).save(os.path.splitext(path)[0] + "_8" + ".png", "png", quality=100)
    im.crop((800, 800, 1200, 1200)).save(os.path.splitext(path)[0] + "_9" + ".png", "png", quality=100)

    os.remove(path)


def move_images():
    # Should not be used

    Path(IMAGE_PATH).mkdir(parents=True, exist_ok=True)
    Path(GROUNDTRUTH_PATH).mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(RAW_IMAGE_PATH, topdown=False):
        for name in files:
            if(root.endswith("labels")):
                path = os.path.join(GROUNDTRUTH_PATH, name)
                print("label", root)
            else:
                path = os.path.join(IMAGE_PATH, name)
                print("image", root)

            outfile = os.path.splitext(path)[0] + ".png"
            im = Image.open(os.path.join(root, name))
            im.save(outfile, "png", quality=100)

def main():
    """ Runs data processing scripts to turn raw data from (../external) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    Path(RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    logger.info('downloading dataset from kaggle')
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('balraj98/massachusetts-roads-dataset', path=RAW_DATA_PATH)

    logger.info('extracting dataset from zip')
    zip_file = os.path.join(RAW_DATA_PATH, DATASET_NAME + '.zip')
    with zipfile.ZipFile(zip_file, 'r') as f:
        f.extractall(PROCESSED_DATA_PATH)
    
    convert_images()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    crop_images()

    