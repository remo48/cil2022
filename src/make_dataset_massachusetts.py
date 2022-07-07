import logging
import os
import shutil
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from kaggle.api.kaggle_api_extended import KaggleApi

class MassachusettsDataset():
    def __init__(self, data_path, dataset_name="massachusetts"):
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        self.logger = logging.getLogger(__name__)

        self.raw_data_path = os.path.join(data_path, "raw")
        self.processed_data_path = os.path.join(data_path, "processed")
        self.raw_image_path = os.path.join(self.processed_data_path, "tiff")
        self.image_path = os.path.join(self.raw_image_path, "image")
        self.groundtruth_path = os.path.join(self.raw_image_path, "groundtruth")

        self.dataset_name = dataset_name
        self.final_image_path = os.path.join(self.processed_data_path, self.dataset_name, "images")
        self.final_groundtruth_path = os.path.join(self.processed_data_path, self.dataset_name, "groundtruth")

    def download(self):
        Path(self.raw_data_path).mkdir(parents=True, exist_ok=True)
        Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)

        self.logger.info("Download dataset to %s", self.raw_data_path)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('balraj98/massachusetts-roads-dataset', path=self.raw_data_path)

        self.logger.info("Extract files to %s", self.processed_data_path)
        zip_file = os.path.join(self.raw_data_path, 'massachusetts-roads-dataset.zip')
        with zipfile.ZipFile(zip_file, 'r') as f:
            f.extractall(self.processed_data_path)

    def process_images(self):
        self.logger.info("Convert images from .tiff to .png")
        self._convert_images()

        self.logger.info("Crop images")
        self._crop_images()

        self.logger.info("Remove uncropped images")
        shutil.rmtree(self.raw_image_path)


    def _convert_images(self):
        Path(self.image_path).mkdir(parents=True, exist_ok=True)
        Path(self.groundtruth_path).mkdir(parents=True, exist_ok=True)

        for d in os.listdir(self.raw_image_path):
            img_path = os.path.join(self.raw_image_path, d)
            self.logger.info("Convert files in path %s", img_path)
            for name in tqdm(os.listdir(img_path)):
                if os.path.splitext(os.path.join(img_path, name))[1].lower() == ".tiff" or os.path.splitext(os.path.join(img_path, name))[1].lower() == ".tif":
                    if not os.path.isfile(os.path.splitext(os.path.join(img_path, name))[0] + ".png"):
                        if(img_path.endswith("labels")):
                            path_out = os.path.join(self.groundtruth_path, name)
                        else:
                            path_out = os.path.join(self.image_path, name)
                            
                        outfile = os.path.splitext(path_out)[0] + ".png"
                        im = Image.open(os.path.join(img_path, name))
                        im.save(outfile, "png", quality=100)
                        im.close()

                    os.remove(os.path.join(img_path, name))

    def _crop_images(self):
        Path(self.final_image_path).mkdir(parents=True, exist_ok=True)
        Path(self.final_groundtruth_path).mkdir(parents=True, exist_ok=True)
        index = (len([name for name in os.listdir(self.final_image_path)]))

        image_path = os.path.join(self.final_image_path, "satimage_")
        groundtruth_path = os.path.join(self.final_groundtruth_path, "satimage_")
        for name in tqdm(os.listdir(self.image_path)):
            image_in = os.path.join(self.image_path, name)
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

            groundtruth_in = os.path.join(self.groundtruth_path, name)
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


def main():
    """ Runs data processing scripts to turn raw data from (../external) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dataset = MassachusettsDataset("data")
    #dataset.download()
    dataset.process_images()

if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    main()

    