import os

import albumentations as A
import cv2
import zipfile
import numpy as np
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class RoadSegDataset(Dataset):
    def __init__(self, img_path, mask_path=None, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.img_names = os.listdir(img_path)

    def wrapped_transform(self, image, mask):
        if image is not None and mask is not None:
            return self.transform(image=image, mask=mask)
        if image is not None:
            return self.transform(image=image)
        else:
            raise AttributeError("No image data available")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]

        image = cv2.imread(os.path.join(self.img_path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mask_path:
            mask = cv2.imread(os.path.join(self.mask_path, img_name), cv2.IMREAD_UNCHANGED)
            mask //= 255

        if self.transform:
            sample = self.wrapped_transform(image, mask)
            image, mask = sample.get("image"), sample.get("mask")
        return image, mask


class RoadSegDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../../data", batch_size=8, val_split=0.1, shuffle=True, random_seed=42, num_workers=4):
        self.batch_size = batch_size
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.data_dir = data_dir

        self.train_img_path = os.path.join(data_dir, "processed/training/images")
        self.train_mask_path = os.path.join(data_dir, "processed/training/groundtruth")
        self.test_img_path = os.path.join(data_dir, "processed/test/images")

        self.train_transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                           b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ]
        )
        self.val_transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ]
        )
        self.test_transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ]
        )

    def prepare_data(self):
        processed_dir = os.path.join(self.data_dir, "processed")
        raw_dir = os.path.join(self.data_dir, "raw")
        if not os.path.isdir(processed_dir):
            zip_file = os.path.join(raw_dir, "cil-road-segmentation-2022.zip")
            with zipfile.ZipFile(zip_file, 'r') as f:
                f.extractall(processed_dir)

    def setup(self, stage=None):
        self.train_dataset = RoadSegDataset(
            img_path=self.train_img_path, mask_path=self.train_mask_path, transform=self.train_transform)
        self.val_dataset = RoadSegDataset(
            img_path=self.train_img_path, mask_path=self.train_mask_path, transform=self.val_transform)
        self.test_dataset = RoadSegDataset(
            img_path=self.test_img_path, transform=self.test_transform)

        train_len = len(self.train_dataset)
        idx = list(range(train_len))
        split = int(np.floor(self.val_split * train_len))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(idx)

        train_idx, val_idx = idx[split:], idx[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=self.val_sampler, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
