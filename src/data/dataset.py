import os
from torch.utils.data import Dataset
import cv2

class RoadSegDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.img_names = os.listdir(img_path)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]

        image = cv2.imread(os.path.join(self.img_path, img_name))
        mask = cv2.imread(os.path.join(self.mask_path, img_name))
        sample = {"image": image, "mask": mask}

        if self.transform:
            sample = self.transform(sample)
        
        return sample