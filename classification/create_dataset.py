import torch
from torch.utils import data
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2


class Mydataset(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        labels = int(label)
        data = self.transforms(image=img)

        return data['image'], labels

    def __len__(self):
        return len(self.imgs)

class Mydataset_infer(data.Dataset):
    def __init__(self, img_paths,  transform):
        self.imgs = img_paths
        self.transforms = transform

    def __getitem__(self, index):
        img_path_here = self.imgs[index]
        # img = np.load(img_path_here)#[:,:,::-1]
        img  = cv2.imread(img_path_here)[:,:,::-1]
        data = self.transforms(image=img)

        return img_path_here,data['image']

    def __len__(self):
        return len(self.imgs)




def for_train_transform():
    # aug_size=int(size/10)
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.1), rotate_limit=40, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
    return train_transform


test_transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.)