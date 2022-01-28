"""
author: Min Seok Lee and Wooseok Shin
Github repo: https://github.com/Karel911/TRACER
"""

import cv2
import glob
import torch
import numpy as np
import albumentations as albu
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, phase: str = 'train', transform=None, seed=None):
        self.images = sorted(glob.glob(img_folder + '/*'))
        self.gts = sorted(glob.glob(gt_folder + '/*'))
        self.transform = transform

        train_images, val_images, train_gts, val_gts = train_test_split(self.images, self.gts, test_size=0.05,
                                                                        random_state=seed)
        if phase == 'train':
            self.images = train_images
            self.gts = train_gts
        elif phase == 'val':
            self.images = val_images
            self.gts = val_gts
        else:  # Testset
            pass

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.gts[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            augmented = self.transform(image=image, masks=[mask])
            image = augmented['image']
            mask = np.expand_dims(augmented['masks'][0], axis=0)  # (1, H, W)
            mask = mask / 255.0

        return image, mask

    def __len__(self):
        return len(self.images)


class Test_DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, transform=None):
        self.images = sorted(glob.glob(img_folder + '/*'))
        self.gts = sorted(glob.glob(gt_folder + '/*'))
        self.transform = transform

    def __getitem__(self, idx):
        image_name = Path(self.images[idx]).stem
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.gts[idx], original_size, image_name

    def __len__(self):
        return len(self.images)


def get_loader(img_folder, gt_folder, phase: str, batch_size, shuffle,
               num_workers, transform, seed=None):
    if phase == 'test':
        dataset = Test_DatasetGenerate(img_folder, gt_folder, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        dataset = DatasetGenerate(img_folder, gt_folder, phase, transform, seed)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                 drop_last=True)

    print(f'{phase} length : {len(dataset)}')

    return data_loader


def get_train_augmentation(img_size, ver):
    if ver == 1:
        transforms = albu.Compose([
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    if ver == 2:
        transforms = albu.Compose([
            albu.OneOf([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomRotate90()
            ], p=0.5),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=0.5),
            albu.OneOf([
                albu.MotionBlur(blur_limit=5),
                albu.MedianBlur(blur_limit=5),
                albu.GaussianBlur(blur_limit=5),
                albu.GaussNoise(var_limit=(5.0, 20.0)),
            ], p=0.5),
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return transforms


def get_test_augmentation(img_size):
    transforms = albu.Compose([
        albu.Resize(img_size, img_size, always_apply=True),
        albu.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transforms


def gt_to_tensor(gt):
    gt = cv2.imread(gt)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) / 255.0
    gt = np.where(gt > 0.5, 1.0, 0.0)
    gt = torch.tensor(gt, device='cuda', dtype=torch.float32)
    gt = gt.unsqueeze(0).unsqueeze(1)

    return gt
