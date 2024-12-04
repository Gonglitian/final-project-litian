# %matplotlib inline
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import os
import shutil
import glob
from albumentations.pytorch import ToTensorV2
# to avoid the warning of albumentations
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

CLASSES = ['unlabeled',
           'roads',
           'sidewalks',
           'building',
           'wall',
           'fence',
           'pole',
           'trafficlight',
           'trafficsign',
           'vegetation',
           'terrain',
           'sky',
           'pedestrian',
           'rider',
           'car',
           'truck',
           'bus',
           'train',
           'motorcycle',
           'bicycle',
           'static',
           'dynamic',
           'other',
           'water',
           'roadline',
           'ground',
           'bridge',
           'railtrack',
           'guardrail']

label_colors_list = [(0, 0, 0),
 (128, 64, 128),
 (244, 35, 232),
 (70, 70, 70),
 (102, 102, 156),
 (190, 153, 153),
 (153, 153, 153),
 (250, 170, 30),
 (220, 220, 0),
 (107, 142, 35),
 (152, 251, 152),
 (70, 130, 180),
 (220, 20, 60),
 (255, 0, 0),
 (0, 0, 142),
 (0, 0, 70),
 (0, 60, 100),
 (0, 80, 100),
 (0, 0, 230),
 (119, 11, 32),
 (110, 190, 160),
 (170, 120, 50),
 (55, 90, 80),
 (45, 60, 150),
 (157, 234, 50),
 (81, 0, 81),
 (150, 100, 100),
 (230, 150, 140),
 (180, 165, 180)]

def get_label_mask(mask, class_values):
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


class CarlaDataset(Dataset):
    CLASSES = CLASSES

    def __init__(self, path_images, path_segs, image_transform, mask_transform, label_colors_list, classes):
        print(f"TRAINING ON CLASSES: {classes}")

        self.path_images = path_images
        self.path_segs = path_segs
        self.label_colors_list = label_colors_list
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(
            cls.lower()) for cls in classes]

    def __len__(self):
        return len(self.path_images)-1

    def __getitem__(self, index):
        image = np.array(Image.open(self.path_images[index]).convert('RGB'))
        mask = np.array(Image.open(self.path_segs[index]).convert('RGB'))

        image = self.image_transform(image=image)['image']
        mask = self.mask_transform(image=mask)['image']

        # get the colored mask labels
        mask = get_label_mask(mask, self.class_values)

        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long)
        # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        return image, mask

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def get_carla_dataset(dataset_root):
    # train_images = glob.glob(f"{dataset_path}//*")
    train_txt = os.path.join(
        dataset_root, "split_data/train.txt")
    train_images = []
    train_labels = []
    with open(train_txt, "r") as f:
        lines = f.readlines()
        for path in lines:
            train_images.append(dataset_root + '/ori_data/' + path.strip())
            train_labels.append(dataset_root + '/ori_label/' + path.strip())
    # get val
    val_txt = os.path.join(
        dataset_root, "split_data/val.txt")
    valid_images = []
    valid_labels = []
    with open(val_txt, "r") as f:
        lines = f.readlines()
        for path in lines:
            valid_images.append(dataset_root + '/ori_data/' + path.strip())
            valid_labels.append(dataset_root + '/ori_label/' + path.strip())

    train_images.sort()
    train_labels.sort()
    valid_images.sort()
    valid_labels.sort()

    # Dataset Transoframtions which apply in loading phase
    def get_transforms(train=True):
        if train:
            return A.Compose([
                A.Resize(400, 520),
                A.RandomCrop(height=352, width=480),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                            saturation=0.2, hue=0.1),
                A.Normalize(mean=(0.390, 0.405, 0.414), std=(0.274, 0.285, 0.297)),
                # ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(352, 480),
                A.Normalize(mean=(0.390, 0.405, 0.414), std=(0.274, 0.285, 0.297)),
                # ToTensorV2()
            ])
    
    train_image_transform = get_transforms(train=True)
    valid_image_transform = get_transforms(train=False)
    train_mask_transform = get_transforms(train=True)
    valid_mask_transform = get_transforms(train=False)

    # Define train and validation datasets
    return (CarlaDataset(train_images, train_labels, train_image_transform,
                          train_mask_transform,
                          label_colors_list,
                          classes=CLASSES),
            CarlaDataset(valid_images, train_labels, valid_image_transform,
                          valid_mask_transform,
                          label_colors_list,
                          classes=CLASSES))
