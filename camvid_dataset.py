# %matplotlib inline
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import os
import shutil
import glob
# to avoid the warning of albumentations
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations
import matplotlib.pyplot as plt

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# all the classes that are present in the dataset
CLASSES = ['animal', 'archway', 'bicyclist', 'bridge', 'building', 'car',
           'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve',
           'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving',
           'parkingblock', 'pedestrian', 'road', 'road shoulder', 'sidewalk',
           'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight',
           'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void',
           'wall']

label_colors_list = [
    (64, 128, 64),  # animal
    (192, 0, 128),  # archway
    (0, 128, 192),  # bicyclist
    (0, 128, 64),  # bridge
    (128, 0, 0),  # building
    (64, 0, 128),  # car
    (64, 0, 192),  # car luggage pram...???...
    (192, 128, 64),  # child
    (192, 192, 128),  # column pole
    (64, 64, 128),  # fence
    (128, 0, 192),  # lane marking driving
    (192, 0, 64),  # lane maring non driving
    (128, 128, 64),  # misc text
    (192, 0, 192),  # motor cycle scooter
    (128, 64, 64),  # other moving
    (64, 192, 128),  # parking block
    (64, 64, 0),  # pedestrian
    (128, 64, 128),  # road
    (128, 128, 192),  # road shoulder
    (0, 0, 192),  # sidewalk
    (192, 128, 128),  # sign symbol
    (128, 128, 128),  # sky
    (64, 128, 192),  # suv pickup truck
    (0, 0, 64),  # traffic cone
    (0, 64, 64),  # traffic light
    (192, 64, 128),  # train
    (128, 128, 0),  # tree
    (192, 128, 192),  # truck/bus
    (64, 0, 64),  # tunnel
    (192, 192, 0),  # vegetation misc.
    (0, 0, 0),  # 0=background/void
    (64, 192, 0),  # wall
]

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


class CamVidDataset(Dataset):
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


def get_camvid_dataset(dataset_path):
    train_images = glob.glob(f"{dataset_path}/train/*")
    train_images.sort()
    train_segs = glob.glob(f"{dataset_path}/train_labels/*")
    train_segs.sort()
    valid_images = glob.glob(f"{dataset_path}/val/*")
    valid_images.sort()
    valid_segs = glob.glob(f"{dataset_path}/val_labels/*")
    valid_segs.sort()

    # Dataset Transoframtions which apply in loading phase
    train_image_transform = albumentations.Compose([
        albumentations.Resize(224, 224, always_apply=True),
        albumentations.Normalize(
            always_apply=True)
    ])
    valid_image_transform = albumentations.Compose([
        albumentations.Resize(224, 224, always_apply=True),
        albumentations.Normalize(
            always_apply=True)
    ])
    train_mask_transform = albumentations.Compose([
        albumentations.Resize(224, 224, always_apply=True),
    ])
    valid_mask_transform = albumentations.Compose([
        albumentations.Resize(224, 224, always_apply=True),
    ])

    # Define train and validation datasets
    return (CamVidDataset(train_images, train_segs, train_image_transform,
                                train_mask_transform,
                                label_colors_list,
                                classes=CLASSES)
                                ,
    CamVidDataset(valid_images, valid_segs, valid_image_transform,
                                valid_mask_transform,
                                label_colors_list,
                                classes=CLASSES))
                                
