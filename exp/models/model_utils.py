import pandas as pd
import numpy as np
import cv2
import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision
from torchvision.ops import box_iou, MultiScaleRoIAlign
from torchvision import models



def get_val_transform(max_size=1024):
    return A.Compose([
        A.SmallestMaxSize(max_size=1000),
        A.LongestMaxSize(max_size=2500),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))

def get_vil_transform(max_size=1024):
    return A.Compose([
        A.SmallestMaxSize(max_size=1000),
        A.LongestMaxSize(max_size=2500)
     ])

def get_train_transform(max_size=1024):
    return A.Compose([
        A.SmallestMaxSize(max_size=1000),
        A.LongestMaxSize(max_size=2500),
        A.Flip(0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomSizedBBoxSafeCrop(width=1000,height=1000, p=0.3),
        A.Rotate(limit=45, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
        ToTensorV2(p=1.0)

    ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))

def get_valid_transform():
    return A.Compose([
#         A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def format_prediction_string(labels, boxes, scores):
    pred_strings = []
    for j in zip(labels, scores, boxes):
        pred_strings.append("{0} {1:.4f} {2} {3} {4} {5}".format(
            j[0], j[1], j[2][0], j[2][1], j[2][2], j[2][3]))
    return " ".join(pred_strings)

