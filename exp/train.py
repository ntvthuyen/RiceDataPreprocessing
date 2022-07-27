import sys
import argparse
sys.path.insert(1,'models')
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

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchmetrics import AveragePrecision
from models.faster_rcnn import *
from dataset.rice_dataset import *
from models.model_utils import *
from models.dengshanli_faster_rcnn import *
from models.resnet18_faster_rcnn import *

SEED=33
DEVICE=torch.device('cuda')
pl.utilities.seed.seed_everything(SEED)

train_anno_path = '../../new_annotation/anno_train'
train_image_path = '../../Lua/JPGimages/train'

test_anno_path = '../../new_annotation/anno_test'
test_image_path = '../../Lua/JPGimages/test'

val_anno_path = '../../new_annotation/anno_val'
val_image_path = '../../Lua/JPGimages/valid'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', default="resnet50", type=str,
                        help='select mode: restnet50, resnet18, deshangli')

    args = parser.parse_args()
    print(args)

    train_transform = get_train_transform()
    test_transform = get_train_transform()
    net = None

    model = args.model

    if model == 'resnet50':
        net = FasterRCNNDetector(anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = 4)
    if model == 'dengshanli':
        net = DengShanLiFasterRCNNDetector(anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = 2)
    if model == 'resnet18':
        net = ResNet18FasterRCNNDetector(anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = 4)
    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=[1], progress_bar_refresh_rate=100)#, callbacks=[EarlyStopping(monitor="avg_val_iou", mode="max")])
    trainer.fit(net)
    trainer.save_checkpoint("checkpoint.ckpt")
