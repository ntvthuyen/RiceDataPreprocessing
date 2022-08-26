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


DEVICE=torch.device('cuda')

train_anno_path = '../../new_annotation/anno_train'
train_image_path = '../../Lua/JPGimages/train'

test_anno_path = '../../new_annotation/anno_test'
test_image_path = '../../Lua/JPGimages/test'

val_anno_path = '../../new_annotation/anno_val'
val_image_path = '../../Lua/JPGimages/valid'
val_anno_path = '../../new_annotation/anno_test'
val_image_path = '../../Lua/JPGimages/test'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', default="resnet50", type=str,
                        help='select mode: restnet50, resnet18, deshangli')
    parser.add_argument('--seed', default=33, type=int,
                        help='global random seed')
    parser.add_argument('--fromcheckpoint', default="non", type=str,
                        help='Continue training from a checkpoint')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Continue training from a checkpoint')
    parser.add_argument('--batchsize', default=4, type=int,
                        help='Continue training from a checkpoint')

    args = parser.parse_args()
    print(args)
    SEED=args.seed
    pl.utilities.seed.seed_everything(SEED)

    train_transform = get_train_transform()
    test_transform = get_val_transform()
    net = None
    device_num = 0
    model = args.model
    epoch = args.epoch
    batch_size = args.batchsize

    if model == 'resnet50':
        if len(args.fromcheckpoint) != 0:
            net = FasterRCNNDetector(anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = batch_size)
        else:
            net = FasterRCNNDetector.load_from_checkpoint(args.fromcheckpoint,anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = batch_size)
    if model == 'dengshanli':
        #if len(args.fromcheckpoint) != 0:
        #    net = DengShanLiFasterRCNNDetector(anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = batch_size)
        #else:
        net = DengShanLiFasterRCNNDetector.load_from_checkpoint(args.fromcheckpoint,anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = batch_size)
    if model == 'resnet50fpnv2':
        if len(args.fromcheckpoint) != 0:
            net = ResNet18FasterRCNNDetector(anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = batch_size)
        else:
            net = ResNet18FasterRCNNDetector.load_from_checkpoint(args.fromcheckpoint,anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = batch_size)

    trainer = pl.Trainer(max_epochs=epoch, accelerator="gpu", devices=[device_num], callbacks=[pl.callbacks.progress.TQDMProgressBar(refresh_rate=50), pl.callbacks.ModelCheckpoint(every_n_epochs=30)])#, callbacks=[EarlyStopping(monitor="avg_val_iou", mode="max")])
    trainer.fit(net)
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%d :%H:%M:%S")

    if args.fromcheckpoint != '':
        trainer.save_checkpoint(current_time + model +'s' + str(SEED) + '_e' + str(epoch) + "_checkpoint.ckpt")
    else:
        trainer.save_checkpoint(args.fromcheckpoint[:-15] + '_s' + str(SEED) + '_e' + str(epoch) + "checkpoint.ckpt")

