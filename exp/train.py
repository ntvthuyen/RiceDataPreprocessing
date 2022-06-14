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

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchmetrics import AveragePrecision
from models.faster_rcnn import *
from dataset.rice_dataset import *
SEED=2484
DEVICE=torch.device('cuda')
pl.utilities.seed.seed_everything(SEED)

net = FasterRCNNDetector(anno_dir='/home/lightkeima/Project/new_annotation/anno_train', image_dir='/home/lightkeima/Project/Lua/JPGimages/train', batch_size=3)
trainer = pl.Trainer(max_epochs=5, accelerator=['cpu']
#gpus=0
, progress_bar_refresh_rate=100)
trainer.fit(net)

