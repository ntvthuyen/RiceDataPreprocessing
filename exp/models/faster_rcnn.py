import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../dataset/')

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
from pytorch_lightning.metrics import AveragePrecision

from rice_dataset import RiceDataset

from utils import get_train_transform, get_valid_transform, collate_fn, format_prediction_string

class FasterRCNNDetector(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 3
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.learning_rate = 1e-3
        self.batch_size = 4
        self.anno_dir = kwargs.anno_dir
        self.image_dir = kwargs.image_dir

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        self.train_dataset = RiceDataset(self.anno_dir, self.image_dir, get_train_transform())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('Loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay=1e-5, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0, verbose=True)
        return [optimizer], [scheduler]

