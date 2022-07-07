import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../dataset/')
sys.path.insert(1, '.')
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

from pl_bolts.models.detection.faster_rcnn import create_fasterrcnn_backbone
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from model_utils import *
from dataset.rice_dataset import RiceDataset

if _TORCHVISION_AVAILABLE:
    from torchvision.models.detection.faster_rcnn import FasterRCNN as torchvision_FasterRCNN
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
    from torchvision.ops import box_iou
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

def _evaluate_iou(target, pred):
    """Evaluate intersection over union (IOU) for target from dataset and output prediction from model."""
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()

class FasterRCNNDetector(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 3
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.learning_rate = 1e-3
        self.transform = kwargs['transform']
        self.test_transform = kwargs['test_transform']
        self.batch_size = kwargs['batch_size']
        self.anno_dir = kwargs['anno_dir']
        self.image_dir = kwargs['image_dir']

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        self.train_dataset = RiceDataset(self.anno_dir, self.image_dir, self.transform)
        self.val_dataset = RiceDataset(self.anno_dir, self.image_dir, self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "log": loss_dict}
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay=1e-5, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0, verbose=True)
        return [optimizer], [scheduler]

