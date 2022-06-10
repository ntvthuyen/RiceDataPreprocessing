import pandas as pd
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader, Dataset
import torch

class RiceDataset(Dataset):
    def __init__(self, anno_dir, image_dir, transforms=None, phase='train'):
        super().__init__()

        self.anno_dir = anno_dir
        self.anno_list = sorted(os.listdir(anno_dir))
        self.image_dir = image_dir
        self.transforms = transforms
        self.phase = phase

    def __getitem__(self, idx):
        
        image_id = self.anno_list[idx][:-4]
        records = pd.read_csv(os.path.join(self.anno_dir, self.anno_list[idx]))

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.phase == 'test':
            if self.transforms:
                sample = {
                    'image': image,
                }
                sample = self.transforms(**sample)
                image = sample['image']
            return image, image_id

        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        # all the labels are shifted by 1 to accomodate background
        class_id = None
        if '10018' in image_id:
            class_id = 0
        elif '12221' in image_id:
            class_id = 1
        else:
            class_id = 2
        labels = torch.squeeze(torch.as_tensor((class_id,), dtype=torch.int64))
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['image_id'] = torch.tensor([idx])
        target['iscrowd'] = iscrowd
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.as_tensor(sample['bboxes'])

        return image, target

    def __len__(self):
        return len(self.anno_list)

