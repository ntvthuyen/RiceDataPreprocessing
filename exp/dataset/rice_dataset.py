import pandas as pd
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader, Dataset
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



class RiceDataset(Dataset):
    def __init__(self, anno_dir, image_dir, transforms=None, phase='train', anno_list = None):
        super().__init__()
        
        self.anno_dir = anno_dir
        if anno_list:
            self.anno_list=anno_list
        else:
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

        if self.phase == 'use':
            if self.transforms:
                sample = {
                    'image': image,
                }
                sample = self.transforms(**sample)
                image = sample['image']
            return image, image_id

        boxes = torch.as_tensor(records[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype = torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        if self.phase == 'train':
            class_id = None
            if '10018' in image_id:
                class_id = 1.0
            elif '12221' in image_id:
                class_id = 2.0
            else:
                class_id = 3.0
            class_ids = [class_id]*(len(records))
        else:
            try:
                class_ids = records['label'].to_list()          
            except:
                class_id = None
                if '10018' in image_id:
                    class_id = 1.0
                elif '12221' in image_id:
                    class_id = 2.0
                else:
                    class_id = 3.0
                class_ids = [class_id]*(len(records))
        
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['image_id'] = torch.tensor([idx])
        target['iscrowd'] = iscrowd
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)

            image = transformed['image']
            target["boxes"] = torch.as_tensor(transformed['bboxes'], dtype = torch.float32)
            target["labels"] = torch.as_tensor(transformed['class_labels'], dtype=torch.int64)
        '''
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.as_tensor(sample['bboxes'])
        '''
        return image, target

    def __len__(self):
        return len(self.anno_list)
