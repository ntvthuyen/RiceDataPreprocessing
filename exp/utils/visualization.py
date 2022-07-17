import os
import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A
from models.model_utils import get_vil_transform
transform = get_vil_transform()
image_transform = A.Compose([
        A.LongestMaxSize(max_size=1024)
    ])

def plot_image_bboxes(image, bboxes, color, line):    
    for box in bboxes: 
        a = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, line)
    return image
    
def visualize_single_image(image, prediction, target, plabels, tlables, store_name = ""):
    image = transform(image=image)['image']
    image = plot_image_bboxes(image, prediction, (0,0,255), 4)
    image = plot_image_bboxes(image, target, (255,0,0), 3)
    image = image_transform(image=image)['image']
    cv2.imwrite(store_name, image)


def visualize(anno_path, image_path, folder):
    anno_list = sorted(os.listdir(anno_path))
    for anno in range(len(anno_list)):
        # print(anno_list[anno])
        image = cv2.imread(os.path.join(image_path, anno_list[anno][:-3]+'jpg'))
        boxes_path = os.path.join(os.path.join(anno_path, anno_list[anno]))
        boxes = []
        labels = []
        df = pd.read_csv(boxes_path)
        for index, row in df.iterrows():
            labels.append(0)
            boxes.append([row['xmin'], row['ymin'],  row['xmax'], row['ymax']])
        # print(boxes)
        target = {}
        transformed = transform(image=image, bboxes=boxes, class_labels=labels)
        transformed_image = transformed['image']
        target["boxes"] = transformed['bboxes']
        plot_image_bboxes(transformed_image, target["boxes"], os.path.join(folder, anno_list[anno][:-3]+'jpg'))

if __name__=='__main__': 
    anno_path = '../anno_train'
    image_path = '../Lua/JPGimages/train'
    visualize(anno_path, image_path, 'train')
    anno_path = '../anno_test'
    image_path = '../Lua/JPGimages/test'
    visualize(anno_path, image_path, 'test')
    anno_path = '../anno_val'
    image_path = '../Lua/JPGimages/valid'
    visualize(anno_path, image_path, 'val')
