import os
import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A

transform = A.Compose([
    A.LongestMaxSize(max_size=512),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def plot_image_bboxes(image, bboxes, store_name = ""):
    # based on Peter's kernel
    # fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    for box in bboxes: 
        a = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (random.randint(100,255), random.randint(0,150), random.randint(100,255)), 3)
        
    if store_name != "":
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
    anno_path = 'new_augmentation/anno'
    image_path = 'new_augmentation/imgs'
    visualize(anno_path, image_path, 'new_augmentation/visualize')
    #anno_path = '../anno_train'
    #image_path = '../Lua/JPGimages/train'
    #visualize(anno_path, image_path, 'train')
    #anno_path = '../anno_test'
    #image_path = '../Lua/JPGimages/test'
    #visualize(anno_path, image_path, 'test')
    #anno_path = '../anno_val'
    #image_path = '../Lua/JPGimages/valid'
    #visualize(anno_path, image_path, 'val')
