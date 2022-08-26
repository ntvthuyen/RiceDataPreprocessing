import os
import pandas as pd
import cv2
import numpy as np

def convert_passcalVOC_2_yolo(W, H, xmin, ymin, xmax, ymax):
    w = float(xmax - xmin)/W
    h = float(ymax - ymin)/H
    x = float(xmin)/W + w/2
    y = float(ymin)/H + h/2   
    return [x, y, w, h]




def visualize(anno_path, image_path):
    anno_list = sorted(os.listdir(anno_path))
    for anno in range(len(anno_list)):
        if anno_list[anno][0] == '.':
            continue
        image = cv2.imread(os.path.join(
            image_path, anno_list[anno][:-3]+'jpg'))
        w = image.shape[1]
        h = image.shape[0]
        boxes_path = os.path.join(os.path.join(anno_path, anno_list[anno][:-3]+'csv'))
        boxes = []
        labels = []
        df = pd.read_csv(boxes_path)
        for index, row in df.iterrows():
            if 'label' not in df.columns:
                class_id = None
                if '10018' in anno_list[anno]:
                    class_id = 0
                elif '12221' in anno_list[anno]:
                    class_id = 2
                else:
                    class_id = 1
            else:
                class_id = int(row['label'] - 1)

            labels.append(class_id)
            boxes.append(convert_passcalVOC_2_yolo(w,h,row['xmin'], row['ymin'],  row['xmax'], row['ymax']))
        if len(boxes) == 0:
            continue
        boxes = np.array(boxes)
        data = {'label': labels, 'xmin': boxes[:, 0], 'ymin': boxes[:, 1],
                'xmax': boxes[:, 2], 'ymax': boxes[:, 3]}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join('new_augmentation_yolo', anno_list[anno][:-3] + 'txt'), header=False, sep= ' ',index=False)

       
anno_path = 'new_augmentation/anno'
image_path = 'new_augmentation/imgs'
visualize(anno_path, image_path)
