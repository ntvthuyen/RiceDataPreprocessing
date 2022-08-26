import os
import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A

transform = A.Compose([
    A.LongestMaxSize(max_size=512),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def plot_image_bboxes(image, bboxes, store_name=""):
    # based on Peter's kernel
    # fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in bboxes:
        a = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(
            box[3])), (random.randint(100, 255), random.randint(0, 150), random.randint(100, 255)), 3)

    if store_name != "":
        cv2.imwrite(store_name, image)


relabel = [
    "10018_00005724",
    "10018_00005753",
    "10018_00005759",
    "12131_00003788",
    "12131_00003873",
    "12131_00003930",
    "12131_00003939",
    "12131_00003976",
    "12131_00003978",
    "12131_00003979",
    "12131_00003988",
    "12131_00003998",
    "12131_00004009",
    "12131_00004034"]


def convert_yolo_2_pascalVOC(W, H, x, y, w, h):
    xmin = int(x*W - w*W/2)
    xmax = int(x*W + w*W/2)
    ymin = int(y*H - h*H/2)
    ymax = int(y*H + h*H/2)
    return [xmin, ymin, xmax, ymax]


def visualize(anno_path, image_path, folder, anno_new_path):
    anno_list = sorted(os.listdir(anno_new_path))
    for anno in range(len(anno_list)):
        print(anno_list[anno])
        image = cv2.imread(os.path.join(
            image_path, anno_list[anno][:-3]+'jpg'))
        w = image.shape[1]
        h = image.shape[0]
        boxes_path = os.path.join(os.path.join(anno_path, anno_list[anno][:-3]+'csv'))
        boxes = []
        labels = []
        df = pd.read_csv(boxes_path)
        print('dude wut 1',np.array(boxes))

        if anno_list[anno][:-4] not in relabel:
            for index, row in df.iterrows():
                class_id = None
                if '10018' in anno_list[anno]:
                    class_id = 1
                elif '12221' in anno_list[anno]:
                    class_id = 2
                else:
                    class_id = 3
                labels.append(class_id)
                boxes.append([row['xmin'], row['ymin'],  row['xmax'], row['ymax']])
        print('dude wut',np.array(boxes))

        ndf = pd.read_csv(os.path.join(anno_new_path,anno_list[anno][:-3]+'txt'),sep=" ", header=None)
        print(np.array(boxes))
        for index, row in ndf.iterrows():
            labels.append(int(row[0])+1)
            print(convert_yolo_2_pascalVOC(w,h,float(row[1]),float(row[2]),float(row[3]),float(row[4])))
            boxes.append(convert_yolo_2_pascalVOC(w,h,float(row[1]),float(row[2]),float(row[3]),float(row[4])))
            print(np.array(boxes))
        

        boxes = np.array(boxes)
        data = {'label': labels, 'xmin': boxes[:, 0], 'ymin': boxes[:, 1],
                'xmax': boxes[:, 2], 'ymax': boxes[:, 3]}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join('atext', anno_list[anno][:-3] + 'csv'), index=False)

            # print(boxes)
        target = {}
        transformed = transform(image=image, bboxes=boxes, class_labels=labels)
        transformed_image = transformed['image']
        target["boxes"] = transformed['bboxes']
        plot_image_bboxes(transformed_image, target["boxes"], os.path.join(
            folder, anno_list[anno][:-3]+'jpg'))
       
        

anno_path = '../../new_annotation_2/anno_test'
image_path = '../../Lua/JPGimages/test'
anno_new_path = '../../newlabelv2'
visualize(anno_path, image_path, 'test', anno_new_path)
# if __name__=='__main__':
#anno_path = '../anno_train'
#image_path = '../Lua/JPGimages/train'
#visualize(anno_path, image_path, 'train')

#anno_path = '../anno_val'
#image_path = '../Lua/JPGimages/valid'
#visualize(anno_path, image_path, 'val')
