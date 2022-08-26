import pandas as pd
import numpy as np
import cv2
import os
import torch
import albumentations as A

ignore_list =  [3766,
3756,
3755,
3754,
3753,
3752,
3751,
3750,
3749,
3748,
3747,
3636,
3624,
3601,
3598,
3563,
3561,
3555,
3532,
3527,
3521,
3520,
3505,
3503,
3501,
3473,
3445,
3346,
3344,
3284,
3276,
3191,
3137,
3136,
3133,
3127,
3098,
3086,
3081,
3079,
3078,
3077,
3075,
3073]

'''
transforms = [
    A.Compose([
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.Rotate(45, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.Rotate(-45, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.Rotate(30, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.Rotate(-30, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.Rotate(15, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.Rotate(-15, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.HorizontalFlip(p=1),
        A.Rotate(45, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.HorizontalFlip(p=1),
        A.Rotate(-45, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.HorizontalFlip(p=1),
        A.Rotate(30, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.HorizontalFlip(p=1),
        A.Rotate(-30, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.HorizontalFlip(p=1),
        A.Rotate(15, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    A.Compose([
        A.HorizontalFlip(p=1),
        A.Rotate(-15, p=1),
        A.SmallestMaxSize(max_size=1024),
        A.LongestMaxSize(max_size=1280),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
]
'''
def get_transform():
    transforms = []
    transform_names = []
    rotate_angle = [45,-45,30,-30,15,-15]
    shearing_angle = [15,-15,30,-30,20,-20]
    transform_type = ['ori','r','hr','sh', 'hsh']
    angle = ['']

    for i in transform_type:
        if i == 'ori':
            transforms.append(A.Compose([
                A.SmallestMaxSize(max_size=1024),
                A.LongestMaxSize(max_size=1280),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])))
            transform_names.append(i)
        elif i == 'r':
            angle = rotate_angle
            for a in angle:
                transforms.append(A.Compose([
                    A.Rotate(a, p=1),
                    A.SmallestMaxSize(max_size=1024),
                    A.LongestMaxSize(max_size=1280),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])))
                transform_names.append(i + str(a))
        elif i == 'hr':
            angle = rotate_angle
            for a in angle:
                transforms.append(A.Compose([
                    A.HorizontalFlip(p=1),
                    A.Rotate(a, p=1),
                    A.SmallestMaxSize(max_size=1024),
                    A.LongestMaxSize(max_size=1280),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])))
                transform_names.append(i + str(a))
        elif i == 'sh':
            angle = shearing_angle
            for a in angle:
                transforms.append(A.Compose([
                    A.Affine(shear=a, p=1),
                    A.SmallestMaxSize(max_size=1024),
                    A.LongestMaxSize(max_size=1280),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])))
                transform_names.append(i + str(a))
        elif i == 'hsh':
            angle = shearing_angle
            for a in angle:
                transforms.append(A.Compose([
                    A.HorizontalFlip(p=1),
                    A.Affine(shear=a, p=1),
                    A.SmallestMaxSize(max_size=1024),
                    A.LongestMaxSize(max_size=1280),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])))
                transform_names.append(i + str(a))
    return transforms, transform_names


transforms, transform_names = get_transform()
'''['ori', 'r45', 'r-45', 'r30', 'r-30', 'r15', 'r-15',
                'hr45', 'hr-45', 'hr30', 'hr-30', 'hr15', 'hr-15', ]
'''
anno_dir = '../new_annotation/anno_train'
image_dir = '../../Lua/JPGimages/train'

anno_list = sorted(os.listdir(anno_dir))

for idx in range(len(anno_list)):
    check = False
    image_id = anno_list[idx][:-4]
    for ignore_name in ignore_list:
        if str(ignore_name) in image_id:
            check = True
            break
    if check:
        continue
    records = pd.read_csv(os.path.join(anno_dir, anno_list[idx]))

    image = cv2.imread(f'{image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
    boxes = np.array(
        records[['xmin', 'ymin', 'xmax', 'ymax']].values)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    area = np.array(area)

    class_id = None
    if '10018' in image_id:
        class_id = 1.0
    elif '12221' in image_id:
        class_id = 2.0
    else:
        class_id = 3.0
    class_ids = [class_id]*(len(records))

    iscrowd = np.zeros((records.shape[0],), dtype=np.int64)
    target = {}
    target['boxes'] = boxes
    target['labels'] = class_ids
    target['area'] = area
    target['image_id'] = torch.tensor([idx])
    target['iscrowd'] = iscrowd
    ti = 0
    for transform in transforms:
        labels = []
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        transformed = transform(
            image=image, bboxes=boxes, class_labels=class_ids)

        timage = transformed['image']
    
        for i in range(len(transformed["class_labels"])):
            xmins.append(transformed["bboxes"][i][0])
            ymins.append(transformed["bboxes"][i][1])
            xmaxs.append(transformed["bboxes"][i][2])
            ymaxs.append(transformed["bboxes"][i][3])
        labels = transformed["class_labels"]
        dist = {'label': labels, 'xmin': xmins,
                'ymin': ymins, 'xmax': xmaxs, 'ymax': ymaxs}
        df = pd.DataFrame(dist)
        df.to_csv("new_augmentation/anno/aug_" +
                  transform_names[ti] + "_" + image_id + ".csv", index=False)
        cv2.imwrite(
            "new_augmentation/imgs/aug_" + transform_names[ti] + "_" + image_id + ".jpg", timage)

        ti = ti+1
