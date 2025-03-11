import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# visdrone2019
# classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
classes = ['D00', 'D10', 'D20', 'D30']

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='myrdd2022\\images\\val',type=str, help="path of images")
parser.add_argument('--label_path', default='myrdd2022\\labels\\val',type=str, help="path of labels .txt")
parser.add_argument('--save_path', default='myrdd2022\\annotations\\myrdd2022_val.json', type=str, help="if not split the dataset, give a path to a json file")
arg = parser.parse_args()

def yolo2coco(arg):
    print("Loading data from ", arg.image_path, arg.label_path)

    assert os.path.exists(arg.image_path)
    assert os.path.exists(arg.label_path)
    
    originImagesDir = arg.image_path                                   
    originLabelsDir = arg.label_path
    # images dir name
    indexes = os.listdir(originImagesDir)

    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片.
        txtFile = f'{index[:index.rfind(".")]}.txt'
        stem = index[:index.rfind(".")]
        # 读取图像的宽和高
        try:
            im = cv2.imread(os.path.join(originImagesDir, index))
            height, width, _ = im.shape
        except Exception as e:
            print(f'{os.path.join(originImagesDir, index)} read error.\nerror:{e}')
        # 添加图像的信息
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息.
            continue
        dataset['images'].append({'file_name': index,
                            'id': stem,
                            'width': width,
                            'height': height})
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(label[0])   
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': stem,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    os.makedirs(os.path.dirname(arg.save_path), exist_ok=True)
    with open(arg.save_path, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(arg.save_path))

if __name__ == "__main__":
    yolo2coco(arg)

# import os
# import cv2 
# import json
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# import argparse

# # visdrone2019
# # classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
# #         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# classes = ['D00', 'D10', 'D20', 'D30']



# parser = argparse.ArgumentParser()
# parser.add_argument('--image_path', default='myrdd2022\\images\\val', type=str, help="path of images")
# parser.add_argument('--label_path', default='myrdd2022\\labels\\val', type=str, help="path of labels .txt")
# parser.add_argument('--save_path', default='myrdd2022\\annotations\\myrdd2022_val.json', type=str, help="if not split the dataset, give a path to a json file")
# arg = parser.parse_args()

# def yolo2coco(arg):
#     print("Loading data from ", arg.image_path, arg.label_path)

#     assert os.path.exists(arg.image_path), f"Image path {arg.image_path} does not exist."
#     assert os.path.exists(arg.label_path), f"Label path {arg.label_path} does not exist."

#     originImagesDir = arg.image_path
#     originLabelsDir = arg.label_path
#     # images dir name
#     indexes = os.listdir(originImagesDir)

#     dataset = {'categories': [], 'annotations': [], 'images': []}
#     for i, cls in enumerate(classes, 1):  # 类别ID从1开始
#         dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

#     # 标注的id
#     ann_id_cnt = 0
#     for k, index in enumerate(tqdm(indexes)):
#         # 支持 png jpg 格式的图片.
#         txtFile = f'{index[:index.rfind(".")]}.txt'
#         stem = int(index[:index.rfind(".")])  # image_id 转换为整数
#         # 读取图像的宽和高
#         try:
#             im = cv2.imread(os.path.join(originImagesDir, index))
#             height, width, _ = im.shape
#         except Exception as e:
#             print(f'{os.path.join(originImagesDir, index)} read error.\nerror:{e}')
#             continue
        
#         # 添加图像的信息
#         if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
#             # 如没标签，跳过，只保留图片信息.
#             dataset['images'].append({'file_name': index, 'id': stem, 'width': width, 'height': height})
#             continue

#         dataset['images'].append({'file_name': index, 'id': stem, 'width': width, 'height': height})

#         with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
#             labelList = fr.readlines()
#             for label in labelList:
#                 label = label.strip().split()
#                 x = float(label[1])
#                 y = float(label[2])
#                 w = float(label[3])
#                 h = float(label[4])

#                 # convert x,y,w,h to x1,y1,x2,y2
#                 H, W, _ = im.shape
#                 x1 = (x - w / 2) * W
#                 y1 = (y - h / 2) * H
#                 x2 = (x + w / 2) * W
#                 y2 = (y + h / 2) * H
                
#                 # 确保边界框合法
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(W, x2), min(H, y2)
#                 width = max(0, x2 - x1)
#                 height = max(0, y2 - y1)

#                 cls_id = int(label[0]) + 1  # 类别ID从1开始
#                 dataset['annotations'].append({
#                     'area': width * height,
#                     'bbox': [x1, y1, width, height],
#                     'category_id': cls_id,
#                     'id': ann_id_cnt,
#                     'image_id': stem,
#                     'iscrowd': 0,
#                     'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
#                 })
#                 ann_id_cnt += 1

#     # 保存结果
#     os.makedirs(os.path.dirname(arg.save_path), exist_ok=True)
#     with open(arg.save_path, 'w') as f:
#         json.dump(dataset, f, indent=4)
#         print('Save annotation to {}'.format(arg.save_path))

# if __name__ == "__main__":
#     yolo2coco(arg)
