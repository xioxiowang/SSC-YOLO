import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train-voc/yolov8-MFConv2/weights/best.pt')
    model.val(data='F:/yolov8/ultralytics-main/datasets/voc.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val/voc2007',
              name='yolov8-MFConv2',
              )
