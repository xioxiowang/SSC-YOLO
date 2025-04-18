import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__': 
    # model = YOLO('ultralytics/cfg/models/my-experments/rdd2022-to-do/yolov8.yaml') ##exp2
    model = YOLO('ultralytics/cfg/models/my-experments/todo/yolov8-MFConv.yaml')   ##exp
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='F:/yolov8/ultralytics-main/datasets/myrdd2022.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=32,
                close_mosaic=0,
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0   
                # device='0',
                optimizer='SGD', # using SGD  
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='yolov8-MFConv2',     
                )
