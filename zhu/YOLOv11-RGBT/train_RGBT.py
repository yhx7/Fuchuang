import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion.yaml')
    # model.info(True,True)
    model.load('yolo11n-obb.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/VEDAI512.yaml',
                cache=False,
                imgsz=640,
                epochs=30,
                batch=8,
                close_mosaic=10,
                workers=1,
                device='0',
                optimizer='SGD',  # using SGD
                # lr0=0.002,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                # pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
                use_simotm="RGBT",
                channels=4,
                project='runs/VEDAI512_test',
                name='VEDAI512-yolo11n-obb-RGBT-midfusion-',
                )