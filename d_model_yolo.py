import os
import glob
import shutil
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


images = glob.glob("dataset/images/*.png")
labels = glob.glob("dataset/labels/*.txt")

images.sort()
labels.sort()

train_im, test_im, train_lb, test_lb = train_test_split(images, labels, train_size=0.9, random_state=1)
train_im, val_im, train_lb,val_lb = train_test_split(train_im, train_lb, train_size=0.8, random_state=1)

past = ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']

for p in past:
    os.makedirs(os.path.join('dataset/sort', p), exist_ok=True)

def coprit(dst_dir, files):
    for file_path in files:
        filename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(dst_dir, filename))



coprit(os.path.join('dataset/sort/train/images'), train_im)
coprit(os.path.join('dataset/sort/train/labels'), train_lb)

coprit(os.path.join('dataset/sort/val/images'), val_im)
coprit(os.path.join('dataset/sort/val/labels'), val_lb)

coprit(os.path.join('dataset/sort/test/images'), test_im)
coprit(os.path.join('dataset/sort/test/labels'), test_lb)

data = {
    'path': os.path.abspath('dataset/sort'),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 13,
    'names': ['1','2','3','4','5','6','7','8','9','10','11','12','13']
}
with open('dataset/data_yaml.yaml', 'w', encoding='utf-8') as file:
    yaml.dump(data, file, sort_keys=False)

model = YOLO("yolo12m.pt")

fit = model.train(data='dataset/data_yaml.yaml', epochs=30, imgsz=640, batch=16)
pred = model.predict(source='dataset/sort/test/images', save=True)

model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data='dataset/data_yaml.yaml', split='test')
print(metrics.box.map)      
print(metrics.box.map50)    
print(metrics.box.mp)      
print(metrics.box.mr)       