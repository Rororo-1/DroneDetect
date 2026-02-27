import os
from ultralytics import YOLO
from pathlib import Path

while True:
    x = input('Введіть шлях до файлу:').strip('"')
    if x == 'stop':
        break
    else:

        model = YOLO('runs/detect/train/weights/best.pt')

        pred = model.predict(source=x, save=True, show=True)

        out = list(Path(pred[0].save_dir).iterdir())[0]
        print(out)
        os.startfile(out)