"""
Обучение базового детектора деталей YOLOv8n-seg (один класс: part).

Датасет: экспорт из Roboflow в формате YOLOv8 (описан в detector_data.yaml).
Результат: runs/detector/part_detector_nano/weights/best.pt

Запуск (из папки training):
    python detector_train.py
"""
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / 'detector_data.yaml'
BASE_MODEL = 'yolov8n-seg.pt'
EPOCHS = 100
BATCH = 16
IMG_SIZE = 640
PATIENCE = 30
DEVICE = '0'


def train():
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        patience=PATIENCE,
        # аугментации под конвейер
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.4,      # изменения освещения
        degrees=180.0,  # деталь лежит под любым углом
        translate=0.2,  # деталь не всегда по центру кадра
        scale=0.3,      # разные размеры деталей
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        project='runs/detector',
        name='part_detector_nano',
        exist_ok=True,
    )
    best = Path('runs/detector/part_detector_nano/weights/best.pt')
    print(f"Модель: {best}")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)')}")


if __name__ == '__main__':
    train()
