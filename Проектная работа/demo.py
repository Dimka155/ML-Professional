"""
Проверка детектора на тестовых кадрах камеры без Raspberry Pi и конвейера.

Каждый кадр приводится к 320×240 (как с камеры OpenMV), прогоняется через ObjectDetector
с рабочими настройками (conf 0.4, NMS 0.4), число найденных деталей сравнивается
с разметкой, кадры с контурами сохраняются в папку demo_out/.

Запуск (из этой папки):
    python demo.py                              # NCNN, если установлен пакет ncnn, иначе best.pt
    python demo.py --model models/best.pt       # через Ultralytics
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detector import ObjectDetector

HERE = Path(__file__).resolve().parent
CAM_SIZE = (320, 240)


def default_model():
    try:
        import ncnn  # noqa: F401
        return str(HERE / 'models' / 'best_ncnn_model')
    except ImportError:
        return str(HERE / 'models' / 'best.pt')


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=default_model())
    ap.add_argument('--images', default=str(HERE / 'test_frames' / 'images'))
    ap.add_argument('--out', default=str(HERE / 'demo_out'))
    a = ap.parse_args()

    det = ObjectDetector(a.model)
    det.confidence_threshold = 0.4
    det.low_confidence_threshold = 0.1
    det.iou_threshold = 0.4
    det.set_optimization_params(input_size=320)

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    images = sorted(Path(a.images).glob('*.jpg'))
    exact = abs_err = 0
    times = []
    for img_path in images:
        frame = cv2.resize(read_image(img_path), CAM_SIZE, interpolation=cv2.INTER_AREA)
        label = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        n_gt = sum(1 for s in label.read_text(encoding='utf-8').splitlines() if s.strip()) if label.exists() else 0

        t0 = time.perf_counter()
        detections, _ = det.detect(frame)
        times.append(time.perf_counter() - t0)

        n = len(detections)
        exact += n == n_gt
        abs_err += abs(n - n_gt)
        vis = det.draw_detections(frame, detections, draw_zone=False)
        cv2.putText(vis, f"found {n} / labeled {n_gt}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imencode('.jpg', vis)[1].tofile(str(out / img_path.name))

    print(f"Кадров: {len(images)}; число деталей определено точно на {exact} "
          f"({exact / max(1, len(images)):.0%}), средняя ошибка числа {abs_err / max(1, len(images)):.2f}")
    print(f"Среднее время кадра: {1000 * np.mean(times[1:] or times):.1f} мс (без первого кадра)")
    print(f"Кадры с контурами: {out}")


if __name__ == '__main__':
    main()
