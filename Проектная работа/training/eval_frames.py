"""
Покадровая оценка детектора на размеченных тестовых кадрах камеры (без трекера).

Найденные рамки сопоставляются с разметкой (рамка вокруг полигона, IoU >= 0.5).
Считаются: найдено, пропущено, лишние срабатывания, полнота, точность
и число кадров, на которых количество деталей определено точно.
Кадры приводятся к 320×240 — так они приходят с камеры.

Запуск (из папки training):
    python eval_frames.py ../models/best.pt --imgsz 320 --iou 0.4
--list — папка с кадрами (рядом с ней labels/) или текстовый файл со списком путей.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAM_SIZE = (320, 240)


def read_image(path):
    """Чтение кадра (cv2.imread не открывает пути с кириллицей в Windows)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def gt_boxes(label_path, w, h):
    """Рамки из разметки YOLO: полигон → описанная рамка; рамка cx cy w h → x1 y1 x2 y2."""
    boxes = []
    if not os.path.exists(label_path):
        return np.zeros((0, 4))
    for line in open(label_path, encoding='utf-8'):
        v = line.split()
        if not v:
            continue
        xy = np.array(list(map(float, v[1:])))
        if len(xy) == 4:
            cx, cy, bw, bh = xy
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        else:
            p = xy.reshape(-1, 2) * [w, h]
            boxes.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
    return np.array(boxes).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt, pred, thr=0.5):
    """Жадное сопоставление по IoU → (верно найдено, пропущено, лишние)."""
    m = iou_matrix(gt, pred)
    tp = 0
    used_g, used_p = set(), set()
    for idx in np.argsort(-m, axis=None):
        g, p = np.unravel_index(idx, m.shape)
        if m[g, p] < thr:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p); tp += 1
    return tp, len(gt) - tp, len(pred) - tp


def evaluate(model_path, images, conf, iou, imgsz):
    from ultralytics import YOLO
    model = YOLO(model_path, task='segment')
    tp = fn = fp = exact = 0
    abs_err = []
    worst = []
    for img_path in images:
        img = read_image(str(img_path))
        img = cv2.resize(img, CAM_SIZE, interpolation=cv2.INTER_AREA)
        w, h = CAM_SIZE
        gt = gt_boxes(str(img_path.parent.parent / 'labels' / (img_path.stem + '.txt')), w, h)
        r = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        pred = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        a, b, c = match(gt, pred)
        tp += a; fn += b; fp += c
        exact += len(gt) == len(pred)
        abs_err.append(abs(len(gt) - len(pred)))
        worst.append((b + c, img_path.name, len(gt), len(pred), b, c))
    n_gt = tp + fn
    worst.sort(reverse=True)
    return {
        'кадров': len(images), 'деталей': n_gt,
        'найдено': tp, 'пропущено': fn, 'лишних': fp,
        'полнота': tp / max(1, n_gt), 'точность': tp / max(1, tp + fp),
        'кадров с точным числом': exact, 'ср. ошибка числа': float(np.mean(abs_err)),
        'худшие': worst[:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--list', default=str(HERE.parent / 'test_frames' / 'images'), help='тестовые кадры')
    ap.add_argument('--conf', type=float, default=0.4)
    ap.add_argument('--iou', type=float, default=0.7, help='порог NMS')
    ap.add_argument('--imgsz', type=int, default=640)
    a = ap.parse_args()
    src = Path(a.list)
    if src.is_dir():
        images = sorted(src.glob('*.jpg'))
    else:
        images = [Path(p) for p in src.read_text(encoding='utf-8').splitlines() if p.strip()]
    for m in a.models:
        r = evaluate(m, images, a.conf, a.iou, a.imgsz)
        print(f"\n### {m}  (conf {a.conf}, NMS {a.iou}, imgsz {a.imgsz})")
        print(f"кадров {r['кадров']}, деталей {r['деталей']}: найдено {r['найдено']}, "
              f"пропущено {r['пропущено']}, лишних {r['лишних']} | полнота {r['полнота']:.1%}, "
              f"точность {r['точность']:.1%} | точное число на {r['кадров с точным числом']} кадрах, "
              f"ср. ошибка {r['ср. ошибка числа']:.2f}")
        for w in r['худшие']:
            print(f"   {w[1][:60]}: размечено {w[2]}, найдено {w[3]} (пропущено {w[4]}, лишних {w[5]})")


if __name__ == '__main__':
    main()
