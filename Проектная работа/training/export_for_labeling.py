"""
Отбор кадров камеры конвейера для разметки в Roboflow с предразметкой моделью.

Берёт записанные партии (recordings/<запись>/frames.csv + frames/*.jpg), отбирает кадры:
треть — кучи (от 8 деталей), треть — «подозрительные» кадры, где рамки касаются или
перекрываются, остальное — прочие кадры с деталями; плюс немного пустых кадров.
Каждый кадр размечается текущей моделью (полигоны) и сохраняется в формате YOLOv8,
который Roboflow принимает при загрузке папки. Предразметку затем правят вручную.

Запуск (из корня проекта):
    python training/export_for_labeling.py --exclude 2026-10-03_21-4 2026-10-03_21-5
    python training/export_for_labeling.py --only 2026-10-04_13 --frames 200 --empty 0
Результат: training/to_label/<дата>[_<tag>]/ (images/, labels/, data.yaml, manifest.json)
"""
import argparse
import csv
import datetime
import json
import os
import random

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_image(path):
    """Чтение кадра (cv2.imread не открывает пути с кириллицей в Windows)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def list_recordings(base_dir):
    """Записанные партии: папки с frames.csv, новые первыми."""
    if not os.path.isdir(base_dir):
        return []
    return [os.path.join(base_dir, n) for n in sorted(os.listdir(base_dir), reverse=True)
            if os.path.isfile(os.path.join(base_dir, n, "frames.csv"))]


def recording_frames(path):
    """Список (время, путь к кадру) одной записи."""
    with open(os.path.join(path, "frames.csv"), encoding="utf-8") as f:
        return [(float(r["t"]), os.path.join(path, "frames", r["file"])) for r in csv.DictReader(f)]


def count_parts(model, path, belt, conf=0.4, imgsz=640):
    """Число деталей на ленте и признак «подозрительного» кадра.

    Рамки с центром вне полосы ленты belt=(x_min, x_max) (доли ширины кадра) не считаются:
    модель находит ложные «детали» на краях кадра. Кадр подозрительный, если какие-то
    рамки касаются или перекрываются — на таких кадрах чаще всего ошибается трекер.
    """
    img = read_image(path)
    if img is None:
        return -1, False
    res = model.predict(img, imgsz=imgsz, conf=0.25, iou=0.7, verbose=False)[0]
    if not len(res.boxes):
        return 0, False
    xywhn = res.boxes.xywhn.cpu().numpy()
    conf_ = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    on_belt = (xywhn[:, 0] >= belt[0]) & (xywhn[:, 0] <= belt[1])
    n = int(((conf_ >= conf) & on_belt).sum())
    b = xyxy[on_belt]
    suspicious = False
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            iw = min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0])
            ih = min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1])
            if iw > -3 and ih > -3:
                suspicious = True
    return n, suspicious


def already_exported(root):
    """Кадры из прошлых выгрузок (по их manifest.json), чтобы не размечать повторно."""
    done = set()
    base = os.path.join(root, "training", "to_label")
    for d in os.listdir(base) if os.path.isdir(base) else []:
        m = os.path.join(base, d, "manifest.json")
        if os.path.exists(m):
            with open(m, encoding="utf-8") as fh:
                done |= {x["image"] for x in json.load(fh)}
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", default=os.path.join(ROOT, "recordings"))
    ap.add_argument("--frames", type=int, default=250, help="сколько кадров с деталями выгрузить")
    ap.add_argument("--empty", type=int, default=20, help="сколько пустых кадров добавить")
    ap.add_argument("--scan-step", type=int, default=2, help="просматривать каждый N-й кадр записи")
    ap.add_argument("--step", type=int, default=2, help="из кадров с деталями брать каждый N-й")
    ap.add_argument("--model", default=os.path.join(ROOT, "training", "runs", "detector",
                                                     "v8_mixed_320", "weights", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=320, help="вход модели")
    ap.add_argument("--conf", type=float, default=0.25, help="порог уверенности для предразметки")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[], help="не брать записи с такими префиксами")
    ap.add_argument("--only", nargs="*", default=[], help="брать только записи с такими префиксами")
    ap.add_argument("--simplify", type=float, default=1.0, help="упрощение контура, px (0 — не упрощать)")
    ap.add_argument("--tag", default="", help="добавка к имени папки выгрузки")
    ap.add_argument("--belt", type=float, nargs=2, default=(0.10, 0.78),
                    help="полоса ленты по x, доли ширины кадра")
    ap.add_argument("--allow-repeat", action="store_true", help="не пропускать кадры прошлых выгрузок")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    rnd = random.Random(a.seed)

    recs = []
    for p in list_recordings(a.recordings):
        name = os.path.basename(p)
        if any(name.startswith(x) for x in a.exclude):
            continue
        if a.only and not any(name.startswith(x) for x in a.only):
            continue
        recs.append((name, recording_frames(p)))

    skip = set() if a.allow_repeat else already_exported(ROOT)
    busy, empty = [], []
    total = sum(len(frames[::a.scan_step]) for _, frames in recs)
    done = 0
    for rec_name, frames in recs:
        hot = []
        for t, f in frames[::a.scan_step]:
            if f"openmv_{rec_name}_{os.path.splitext(os.path.basename(f))[0]}.jpg" in skip:
                continue
            n, susp = count_parts(model, f, a.belt, imgsz=a.imgsz)
            done += 1
            print(f"\rотбор кадров {done}/{total}", end="", flush=True)
            if n > 0:
                hot.append((n, rec_name, f, susp))
            elif n == 0:
                empty.append((0, rec_name, f, False))
        busy += hot[::a.step]
    print()

    dense = [x for x in busy if x[0] >= 8]
    susp = [x for x in busy if x[0] < 8 and x[3]]
    sparse = [x for x in busy if x[0] < 8 and not x[3]]
    rnd.shuffle(dense); rnd.shuffle(susp); rnd.shuffle(sparse); rnd.shuffle(empty)
    n_dense = min(len(dense), a.frames // 3)
    n_susp = min(len(susp), a.frames - n_dense - a.frames // 3)
    chosen = (dense[:n_dense] + susp[:n_susp] + sparse[:a.frames - n_dense - n_susp]
              + empty[:a.empty])
    print(f"кандидатов: кучи {len(dense)}, подозрительные {len(susp)}, прочие {len(sparse)}")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + (f"_{a.tag}" if a.tag else "")
    out = os.path.join(ROOT, "training", "to_label", stamp)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    manifest = []
    for k, (n, rec_name, f, _) in enumerate(chosen, 1):
        img = read_image(f)
        if img is None:
            continue
        name = f"openmv_{rec_name}_{os.path.splitext(os.path.basename(f))[0]}"
        cv2.imencode(".jpg", img)[1].tofile(os.path.join(out, "images", name + ".jpg"))
        res = model.predict(img, imgsz=a.imgsz, conf=a.conf, iou=0.5, verbose=False)[0]
        lines = []
        h, w = img.shape[:2]
        if res.masks is not None:
            for poly in res.masks.xy:
                if len(poly) < 3:
                    continue
                cx = poly[:, 0].mean() / w
                if cx < a.belt[0] - 0.04 or cx > a.belt[1] + 0.02:
                    continue
                if a.simplify > 0:
                    poly = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2),
                                            a.simplify, True).reshape(-1, 2)
                    if len(poly) < 3:
                        continue
                lines.append("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly))
        with open(os.path.join(out, "labels", name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        manifest.append({"image": name + ".jpg", "recording": rec_name,
                         "seen_at_record": n, "draft_parts": len(lines)})
        print(f"\r{k}/{len(chosen)}", end="", flush=True)
    print()
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("train: images\nval: images\nnc: 1\nnames: ['part']\n")
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    n_empty = min(a.empty, len(empty))
    print(f"Готово: {len(manifest)} кадров из {len({m['recording'] for m in manifest})} записей → {out}")
    print(f"  кучи: {n_dense}, остальные с деталями: {len(manifest) - n_dense - n_empty}, пустые: {n_empty}")


if __name__ == "__main__":
    main()
