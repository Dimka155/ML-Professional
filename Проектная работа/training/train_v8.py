"""
Дообучение детектора на датасете с кадрами камеры конвейера и экспорт в NCNN.

Разделение данных — по именам файлов:
  тест       — кадры серии C (в обучении не участвуют);
  валидация  — valid/ и test/ из Roboflow, кроме теста;
  обучение   — train/ из Roboflow, кроме теста.

Варианты:
  mixed   — старый датасет + кадры камеры (кадры камеры повторяются OVERSAMPLE раз);
  camera  — только кадры камеры.

Запуск (из папки training):
    python train_v8.py mixed          # вход 640, дообучение от базовой модели
    python train_v8.py mixed 320      # вход 320 для Raspberry Pi, дообучение от v8_mixed
Результат: runs/detector/v8_<вариант>[_320]/weights/best.pt и best_ncnn_model/
"""
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / 'dataset_v8'
BASE_MODEL = HERE / 'runs/detector/part_detector_nano/weights/best.pt'
TEST_PREFIXES = ('openmv_2026-10-03_21-4', 'openmv_2026-10-03_21-5')
OVERSAMPLE = 3
EPOCHS = 60
PATIENCE = 20
BATCH = 16
IMG_SIZE = 640
WORKERS = 4


def _kind(name):
    if name.startswith(TEST_PREFIXES):
        return 'test'
    return 'camera' if name.startswith('openmv_') else 'old'


def build_split(variant):
    """Собирает списки файлов train/val/test и data.yaml в splits/v8_<variant>/."""
    out = HERE / 'splits' / f'v8_{variant}'
    if out.exists():
        shutil.rmtree(out)
    lists = {'train': [], 'val': [], 'test': []}
    for split in ('train', 'valid', 'test'):
        for img in sorted((DATASET / split / 'images').iterdir()):
            kind = _kind(img.name)
            if kind == 'test':
                lists['test'].append(img)
            elif variant == 'camera' and kind != 'camera':
                continue
            else:
                lists['train' if split == 'train' else 'val'].append(img)

    if variant == 'mixed' and OVERSAMPLE > 1:
        # ultralytics убирает повторяющиеся пути, поэтому повторы — копиями файлов
        rep_img = out / 'repeat' / 'images'
        rep_lbl = out / 'repeat' / 'labels'
        rep_img.mkdir(parents=True)
        rep_lbl.mkdir(parents=True)
        extra = []
        for img in lists['train']:
            if _kind(img.name) != 'camera':
                continue
            lbl = img.parent.parent / 'labels' / (img.stem + '.txt')
            for k in range(1, OVERSAMPLE):
                dst = rep_img / f'{img.stem}_rep{k}{img.suffix}'
                shutil.copyfile(img, dst)
                shutil.copyfile(lbl, rep_lbl / f'{img.stem}_rep{k}.txt')
                extra.append(dst)
        lists['train'] += extra

    out.mkdir(parents=True, exist_ok=True)
    for name, files in lists.items():
        (out / f'{name}.txt').write_text('\n'.join(str(f.resolve()) for f in files) + '\n', encoding='utf-8')
    yaml = out / 'data.yaml'
    yaml.write_text(
        f"train: {(out / 'train.txt').resolve().as_posix()}\n"
        f"val: {(out / 'val.txt').resolve().as_posix()}\n"
        f"test: {(out / 'test.txt').resolve().as_posix()}\n"
        "nc: 1\nnames: ['part']\n", encoding='utf-8')
    counts = {k: len(v) for k, v in lists.items()}
    cam = sum(_kind(f.name) == 'camera' for f in lists['train'])
    print(f"разделение {variant}: {counts}, кадров камеры в обучении: {cam}")
    return yaml


def train(variant, imgsz=IMG_SIZE):
    from ultralytics import YOLO
    data = build_split(variant)
    name = f'v8_{variant}' + ('' if imgsz == IMG_SIZE else f'_{imgsz}')
    base = BASE_MODEL if imgsz == IMG_SIZE else HERE / f'runs/detector/v8_{variant}/weights/best.pt'
    model = YOLO(str(base))
    model.train(
        data=str(data),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=imgsz,
        device='0',
        workers=WORKERS,
        patience=PATIENCE,
        hsv_h=0.015, hsv_s=0.3, hsv_v=0.4,
        degrees=180.0, translate=0.2, scale=0.3,
        flipud=0.5, fliplr=0.5, mosaic=0.5,
        project=str(HERE / 'runs/detector'),
        name=name,
        exist_ok=True,
        plots=False,
    )
    best = HERE / f'runs/detector/{name}/weights/best.pt'
    print(f"Модель: {best}\nNCNN: {export_ncnn(best, imgsz)}")


def export_ncnn(best, imgsz=None):
    """Экспорт в NCNN для Raspberry Pi.

    Для входа 320 экспортируется прямоугольный вход 320×256 под кадр камеры 320×240.
    torch.jit не сохраняет файлы по пути с кириллицей, поэтому экспорт идёт
    во временной папке, а результат копируется рядом с весами.
    """
    from ultralytics import YOLO
    if imgsz == 320:
        imgsz = (256, 320)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'best.pt'
        shutil.copyfile(best, src)
        YOLO(str(src)).export(format='ncnn', imgsz=imgsz or IMG_SIZE, half=False)
        dst = best.parent / 'best_ncnn_model'
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(Path(tmp) / 'best_ncnn_model', dst, dirs_exist_ok=True)
    return dst


if __name__ == '__main__':
    variant = sys.argv[1] if len(sys.argv) > 1 else 'mixed'
    assert variant in ('mixed', 'camera'), variant
    train(variant, int(sys.argv[2]) if len(sys.argv) > 2 else IMG_SIZE)
