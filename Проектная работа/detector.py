"""
Детектор деталей на основе YOLOv8n-seg.

Бэкенды:
  - NCNN (папка *_ncnn_model/) — основной, для Raspberry Pi 5;
  - ONNX Runtime (.onnx);
  - Ultralytics (.pt) — для ПК.

detect() возвращает два списка:
  detections      — уверенные детали (conf >= confidence_threshold);
  low_detections  — слабые срабатывания (low_confidence_threshold <= conf < confidence_threshold),
                    которые трекер использует только для продления уже существующих треков.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[int, int]
    track_id: int = None
    polygon: Optional[np.ndarray] = None  # контур маски сегментации (N, 2)


class ObjectDetector:
    def __init__(self, model_path='best.onnx'):
        self.model = None
        self.ort_session = None
        self.ncnn_net = None
        self.model_input_h = 640
        self.model_input_w = 640
        self.class_names = ['part']

        self.confidence_threshold = 0.5
        self.low_confidence_threshold = 0.1
        self.iou_threshold = 0.5
        self.detection_zone = None

        self.input_size = 640
        self.use_half = False
        self.device = 'cpu'

        self.use_clahe = False
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.auto_brightness = False
        self.zone_overlap_threshold = 0.5
        self.use_mask = True

        self._nc = 1    # число классов
        self._nm = 32   # число коэффициентов маски YOLOv8-seg
        self._ncnn_input_name = 'in0'
        self._ncnn_output_names = ('out0', 'out1')

        self.load_model(model_path)

    # ── Загрузка модели ──────────────────────────────────────────────────

    def load_model(self, model_path):
        try:
            if model_path.endswith('.onnx'):
                self._load_onnx(model_path)
            elif os.path.isdir(model_path) or 'ncnn' in model_path:
                self._load_ncnn(model_path)
            else:
                self._load_ultralytics(model_path)
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def _init_buffers(self):
        self._letterbox_buf = np.full(
            (self.model_input_h, self.model_input_w, 3), 114, dtype=np.uint8)
        self._blob_buf = np.empty(
            (1, 3, self.model_input_h, self.model_input_w), dtype=np.float32)
        self._rgb_buf = np.empty(
            (self.model_input_h, self.model_input_w, 3), dtype=np.uint8)

    def _load_onnx(self, model_path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(
            model_path, opts, providers=['CPUExecutionProvider'])

        inp = self.ort_session.get_inputs()[0]
        self.model_input_h = inp.shape[2]
        self.model_input_w = inp.shape[3]

        out0 = self.ort_session.get_outputs()[0]
        out1 = self.ort_session.get_outputs()[1]
        self._nm = out1.shape[1]
        self._nc = out0.shape[1] - 4 - self._nm
        self.class_names = ['part'] if self._nc == 1 else [f'class{i}' for i in range(self._nc)]

        self._init_buffers()
        print(f"ONNX: {model_path}, вход {self.model_input_w}x{self.model_input_h}")

    def _load_ncnn(self, model_dir):
        import ncnn

        meta_path = os.path.join(model_dir, 'metadata.yaml')
        if os.path.exists(meta_path):
            import yaml
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            imgsz = meta.get('imgsz', [640, 640])
            if isinstance(imgsz, list):
                self.model_input_h, self.model_input_w = imgsz[0], imgsz[1]
            else:
                self.model_input_h = self.model_input_w = int(imgsz)
            names = meta.get('names', {0: 'part'})
            self.class_names = [names[i] for i in range(len(names))]
            self._nc = len(self.class_names)
        else:
            self.model_input_h = self.model_input_w = 640
            self._nc = 1
            self.class_names = ['part']
        self._nm = 32

        param_path = os.path.join(model_dir, 'model.ncnn.param')
        bin_path = os.path.join(model_dir, 'model.ncnn.bin')
        self._ncnn_input_name, self._ncnn_output_names = self._parse_ncnn_io(param_path)

        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = 4
        # FP16 на ARM Cortex-A76 (Raspberry Pi 5) примерно вдвое быстрее FP32
        net.opt.use_fp16_packed = True
        net.opt.use_fp16_storage = True
        net.opt.use_fp16_arithmetic = True
        net.load_param(param_path)
        net.load_model(bin_path)
        self.ncnn_net = net
        self.ort_session = None
        self.model = None

        self._init_buffers()
        print(f"NCNN: {model_dir}, вход {self.model_input_w}x{self.model_input_h}")

    def _parse_ncnn_io(self, param_path):
        """Имена входного и двух выходных блобов из .param файла NCNN."""
        input_name = 'in0'
        output_names = ('out0', 'out1')
        try:
            with open(param_path) as f:
                lines = f.readlines()[2:]

            produced = {}
            consumed = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # формат строки: Тип Имя число_входов число_выходов [входы...] [выходы...]
                try:
                    n_in = int(parts[2])
                    n_out = int(parts[3])
                except ValueError:
                    continue
                inputs = parts[4:4 + n_in]
                outputs = parts[4 + n_in:4 + n_in + n_out]
                consumed.update(inputs)
                for b in outputs:
                    produced[b] = parts[1]

            # выходы сети — блобы, которые нигде не используются как вход
            terminal = [b for b in produced if b not in consumed]

            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == 'Input':
                    n_in = int(parts[2])
                    n_out = int(parts[3])
                    outs = parts[4 + n_in:4 + n_in + n_out]
                    if outs:
                        input_name = outs[0]
                    break

            if len(terminal) >= 2:
                output_names = (terminal[0], terminal[1])
            elif len(terminal) == 1:
                output_names = (terminal[0], output_names[1])
        except Exception as e:
            print(f"Не удалось разобрать {param_path}: {e}")

        return input_name, output_names

    def _load_ultralytics(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task='segment')
        self.ort_session = None
        self.ncnn_net = None
        self._nc = len(self.model.names)
        self._nm = 32
        print(f"Ultralytics: {model_path}")

    # ── Настройки и зона детекции ────────────────────────────────────────

    def set_detection_zone(self, zone):
        self.detection_zone = zone

    def set_optimization_params(self, input_size=640, use_half=False):
        self.input_size = input_size
        self.use_half = use_half

    def preprocess_frame(self, frame):
        """Необязательная коррекция яркости и CLAHE (в рабочей конфигурации выключены)."""
        if frame is None:
            return None
        processed = frame.copy()
        if self.auto_brightness:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean_brightness = np.mean(l)
            if abs(mean_brightness - 127) > 15:
                adjustment = (127 - mean_brightness) * 0.5
                l = np.clip(l.astype(np.float32) + adjustment, 0, 255).astype(np.uint8)
            if self.use_clahe:
                l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif self.use_clahe:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return processed

    def calculate_bbox_intersection(self, bbox, zone):
        """Доля площади рамки внутри зоны."""
        bx1, by1, bx2, by2 = bbox
        zx1, zy1, zx2, zy2 = zone
        ix1 = max(bx1, zx1); iy1 = max(by1, zy1)
        ix2 = min(bx2, zx2); iy2 = min(by2, zy2)
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1) / max((bx2 - bx1) * (by2 - by1), 1)

    def is_in_zone(self, center=None, bbox=None, zone=None):
        if zone is None:
            zone = self.detection_zone
        if zone is None:
            return True
        if bbox is not None:
            return self.calculate_bbox_intersection(bbox, zone) >= self.zone_overlap_threshold
        if center is not None:
            x, y = center
            x1, y1, x2, y2 = zone
            return x1 <= x <= x2 and y1 <= y <= y2
        return True

    # ── Подавление перекрывающихся рамок (NMS) ───────────────────────────

    def _nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= self.iou_threshold]
        return keep

    # ── Подготовка входа ─────────────────────────────────────────────────

    def _letterbox(self, frame):
        """Масштабирование кадра под вход модели без искажения пропорций, поля — серые (114)."""
        orig_h, orig_w = frame.shape[:2]
        ih, iw = self.model_input_h, self.model_input_w
        scale = min(iw / orig_w, ih / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_w = (iw - new_w) // 2
        pad_h = (ih - new_h) // 2
        resized = cv2.resize(frame, (new_w, new_h))
        buf = self._letterbox_buf
        buf[:] = 114
        buf[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        return scale, pad_w, pad_h, orig_w, orig_h

    # ── Разбор выхода модели ─────────────────────────────────────────────

    def _postprocess(self, pred, protos, scale, pad_w, pad_h, orig_w, orig_h):
        """
        pred:   [кандидаты, 4 + nc + nm] — рамка (cx, cy, w, h), уверенности классов, коэффициенты маски
        protos: [nm, proto_h, proto_w]   — прототипы масок
        """
        nc, nm = self._nc, self._nm
        proto_h, proto_w = protos.shape[1], protos.shape[2]

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:4 + nc]
        mask_coefs = pred[:, 4 + nc:4 + nc + nm]

        conf = class_scores.max(axis=1)
        cls_ids = class_scores.argmax(axis=1)

        mask = conf >= self.low_confidence_threshold
        if not mask.any():
            return [], []

        boxes_xywh = boxes_xywh[mask]
        conf = conf[mask]
        cls_ids = cls_ids[mask]
        mask_coefs = mask_coefs[mask]

        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = self._nms(boxes_xyxy, conf)
        boxes_xyxy = boxes_xyxy[keep]
        conf = conf[keep]
        cls_ids = cls_ids[keep]
        mask_coefs = mask_coefs[keep]

        iw, ih = self.model_input_w, self.model_input_h
        detections = []
        low_detections = []
        for i in range(len(boxes_xyxy)):
            # координаты обратно в систему исходного кадра
            bx1 = int(np.clip((boxes_xyxy[i, 0] - pad_w) / scale, 0, orig_w))
            by1 = int(np.clip((boxes_xyxy[i, 1] - pad_h) / scale, 0, orig_h))
            bx2 = int(np.clip((boxes_xyxy[i, 2] - pad_w) / scale, 0, orig_w))
            by2 = int(np.clip((boxes_xyxy[i, 3] - pad_h) / scale, 0, orig_h))

            bbox = (bx1, by1, bx2, by2)
            center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

            if not self.is_in_zone(bbox=bbox):
                continue

            polygon = None
            if self.use_mask:
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)
                mask_map = (mask_coefs[i] @ protos.reshape(nm, -1)).reshape(proto_h, proto_w)
                mask_map = 1.0 / (1.0 + np.exp(-mask_map))
                mask_map = cv2.resize(mask_map, (iw, ih))
                mask_map = mask_map[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
                mask_map = cv2.resize(mask_map, (orig_w, orig_h))
                # маска обрезается по рамке, как в Ultralytics, иначе контур может уйти на фон
                binary = np.zeros((orig_h, orig_w), np.uint8)
                binary[by1:by2, bx1:bx2] = (mask_map[by1:by2, bx1:bx2] > 0.5) * 255
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    pts = largest.reshape(-1, 2)
                    if len(pts) >= 3:
                        polygon = pts

            det = Detection(
                bbox=bbox,
                confidence=float(conf[i]),
                class_id=int(cls_ids[i]),
                class_name=self.class_names[int(cls_ids[i])],
                center=center,
                polygon=polygon
            )
            if float(conf[i]) >= self.confidence_threshold:
                detections.append(det)
            else:
                low_detections.append(det)

        return detections, low_detections

    # ── Инференс ─────────────────────────────────────────────────────────

    def _detect_onnx(self, frame):
        scale, pad_w, pad_h, orig_w, orig_h = self._letterbox(frame)

        buf = self._letterbox_buf
        self._blob_buf[0, 0] = buf[:, :, 2]
        self._blob_buf[0, 1] = buf[:, :, 1]
        self._blob_buf[0, 2] = buf[:, :, 0]
        self._blob_buf *= (1.0 / 255.0)

        outputs = self.ort_session.run(
            None, {self.ort_session.get_inputs()[0].name: self._blob_buf})

        pred = outputs[0][0].T
        protos = outputs[1][0]
        return self._postprocess(pred, protos, scale, pad_w, pad_h, orig_w, orig_h)

    def _detect_ncnn(self, frame):
        import ncnn

        scale, pad_w, pad_h, orig_w, orig_h = self._letterbox(frame)

        # BGR → RGB, нормализация 0..1
        self._rgb_buf[:, :, 0] = self._letterbox_buf[:, :, 2]
        self._rgb_buf[:, :, 1] = self._letterbox_buf[:, :, 1]
        self._rgb_buf[:, :, 2] = self._letterbox_buf[:, :, 0]
        PIXEL_RGB = 2
        mat_in = ncnn.Mat.from_pixels(self._rgb_buf, PIXEL_RGB,
                                       self.model_input_w, self.model_input_h)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])

        ex = self.ncnn_net.create_extractor()
        ex.input(self._ncnn_input_name, mat_in)
        ret0, mat_out0 = ex.extract(self._ncnn_output_names[0])
        ret1, mat_out1 = ex.extract(self._ncnn_output_names[1])
        if ret0 != 0 or ret1 != 0:
            print(f"NCNN: ошибка извлечения выходов (ret={ret0},{ret1})")
            return [], []

        arr0 = np.array(mat_out0)
        arr1 = np.array(mat_out1)
        n_pred = 4 + self._nc + self._nm

        # выход с прототипами масок больше по размеру, чем выход с предсказаниями
        if arr0.size > arr1.size:
            proto_arr, pred_arr = arr0, arr1
        else:
            pred_arr, proto_arr = arr0, arr1
        pred = pred_arr.reshape(n_pred, -1).T

        # прототипы масок — 1/4 входа; вход может быть прямоугольным (320×256)
        nm = self._nm
        proto_h, proto_w = self.model_input_h // 4, self.model_input_w // 4
        if proto_h * proto_w * nm != proto_arr.size:
            proto_h = proto_w = int(round((proto_arr.size / nm) ** 0.5))
        protos = proto_arr.reshape(nm, proto_h, proto_w)

        return self._postprocess(pred, protos, scale, pad_w, pad_h, orig_w, orig_h)

    def detect(self, frame, use_optimization=True, use_preprocessing=True):
        if frame is None:
            return [], []

        if use_preprocessing and (self.auto_brightness or self.use_clahe):
            frame = self.preprocess_frame(frame)

        if self.ncnn_net is not None:
            return self._detect_ncnn(frame)
        if self.ort_session is not None:
            return self._detect_onnx(frame)
        if self.model is None:
            return [], []

        imgsz = self.input_size if use_optimization else max(frame.shape[:2])
        results = self.model(
            frame,
            conf=self.low_confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            imgsz=imgsz,
            half=self.use_half,
            device=self.device
        )

        detections = []
        low_detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            masks = result.masks.xy if result.masks is not None else None

            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2, y2)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                polygon = None
                if masks is not None and i < len(masks):
                    pts = masks[i].astype(np.int32)
                    if len(pts) >= 3:
                        polygon = pts

                if self.is_in_zone(bbox=bbox):
                    det = Detection(
                        bbox=bbox,
                        confidence=float(conf),
                        class_id=int(cls),
                        class_name=self.model.names[int(cls)],
                        center=center,
                        polygon=polygon
                    )
                    if float(conf) >= self.confidence_threshold:
                        detections.append(det)
                    else:
                        low_detections.append(det)

        return detections, low_detections

    # ── Отрисовка ────────────────────────────────────────────────────────

    def draw_detections(self, frame, detections, draw_zone=True, counting_line_y=0):
        frame_copy = frame.copy()

        if draw_zone and self.detection_zone is not None:
            x1, y1, x2, y2 = self.detection_zone
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_copy, "Detection Zone", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if counting_line_y > 0:
            h, w = frame_copy.shape[:2]
            cv2.line(frame_copy, (0, counting_line_y), (w, counting_line_y), (0, 255, 255), 2)
            cv2.putText(frame_copy, "Count Line", (4, counting_line_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        polygons = [det.polygon for det in detections if det.polygon is not None]
        if polygons:
            overlay = frame_copy.copy()
            for poly in polygons:
                cv2.fillPoly(overlay, [poly], (255, 0, 0))
            cv2.addWeighted(overlay, 0.15, frame_copy, 0.85, 0, frame_copy)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (255, 0, 0)
            if det.polygon is not None:
                cv2.polylines(frame_copy, [det.polygon], isClosed=True, color=color, thickness=2)
            else:
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name}: {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"ID:{det.track_id} {label}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_copy, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame_copy, det.center, 3, (0, 255, 0), -1)

        return frame_copy
