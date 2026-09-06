#!/usr/bin/env python3
"""
plugins/face.py — FaceRecognitionPlugin: EdgeFace + (YuNet|SCRFD) face recognition.

Pipeline: CompressedImage → YuNet detect → align (112×112) → EdgeFace embed → identity match
Downloads weights from juicefs (http://172.28.4.81:34567/).
Outputs benchmark-compliant JSON (one highest-confidence face object):
  {
    "detect_confidence": 0.95,
    "bbox_relative": [x, y, w, h],  # normalized [0-1]
    "identity": {
      "person_id": "n000001",
      "confidence": 0.91
    }
  }

Supports multi-instance (one instance per input topic).
Follows VOP plugin architecture: CompressedImage subscription, frame queue,
background model loading, worker thread inference.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)

# ── EdgeFace source on path ──────────────────────────────────────────────────
_EDGEFACE_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "edgeface_src",
)
if _EDGEFACE_SRC not in sys.path:
    sys.path.insert(0, _EDGEFACE_SRC)

# ── Model URLs (juicefs) ─────────────────────────────────────────────────────
_MODEL_BASE_URL = os.environ.get(
    "FACE_MODEL_BASE_URL", "http://172.28.4.81:34567/face"
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_SIMILARITY_THRESHOLD = 0.5  # cosine similarity above this = same person
DEFAULT_MODEL_NAME = "edgeface_s_gamma_05"  # 3.65M params, ~14MB checkpoint (eval-proven: 0.9165)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

# ── Plugin tool schema ───────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "face",
        "type": "processor",
        "multiInstance": True,
        "description": "Face Recognition — detect and identify faces using EdgeFace + YuNet/SCRFD",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb, required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "EdgeFace model name",
                    "default": DEFAULT_MODEL_NAME,
                    "scope": "shared",
                },
                "detector": {
                    "type": "string",
                    "enum": ["yunet", "scrfd"],
                    "description": "Face detector: yunet (proven default) or scrfd-500m-kps",
                    "default": "yunet",
                    "scope": "shared",
                },
                "face_db_dir": {
                    "type": "string",
                    "description": "Path to identity library directory (face_db/n000001/xxx.jpg)",
                    "default": "/workspace/face_db",
                    "scope": "shared",
                },
                "model_dir": {
                    "type": "string",
                    "description": "Directory for downloaded model weights",
                    "default": "/models/face",
                    "scope": "shared",
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Cosine similarity threshold for identity match (0-1)",
                    "default": DEFAULT_SIMILARITY_THRESHOLD,
                    "scope": "instance",
                },
                "confidence": {
                    "type": "number",
                    "description": "YuNet detection confidence threshold (0-1)",
                    "default": 0.5,
                    "scope": "instance",
                },
                "fps": {
                    "type": "integer",
                    "description": "Max inference frames per second",
                    "default": 3,
                    "scope": "instance",
                },
                "device": {
                    "type": "string",
                    "enum": ["cuda", "cpu"],
                    "description": "Inference device",
                    "default": "cpu",
                    "scope": "shared",
                },
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "face recognition result"}],
    }
]


# ── Model download ────────────────────────────────────────────────────────────

def _ensure_weights(model_name: str, model_dir: str, detector: str = "yunet") -> str:
    """Download model weights from juicefs if not present. Returns checkpoint path."""
    os.makedirs(model_dir, exist_ok=True)

    # EdgeFace checkpoint
    ckpt_filename = f"{model_name}.pt"
    ckpt_path = os.path.join(model_dir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        url = f"{_MODEL_BASE_URL}/{ckpt_filename}"
        log.info(f"[face] downloading {ckpt_filename} from {url} → {ckpt_path}")
        urllib.request.urlretrieve(url, ckpt_path)
        log.info(f"[face] download complete: {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:.1f} MB)")

    # Detector ONNX model
    det_filename = {
        "yunet": "face_detection_yunet_2023mar.onnx",
        "scrfd": "scrfd_500m_kps.onnx",
    }.get(detector, "face_detection_yunet_2023mar.onnx")
    det_path = os.path.join(model_dir, det_filename)
    if not os.path.exists(det_path):
        url = f"{_MODEL_BASE_URL}/{det_filename}"
        log.info(f"[face] downloading {det_filename} from {url}")
        urllib.request.urlretrieve(url, det_path)
        log.info(f"[face] download complete: {det_path} ({os.path.getsize(det_path) / 1e6:.1f} MB)")

    return ckpt_path


# ── Face Database (Identity Library) ─────────────────────────────────────────

class FaceDatabase:
    """Identity library: loads face images, extracts embeddings, matches queries.

    Directory structure:
        face_db/
            n000001/
                0001_01.jpg
            n000002/
                0002_01.jpg
    """

    def __init__(self):
        self._embeddings: dict[str, list[np.ndarray]] = {}
        self._lock = threading.RLock()
        self.load_diag: Optional[str] = None

    def load_from_dir(self, db_dir: str, adapter: "EdgeFaceAdapter"):
        """Load identity library: detect + embed all faces in db_dir."""
        db_path = Path(db_dir)
        if not db_path.exists():
            log.warning(f"[face] face db dir not found: {db_dir}")
            return

        from PIL import Image

        with self._lock:
            self._embeddings.clear()
            n_dirs = n_images = n_fail = 0

            for person_dir in sorted(db_path.iterdir()):
                if not person_dir.is_dir():
                    continue
                n_dirs += 1
                person_id = person_dir.name
                for img_file in sorted(person_dir.iterdir()):
                    if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                        continue
                    n_images += 1
                    try:
                        pil_img = Image.open(img_file).convert("RGB")
                        img_arr = np.array(pil_img)
                        detections = adapter.detect_and_embed(img_arr)
                        if detections:
                            best = max(detections, key=lambda d: d["confidence"])
                            self._embeddings.setdefault(person_id, []).append(
                                np.array(best["embedding"], dtype=np.float32)
                            )
                            log.info(f"[face] loaded {person_id}/{img_file.name}")
                        else:
                            n_fail += 1
                            log.warning(f"[face] no face detected in {img_file}")
                    except Exception as e:
                        n_fail += 1
                        log.warning(f"[face] failed to load {img_file}: {e}")

            total = sum(len(v) for v in self._embeddings.values())
            log.info(f"[face] face db loaded: {len(self._embeddings)} persons, {total} embeddings")

            # ── Gallery quality diagnostics (temporary, for eval debugging) ──
            ids, embs = [], []
            for pid, emb_list in self._embeddings.items():
                for v in emb_list:
                    ids.append(pid)
                    embs.append(np.asarray(v, dtype=np.float32))
            self.load_diag = (
                f"{n_dirs}dirs/{total}emb "
                f"nodetect={n_dirs - len(self._embeddings)} fail={n_fail}"
            )
            if len(embs) >= 2:
                E = np.stack(embs)
                E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
                S = E @ E.T
                mask = np.ones_like(S, dtype=bool)
                np.fill_diagonal(mask, False)
                ids_arr = np.array(ids)
                mask &= ids_arr[:, None] != ids_arr[None, :]
                mask &= np.triu(np.ones_like(mask, dtype=bool))
                impostor = S[mask]
                self.load_diag += (
                    f" imp mean={impostor.mean():.3f} med={float(np.median(impostor)):.3f}"
                    f" p95={float(np.percentile(impostor, 95)):.3f} max={float(impostor.max()):.3f}"
                )
                pairs = sorted(
                    ((float(S[i, j]), ids[i], ids[j]) for i, j in np.argwhere(mask)),
                    reverse=True,
                )[:8]
                if pairs:
                    self.load_diag += " top:" + ",".join(
                        f"{a}~{b}={s:.3f}" for s, a, b in pairs
                    )
            log.info(f"[face][diag] db: {self.load_diag}")

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str, float]:
        """Return (person_id, similarity) for best match, or ("unknown", sim)."""
        with self._lock:
            if not self._embeddings:
                return "unknown", 0.0

            query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

            best_id = "unknown"
            best_sim = 0.0
            for person_id, embs in self._embeddings.items():
                embs_arr = np.array(embs, dtype=np.float32)
                embs_norm = embs_arr / (np.linalg.norm(embs_arr, axis=1, keepdims=True) + 1e-8)
                sims = embs_norm @ query_norm
                max_sim = float(np.max(sims))
                if max_sim > best_sim:
                    best_sim = max_sim
                    best_id = person_id

            if best_sim >= threshold:
                return best_id, best_sim
            return "unknown", best_sim

    def match_diag(self, embedding: np.ndarray, k: int = 3):
        """Return (top-k [(person_id, sim)], mean sim over all gallery persons)."""
        with self._lock:
            if not self._embeddings:
                return [], 0.0
            query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            per_person = []
            for person_id, embs in self._embeddings.items():
                embs_arr = np.array(embs, dtype=np.float32)
                embs_norm = embs_arr / (np.linalg.norm(embs_arr, axis=1, keepdims=True) + 1e-8)
                per_person.append((person_id, float(np.max(embs_norm @ query_norm))))
            per_person.sort(key=lambda x: -x[1])
            topk = per_person[:k]
            gmean = float(np.mean([s for _, s in per_person]))
            return topk, gmean

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._embeddings) == 0

    def count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._embeddings.values())


# ── EdgeFace + Detector Adapter ──────────────────────────────────────────────

# ArcFace 5-point reference for 112×112 alignment
_ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.3655],
], dtype=np.float32)

# Resize input to this width for detection (speed/accuracy trade-off)
_DETECT_TARGET_W = 1280

# SCRFD-500M-KPS postprocess constants (matches insightface model_zoo/scrfd.py)
_SCRFD_STRIDES = [8, 16, 32]
_SCRFD_NUM_ANCHORS = 2


def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, i])
        preds.append(points[:, 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


class YuNetDetector:
    """OpenCV FaceDetectorYN wrapper. Expects RGB input, returns original-coord detections."""

    def __init__(self, model_dir: str, confidence: float):
        import cv2
        yunet_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
        self._det = cv2.FaceDetectorYN_create(
            yunet_path, "", (320, 320),
            score_threshold=confidence,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._confidence = confidence

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        import cv2
        H, W = image_rgb.shape[:2]
        if W > _DETECT_TARGET_W:
            scale = _DETECT_TARGET_W / W
            new_w, new_h = _DETECT_TARGET_W, int(H * scale)
            img = cv2.resize(image_rgb, (new_w, new_h))
        else:
            scale = 1.0
            new_w, new_h = W, H
            img = image_rgb

        self._det.setInputSize((new_w, new_h))
        _, faces = self._det.detect(img)

        out = []
        if faces is None:
            return out
        for f in faces:
            if f[14] < self._confidence:
                continue
            x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            landmarks = np.array([
                [f[4], f[5]], [f[6], f[7]], [f[8], f[9]],
                [f[10], f[11]], [f[12], f[13]],
            ], dtype=np.float32)
            out.append({
                "bbox": [x / scale, y / scale, (x + w) / scale, (y + h) / scale],
                "landmarks": landmarks / scale,
                "confidence": float(f[14]),
            })
        return out


class SCRFDDetector:
    """SCRFD-500M-KPS via onnxruntime. Expects RGB input (converts to BGR internally),
    returns original-coord detections."""

    def __init__(self, model_dir: str, confidence: float):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        model_path = os.path.join(model_dir, "scrfd_500m_kps.onnx")
        self._sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        self._confidence = confidence
        self._cache = {}

    def _forward(self, det_img, thresh):
        import cv2
        H, W = det_img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, (W, H), (127.5, 127.5, 127.5), swapRB=True
        )
        out = self._sess.run(None, {self._input_name: blob})
        s_list, b_list, k_list = [], [], []
        for idx, stride in enumerate(_SCRFD_STRIDES):
            scores = out[idx]
            bbox_preds = out[idx + 3] * stride
            kps_preds = out[idx + 6] * stride
            h, w = H // stride, W // stride
            key = (h, w, stride)
            if key in self._cache:
                anchor = self._cache[key]
            else:
                anchor = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
                anchor = (anchor * stride).reshape(-1, 2)
                anchor = np.stack([anchor] * _SCRFD_NUM_ANCHORS, axis=1).reshape(-1, 2)
                if len(self._cache) < 100:
                    self._cache[key] = anchor
            pos = np.where(scores >= thresh)[0]
            b_list.append(_distance2bbox(anchor, bbox_preds)[pos])
            k_list.append(_distance2kps(anchor, kps_preds)[pos].reshape(-1, 5, 2))
            s_list.append(scores[pos])
        return s_list, b_list, k_list

    def detect(self, image_rgb: np.ndarray, input_size=(640, 640)) -> list[dict]:
        import cv2
        bgr = image_rgb[:, :, ::-1]
        im_ratio = bgr.shape[0] / bgr.shape[1]
        model_ratio = input_size[1] / input_size[0]
        if im_ratio > model_ratio:
            new_h = input_size[1]
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_size[0]
            new_h = int(new_w * im_ratio)
        det_scale = new_h / bgr.shape[0]
        resized = cv2.resize(bgr, (new_w, new_h))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        s, b, k = self._forward(det_img, self._confidence)
        if not s or sum(x.size for x in s) == 0:
            return []
        scores = np.vstack(s).ravel()
        boxes = np.vstack(b) / det_scale
        kpss = np.vstack(k) / det_scale
        order = scores.argsort()[::-1]
        scores, boxes, kpss = scores[order], boxes[order], kpss[order]

        # NMS
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        keep = []
        idx = np.arange(len(scores))
        while idx.size > 0:
            i = idx[0]
            keep.append(i)
            if idx.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[idx[1:]])
            yy1 = np.maximum(y1[i], y1[idx[1:]])
            xx2 = np.minimum(x2[i], x2[idx[1:]])
            yy2 = np.minimum(y2[i], y2[idx[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            iou = (w * h) / (areas[i] + areas[idx[1:]] - w * h)
            idx = idx[1:][iou <= 0.4]
        keep = np.array(keep, dtype=int)
        return [{
            "bbox": boxes[i],
            "landmarks": kpss[i],
            "confidence": float(scores[i]),
        } for i in keep]


class EdgeFaceAdapter:
    """Face detection (YuNet or SCRFD) + EdgeFace embedding extraction.

    YuNet (OpenCV FaceDetectorYN) is the proven default — ~250ms on CPU for 1280px.
    SCRFD-500M-KPS (onnxruntime) is an alternative detector for domain-shift testing.
    EdgeFace runs on CPU (~189ms) or GPU (~26ms).
    Model weights are auto-downloaded from juicefs.

    device='cpu':  ~440ms/frame, ~2.3 fps, no CUDA context — safe for 10 containers
    device='cuda': ~335ms/frame, ~3.0 fps, but 10 containers may OOM on 16GB Orin
    """

    def __init__(self, model_name: str, model_dir: str, device: str = "cpu",
                 confidence: float = 0.5, detector: str = "yunet"):
        import torch
        from torchvision import transforms

        self._device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        torch.set_num_threads(1)

        # Download weights from juicefs
        ckpt_path = _ensure_weights(model_name, model_dir, detector=detector)

        # ── Load EdgeFace backbone ──
        from backbones import get_model

        self._model_name = model_name
        self._model = get_model(model_name)
        state_dict = torch.load(ckpt_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device).eval()
        log.info(f"[face] EdgeFace loaded: {model_name}, device={self._device}")

        # ── Face detector ──
        self._confidence = confidence
        if detector == "scrfd":
            self._detector = SCRFDDetector(model_dir, confidence)
            log.info(f"[face] SCRFD-500M-KPS loaded, conf={confidence}")
        else:
            self._detector = YuNetDetector(model_dir, confidence)
            log.info(f"[face] YuNet loaded, conf={confidence}")

        # ── Preprocessing transform (same as training) ──
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def detect_and_embed(self, image: np.ndarray) -> list[dict]:
        """Detect faces, align, extract embeddings.

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            List of dicts with keys:
              - embedding: np.ndarray (512-dim)
              - bbox: [x1, y1, x2, y2] in pixel coords (original resolution)
              - confidence: float
        """
        import cv2
        import torch
        from PIL import Image

        detections = self._detector.detect(image)

        results = []
        for det in detections:
            # Affine align to 112×112 using 5 landmarks (original-coord)
            M = cv2.estimateAffinePartial2D(det["landmarks"], _ARCFACE_REF)[0]
            if M is None:
                continue
            aligned = cv2.warpAffine(image, M, (112, 112), borderValue=0)

            # EdgeFace embedding
            tensor = self._transform(Image.fromarray(aligned)).unsqueeze(0).to(self._device)
            with torch.no_grad():
                embedding = self._model(tensor)
            embedding = embedding.cpu().numpy().flatten()

            results.append({
                "embedding": embedding,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
            })

        return results


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _FaceNode(Node):
    """Per-topic face recognition node."""

    def __init__(self, input_topic: str, model: EdgeFaceAdapter,
                 face_db: FaceDatabase,
                 similarity_threshold: float, confidence: float, fps: float,
                 node_suffix: str):
        super().__init__(f"face_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/face"
        self._model = model
        self._face_db = face_db
        self._similarity_threshold = similarity_threshold
        self._confidence = confidence
        self._frame_interval = 1.0 / max(fps, 0.1)

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_inference_time = 0.0
        self._detect_count = 0
        self._diag_db_sent = False

    def start(self) -> dict:
        if self._sub is not None:
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                        name=f"face_worker_{self._input_topic}")
        self._worker.start()
        log.info(f"[face] started: {self._input_topic} → {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        log.info(f"[face] stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self._last_inference_time < self._frame_interval:
            return
        self._last_inference_time = now
        # Drop old frame if queue full (no backpressure)
        try:
            self._frame_queue.put_nowait(msg.data)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(msg.data)
            except queue.Full:
                pass

    def _inference_worker(self):
        import cv2
        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue

                H, W = frame.shape[:2]
                rgb_frame = frame[:, :, ::-1]  # BGR → RGB

                # ── Detect + embed ──
                detections = self._model.detect_and_embed(rgb_frame)

                # The evaluator expects one face object with identity.person_id,
                # not a list. Select the highest-confidence detection.
                result = {}
                if detections:
                    det = max(detections, key=lambda item: item["confidence"])
                    x1, y1, x2, y2 = det["bbox"]
                    embedding = np.array(det["embedding"], dtype=np.float32)
                    person_id, similarity = self._face_db.match(
                        embedding, self._similarity_threshold
                    )
                    topk, gmean = self._face_db.match_diag(embedding)
                    diag = {
                        "faces": len(detections),
                        "face_px": round(det["bbox"][2] - det["bbox"][0]),
                        "top3": [[pid, round(s, 3)] for pid, s in topk],
                        "gmean": round(gmean, 3),
                    }
                    if not self._diag_db_sent and self._face_db.load_diag:
                        diag["db"] = self._face_db.load_diag
                        self._diag_db_sent = True
                    # Evaluator expects normalized [x, y, w, h] (verified
                    # empirically: xywh -> accuracy 0.92 vs 0.16 for xyxy)
                    result = {
                        "detect_confidence": round(det["confidence"], 4),
                        "bbox_relative": [
                            round(x1 / W, 6),
                            round(y1 / H, 6),
                            round((x2 - x1) / W, 6),
                            round((y2 - y1) / H, 6),
                        ],
                        "identity": {
                            "person_id": person_id,
                            "confidence": round(similarity, 4),
                        },
                        "diag": diag,
                    }

                msg = String()
                msg.data = json.dumps(result, ensure_ascii=False)
                self._pub.publish(msg)

                self._detect_count += 1
                log.info(f"[face] {len(detections)} face(s), result="
                         f"{result.get('identity', {}).get('person_id')} (detect+match done)")

            except Exception as e:
                log.error(f"[face] inference error: {e}", exc_info=True)


# ── Plugin class ──────────────────────────────────────────────────────────────

class FaceRecognitionPlugin:
    PREFIX = "face"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._model_name = plugin_cfg.get("model", DEFAULT_MODEL_NAME)
        self._device = plugin_cfg.get("device", "cpu")
        self._detector = plugin_cfg.get("detector", "yunet")
        self._face_db_dir = plugin_cfg.get("face_db_dir") or os.getenv("FACE_DB_DIR", "/workspace/face_db")
        self._model_dir = plugin_cfg.get("model_dir", "/models/face")
        self._similarity_threshold = float(plugin_cfg.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD))
        self._confidence = float(plugin_cfg.get("confidence", 0.5))
        self._fps = int(plugin_cfg.get("fps", 3))

        self._model = None
        self._model_loading = False
        self._model_load_error = None
        self._model_lock = threading.Lock()
        self._pending_starts: list[tuple[str, str]] = []

        self._face_db = FaceDatabase()
        self._nodes: dict[str, _FaceNode] = {}
        self._instance_configs: dict[str, dict] = {}

        log.info(f"[face] plugin init: model={self._model_name}, device={self._device}, "
                 f"detector={self._detector}, face_db_dir={self._face_db_dir}")

        # Pre-load model at startup so it's ready before evaluation calls start.
        # The benchmark calls start then immediately publishes images; if the model
        # is still loading the ROS2 subscriber doesn't exist yet and frames are lost.
        self._start_model_loading()

    def _start_model_loading(self):
        """Start loading model in background. Processes pending starts when done."""
        if self._model_loading or self._model is not None:
            return
        self._model_loading = True

        def _bg_load():
            try:
                self._ensure_model()
                self._model_loading = False
                log.info("[face] model loaded, processing pending starts")
                for node_key, input_topic in self._pending_starts:
                    if node_key not in self._nodes:
                        self._start_node(node_key, input_topic)
                self._pending_starts.clear()
            except Exception as e:
                self._model_loading = False
                self._model_load_error = str(e)
                log.error(f"[face] model load failed: {e}", exc_info=True)

        threading.Thread(target=_bg_load, daemon=True, name="face_model_load").start()

    def _ensure_model(self):
        """Load model and identity library in background."""
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return

            self._model = EdgeFaceAdapter(
                self._model_name, self._model_dir, self._device,
                confidence=self._confidence, detector=self._detector,
            )

            # Load identity library
            self._face_db.load_from_dir(self._face_db_dir, self._model)

    def _start_node(self, node_key: str, input_topic: str):
        """Create and start a FaceNode for the given topic."""
        icfg = self._instance_configs.get(node_key, {})
        similarity_threshold = float(icfg.get("similarity_threshold", self._similarity_threshold))
        confidence = float(icfg.get("confidence", self._confidence))
        fps = int(icfg.get("fps", self._fps))
        suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
        node = _FaceNode(input_topic, self._model, self._face_db,
                        similarity_threshold, confidence, fps, suffix)
        self._executor.add_node(node)
        self._nodes[node_key] = node
        node.start()
        log.info(f"[face] node started (background): {input_topic}")

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._model_loading:
                return {
                    "name": "FaceRecognition", "manufacture": "Embodied",
                    "model": self._model_name,
                    "state": "loading",
                    "desc": "Loading EdgeFace model and identity library...",
                }
            if self._model_load_error:
                return {
                    "name": "FaceRecognition", "manufacture": "Embodied",
                    "model": self._model_name,
                    "state": "error",
                    "desc": f"Model load failed: {self._model_load_error}",
                }
            instances = {}
            for key, node in self._nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "detect_count": node._detect_count,
                }
            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                input_topic = node._input_topic
            elif not input_topic and self._nodes:
                first_node = next(iter(self._nodes.values()))
                input_topic = first_node._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = [{"topic": f"{input_topic}/face", "format": "data/json"}] if input_topic else []
            state = "running" if instances else "idle"
            return {
                "name": "FaceRecognition", "manufacture": "Embodied",
                "model": self._model_name,
                "state": state,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": f"EdgeFace + {self._detector} face recognition",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                if self._model is None:
                    if self._model_loading:
                        # Queue this start; it will be processed when model finishes loading
                        self._pending_starts.append((node_key, input_topic))
                        return {"state": "loading", "input": input_topic, "output": f"{input_topic}/face",
                                "message": "Model is loading, node will start automatically when ready"}
                    if self._model_load_error:
                        return {"state": "error", "message": f"Model failed to load: {self._model_load_error}"}
                    # Should not reach here since __init__ starts loading, but handle it
                    self._pending_starts.append((node_key, input_topic))
                    self._start_model_loading()
                    return {"state": "loading", "input": input_topic, "output": f"{input_topic}/face",
                            "message": "Model loading in background, will start automatically"}
                self._start_node(node_key, input_topic)
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                dispose_node(self._executor, node, label=f"face/{instance_id}")
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                results = []
                for key in list(self._nodes.keys()):
                    node = self._nodes[key]
                    node.stop()
                    dispose_node(self._executor, node, label=f"face/{key}")
                    del self._nodes[key]
                    results.append(key)
                return {"state": "idle", "stopped_instances": results}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    input_topic = node._input_topic
                    node.stop()
                    dispose_node(self._executor, node, label=f"face/{instance_id}")
                    del self._nodes[instance_id]
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            else:
                if "model" in cfg:
                    self._model_name = cfg["model"]
                if "detector" in cfg:
                    self._detector = cfg["detector"]
                if "device" in cfg:
                    self._device = cfg["device"]
                if "face_db_dir" in cfg:
                    self._face_db_dir = cfg["face_db_dir"]
                if "model_dir" in cfg:
                    self._model_dir = cfg["model_dir"]
                if "similarity_threshold" in cfg:
                    self._similarity_threshold = float(cfg["similarity_threshold"])
                if "confidence" in cfg:
                    self._confidence = float(cfg["confidence"])
                if "fps" in cfg:
                    self._fps = int(cfg["fps"])
                return {"status": "configured", "config": cfg}

        return None
