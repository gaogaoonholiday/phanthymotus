#!/usr/bin/env python3
"""
plugins/face.py — FaceRecognitionPlugin: EdgeFace + YuNet face recognition.

Pipeline: CompressedImage → YuNet detect → align (112×112) → EdgeFace embed → identity match
Downloads weights from juicefs (http://172.28.4.81:34567/).
Outputs benchmark-compliant JSON (faces array only, no wrapper):
  [
    {
      "detect_confidence": 0.95,
      "bbox_relative": [x1, y1, x2, y2],  # normalized [0-1]
      "identity": {
        "person_id": "n000001",
        "confidence": 0.91
      }
    }
  ]

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
DEFAULT_MODEL_NAME = "edgeface_s_gamma_05"  # 3.65M params, ~14MB checkpoint

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
                "description": "Face Recognition — detect and identify faces using EdgeFace + YuNet/MTCNN",
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
                "inference_backend": {
                    "type": "string",
                    "enum": ["pytorch", "onnx"],
                    "description": "EdgeFace inference backend: 'onnx' uses ONNX Runtime (2-3x faster on CPU), 'pytorch' uses PyTorch",
                    "default": "pytorch",
                    "scope": "shared",
                },
                "detector": {
                    "type": "string",
                    "enum": ["yunet", "mtcnn"],
                    "description": "Face detector: 'yunet' (OpenCV, CPU, fast) or 'mtcnn' (PyTorch, GPU, accurate landmarks)",
                    "default": "yunet",
                    "scope": "shared",
                },
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "face recognition result"}],
    }
]


# ── Model download ────────────────────────────────────────────────────────────

def _ensure_weights(model_name: str, model_dir: str,
                    need_onnx: bool = False, need_mtcnn: bool = False) -> str:
    """Download model weights from juicefs if not present.

    Args:
        model_name: EdgeFace model name (e.g. edgeface_s_gamma_05)
        model_dir: Directory for model weights
        need_onnx: If True, download .onnx; otherwise download .pt
        need_mtcnn: If True, download MTCNN .npy weights to edgeface_src

    Returns:
        Path to the checkpoint (.pt) or ONNX model (.onnx)
    """
    os.makedirs(model_dir, exist_ok=True)

    if need_onnx:
        # EdgeFace ONNX model
        onnx_filename = f"{model_name}.onnx"
        onnx_path = os.path.join(model_dir, onnx_filename)
        if not os.path.exists(onnx_path):
            url = f"{_MODEL_BASE_URL}/{onnx_filename}"
            log.info(f"[face] downloading {onnx_filename} from {url} → {onnx_path}")
            urllib.request.urlretrieve(url, onnx_path)
            log.info(f"[face] download complete: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
        # Still need the .pt for face_db loading (FaceDatabase uses PyTorch adapter)
        ckpt_filename = f"{model_name}.pt"
        ckpt_path = os.path.join(model_dir, ckpt_filename)
        if not os.path.exists(ckpt_path):
            url = f"{_MODEL_BASE_URL}/{ckpt_filename}"
            log.info(f"[face] downloading {ckpt_filename} from {url} → {ckpt_path}")
            urllib.request.urlretrieve(url, ckpt_path)
            log.info(f"[face] download complete: {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:.1f} MB)")
        # Download MTCNN weights if needed
        if need_mtcnn:
            _ensure_mtcnn_weights()
        return ckpt_path

    # EdgeFace checkpoint
    ckpt_filename = f"{model_name}.pt"
    ckpt_path = os.path.join(model_dir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        url = f"{_MODEL_BASE_URL}/{ckpt_filename}"
        log.info(f"[face] downloading {ckpt_filename} from {url} → {ckpt_path}")
        urllib.request.urlretrieve(url, ckpt_path)
        log.info(f"[face] download complete: {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:.1f} MB)")

    # YuNet ONNX model (needed for yunet detector)
    if not need_mtcnn:
        yunet_filename = "face_detection_yunet_2023mar.onnx"
        yunet_path = os.path.join(model_dir, yunet_filename)
        if not os.path.exists(yunet_path):
            url = f"{_MODEL_BASE_URL}/{yunet_filename}"
            log.info(f"[face] downloading {yunet_filename} from {url}")
            urllib.request.urlretrieve(url, yunet_path)
            log.info(f"[face] download complete: {yunet_path} ({os.path.getsize(yunet_path) / 1e6:.1f} MB)")

    # Download MTCNN weights if needed
    if need_mtcnn:
        _ensure_mtcnn_weights()

    return ckpt_path


def _ensure_mtcnn_weights():
    """Download MTCNN .npy weights from juicefs to the expected path.

    MTCNN loads weights via np.load('mtcnn_pytorch/src/weights/pnet.npy')
    with a relative path, so they must be in the edgeface_src/face_alignment/
    directory tree. On the Docker image this path is /work/edgeface_src/
    face_alignment/.
    """
    mtcnn_weights_dir = os.path.join(
        _EDGEFACE_SRC, "face_alignment", "mtcnn_pytorch", "src", "weights",
    )
    os.makedirs(mtcnn_weights_dir, exist_ok=True)

    for name in ("pnet.npy", "rnet.npy", "onet.npy"):
        dst = os.path.join(mtcnn_weights_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        url = f"{_MODEL_BASE_URL}/mtcnn/{name}"
        log.info(f"[face] downloading MTCNN {name} from {url}")
        urllib.request.urlretrieve(url, dst)
        log.info(f"[face] MTCNN {name} downloaded ({os.path.getsize(dst) / 1e3:.0f} KB)")


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

    def load_from_dir(self, db_dir: str, adapter: "EdgeFaceAdapter"):
        """Load identity library: detect + embed all faces in db_dir."""
        db_path = Path(db_dir)
        if not db_path.exists():
            log.warning(f"[face] face db dir not found: {db_dir}")
            return

        from PIL import Image

        with self._lock:
            self._embeddings.clear()

            for person_dir in sorted(db_path.iterdir()):
                if not person_dir.is_dir():
                    continue
                person_id = person_dir.name
                for img_file in sorted(person_dir.iterdir()):
                    if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                        continue
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
                    except Exception as e:
                        log.warning(f"[face] failed to load {img_file}: {e}")

            total = sum(len(v) for v in self._embeddings.values())
            log.info(f"[face] face db loaded: {len(self._embeddings)} persons, {total} embeddings")

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

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._embeddings) == 0

    def count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._embeddings.values())


# ── EdgeFace + YuNet Adapter ──────────────────────────────────────────────────

# ArcFace 5-point reference for 112×112 alignment
_ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.3655],
], dtype=np.float32)

# Resize input to this width for detection (speed/accuracy trade-off)
_DETECT_TARGET_W = 1280


class EdgeFaceAdapter:
    """Face detection + EdgeFace embedding extraction.

    Two detector options:
      - 'yunet': OpenCV FaceDetectorYN, single-stage, CPU only (~250ms)
      - 'mtcnn': PyTorch MTCNN three-stage cascade, runs on GPU (~2s) or CPU

    Two inference backends for EdgeFace:
      - 'pytorch': PyTorch EdgeFace, ~189ms/embed on CPU (1 thread)
      - 'onnx':    ONNX Runtime, ~60-80ms/embed on CPU (2-3x faster)

    Batch embedding: all detected faces are stacked into one forward pass.

    detector='yunet', device='cpu':  ~440ms/frame, ~2.3 fps
    detector='mtcnn', device='cuda': EdgeFace ~26ms + MTCNN ~2s (first call),
                  MTCNN is slow but very accurate. Best for quality.
    """

    def __init__(self, model_name: str, model_dir: str, device: str = "cpu",
                 confidence: float = 0.5, inference_backend: str = "pytorch",
                 detector: str = "yunet"):
        import torch

        self._device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        torch.set_num_threads(1)
        self._inference_backend = inference_backend
        self._detector_type = detector
        self._confidence = confidence

        # Download weights from juicefs
        ckpt_path = _ensure_weights(
            model_name, model_dir,
            need_onnx=(inference_backend == "onnx"),
            need_mtcnn=(detector == "mtcnn"),
        )

        if inference_backend == "onnx":
            self._init_onnx(model_name, model_dir)
        else:
            self._init_pytorch(model_name, ckpt_path)

        # ── Face detector ──
        if detector == "mtcnn":
            self._init_mtcnn()
        else:
            self._init_yunet(model_dir, confidence)

        # ── Preprocessing transform (same as training) ──
        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_pytorch(self, model_name: str, ckpt_path: str):
        """Load PyTorch EdgeFace model."""
        import torch
        from backbones import get_model

        self._model_name = model_name
        self._model = get_model(model_name)
        state_dict = torch.load(ckpt_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device).eval()
        self._ort_session = None
        log.info(f"[face] EdgeFace loaded (pytorch): {model_name}, device={self._device}")

    def _init_onnx(self, model_name: str, model_dir: str):
        """Load ONNX Runtime EdgeFace session."""
        import onnxruntime as ort

        onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}. "
                f"Run export_edgeface_onnx.py first."
            )

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        providers = ["CPUExecutionProvider"]
        if str(self._device) == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._ort_session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
        self._model = None
        self._model_name = model_name
        log.info(f"[face] EdgeFace loaded (onnx): {model_name}, providers={providers}")

    def _init_yunet(self, model_dir: str, confidence: float):
        """Load YuNet face detector (OpenCV FaceDetectorYN, CPU only)."""
        import cv2
        yunet_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
        self._detector = cv2.FaceDetectorYN_create(
            yunet_path, "", (320, 320),
            score_threshold=confidence,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._mtcnn = None
        log.info(f"[face] YuNet loaded, conf={confidence}")

    def _init_mtcnn(self):
        """Load MTCNN face detector (PyTorch, runs on GPU or CPU).

        MTCNN three-stage cascade (PNet→RNet→ONet) with image pyramid.
        Very accurate landmark detection. On GPU ~2s/frame, on CPU ~6.6s/frame.
        Weights are downloaded from juicefs by _ensure_weights.
        """
        from face_alignment import mtcnn as mtcnn_mod

        device_str = "cuda:0" if str(self._device) == "cuda" else "cpu"
        self._mtcnn = mtcnn_mod.MTCNN(device=device_str, crop_size=(112, 112))
        self._detector = None
        log.info(f"[face] MTCNN loaded, device={device_str}")

    def _embed_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a batch of aligned 112×112 face images.

        Args:
            aligned_faces: list of RGB numpy arrays, each (112, 112, 3)

        Returns:
            (N, 512) embeddings array
        """
        import torch
        from PIL import Image

        if not aligned_faces:
            return np.zeros((0, 512), dtype=np.float32)

        # Stack all faces into one batch tensor
        tensors = [self._transform(Image.fromarray(f)) for f in aligned_faces]
        batch = torch.stack(tensors).to(self._device)

        if self._inference_backend == "onnx" and self._ort_session is not None:
            embeddings = self._ort_session.run(
                ["embedding"], {"input": batch.cpu().numpy()}
            )[0]
        else:
            with torch.no_grad():
                embeddings = self._model(batch).cpu().numpy()

        return embeddings

    def detect_and_embed(self, image: np.ndarray) -> list[dict]:
        """Detect faces, align, extract embeddings.

        Uses batch embedding: all faces are aligned first, then passed
        through EdgeFace in a single forward pass.

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            List of dicts with keys:
              - embedding: np.ndarray (512-dim)
              - bbox: [x1, y1, x2, y2] in pixel coords (original resolution)
              - confidence: float
        """
        if self._detector_type == "mtcnn":
            return self._detect_and_embed_mtcnn(image)
        return self._detect_and_embed_yunet(image)

    def _detect_and_embed_mtcnn(self, image: np.ndarray) -> list[dict]:
        """MTCNN detection + EdgeFace embedding (GPU pipeline)."""
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        W_orig, H_orig = pil_img.size

        # MTCNN detect + align in one call
        boxes, landmarks = self._mtcnn.detect_faces(
            pil_img,
            self._mtcnn.min_face_size,
            self._mtcnn.thresholds,
            self._mtcnn.nms_thresholds,
            self._mtcnn.factor,
        )

        if len(boxes) == 0:
            return []

        # Align all faces using MTCNN's warp_and_crop_face
        from face_alignment.mtcnn_pytorch.src.align_trans import (
            get_reference_facial_points, warp_and_crop_face,
        )
        ref = get_reference_facial_points(default_square=True)

        aligned_faces = []
        bboxes = []
        confidences = []

        for i in range(len(boxes)):
            conf = float(boxes[i][4])
            if conf < self._confidence:
                continue
            facial5points = [
                [landmarks[i][j], landmarks[i][j + 5]] for j in range(5)
            ]
            warped = warp_and_crop_face(
                image, facial5points, ref, crop_size=(112, 112),
            )
            aligned_faces.append(warped)
            bboxes.append([
                float(boxes[i][0]), float(boxes[i][1]),
                float(boxes[i][2]), float(boxes[i][3]),
            ])
            confidences.append(conf)

        if not aligned_faces:
            return []

        embeddings = self._embed_batch(aligned_faces)

        results = []
        for emb, bbox, conf in zip(embeddings, bboxes, confidences):
            results.append({
                "embedding": emb,
                "bbox": bbox,
                "confidence": conf,
            })
        return results

    def _detect_and_embed_yunet(self, image: np.ndarray) -> list[dict]:
        """YuNet detection + EdgeFace embedding (CPU pipeline)."""
        import cv2

        H_orig, W_orig = image.shape[:2]

        # ── Resize for detection (speed) ──
        if W_orig > _DETECT_TARGET_W:
            scale = _DETECT_TARGET_W / W_orig
            new_h, new_w = int(H_orig * scale), _DETECT_TARGET_W
            img_det = cv2.resize(image, (new_w, new_h))
        else:
            scale = 1.0
            new_h, new_w = H_orig, W_orig
            img_det = image

        # YuNet expects (W, H) for setInputSize
        self._detector.setInputSize((new_w, new_h))
        _, faces = self._detector.detect(img_det)

        if faces is None:
            return []

        # ── Collect aligned faces + metadata ──
        aligned_faces = []
        bboxes = []
        confidences = []

        for f in faces:
            if f[14] < self._confidence:
                continue

            # 5 landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
            landmarks = np.array([
                [f[4], f[5]], [f[6], f[7]], [f[8], f[9]],
                [f[10], f[11]], [f[12], f[13]],
            ], dtype=np.float32)

            # Affine align to 112×112
            M = cv2.estimateAffinePartial2D(landmarks, _ARCFACE_REF)[0]
            if M is None:
                continue
            aligned = cv2.warpAffine(img_det, M, (112, 112), borderValue=0)
            aligned_faces.append(aligned)

            # Scale bbox back to original resolution
            x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            bboxes.append([x / scale, y / scale, (x + w) / scale, (y + h) / scale])
            confidences.append(float(f[14]))

        if not aligned_faces:
            return []

        # ── Batch embedding extraction ──
        embeddings = self._embed_batch(aligned_faces)

        results = []
        for i, (emb, bbox, conf) in enumerate(zip(embeddings, bboxes, confidences)):
            results.append({
                "embedding": emb,
                "bbox": bbox,
                "confidence": conf,
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

                # ── Build benchmark-compliant output (faces array only) ──
                faces = []
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]

                    # Normalize bbox to [0, 1]
                    bbox_relative = [
                        round(x1 / W, 6),
                        round(y1 / H, 6),
                        round(x2 / W, 6),
                        round(y2 / H, 6),
                    ]

                    # ── Identity matching ──
                    embedding = np.array(det["embedding"], dtype=np.float32)
                    person_id, similarity = self._face_db.match(
                        embedding, self._similarity_threshold
                    )

                    faces.append({
                        "detect_confidence": round(det["confidence"], 4),
                        "bbox_relative": bbox_relative,
                        "identity": {
                            "person_id": person_id,
                            "confidence": round(similarity, 4),
                        }
                    })

                msg = String()
                msg.data = json.dumps(faces, ensure_ascii=False)
                self._pub.publish(msg)

                self._detect_count += 1
                identities = [f["identity"]["person_id"] for f in faces]
                log.info(f"[face] {len(faces)} face(s): {identities} "
                         f"(detect+match done)")

            except Exception as e:
                log.error(f"[face] inference error: {e}", exc_info=True)


# ── Plugin class ──────────────────────────────────────────────────────────────

class FaceRecognitionPlugin:
    PREFIX = "face"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._model_name = plugin_cfg.get("model", DEFAULT_MODEL_NAME)
        self._device = plugin_cfg.get("device", "cpu")
        self._inference_backend = plugin_cfg.get("inference_backend", "pytorch")
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
                 f"backend={self._inference_backend}, detector={self._detector}, "
                 f"face_db_dir={self._face_db_dir}")

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
                confidence=self._confidence,
                inference_backend=self._inference_backend,
                detector=self._detector,
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
                "desc": "EdgeFace + YuNet face recognition",
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
                if "device" in cfg:
                    self._device = cfg["device"]
                if "inference_backend" in cfg:
                    self._inference_backend = cfg["inference_backend"]
                if "detector" in cfg:
                    self._detector = cfg["detector"]
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
