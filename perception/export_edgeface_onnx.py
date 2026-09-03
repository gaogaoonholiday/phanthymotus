#!/usr/bin/env python3
"""
EdgeFace ONNX 导出脚本 — 将 PyTorch EdgeFace 模型导出为 ONNX 格式

导出后可在 face.py 中设置 inference_backend: onnx 来使用 ONNX Runtime 推理，
CPU 上比 PyTorch 快 2-3x (189ms → 60-80ms per embedding)。

用法:
    python3 export_edgeface_onnx.py

输出:
    /models/face/edgeface_s_gamma_05.onnx

导出后需将 ONNX 文件上传到 juicefs:
    cp /models/face/edgeface_s_gamma_05.onnx /mnt/data/face/
    (开发机上 /mnt/data/face 映射到 juicefs http://172.28.4.81:34567/face)
"""

import sys, os

EDGEFACE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edgeface_src")
if EDGEFACE_SRC not in sys.path:
    sys.path.insert(0, EDGEFACE_SRC)

import numpy as np
import torch
from backbones import get_model

MODEL_NAME = "edgeface_s_gamma_05"
OUTPUT_DIR = os.environ.get("FACE_MODEL_DIR", "/models/face")
CKPT_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.pt")
ONNX_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = get_model(MODEL_NAME)
state_dict = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 112, 112)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={
        "input": {0: "batch"},
        "embedding": {0: "batch"},
    },
    opset_version=13,
)

print(f"Exported: {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1e6:.1f} MB)")

# Verify
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
out = sess.run(None, {"input": dummy_input.numpy()})
print(f"Verification: output shape = {out[0].shape}, dtype = {out[0].dtype}")

# Compare PyTorch vs ONNX outputs
with torch.no_grad():
    pt_out = model(dummy_input).numpy()
diff = np.abs(pt_out - out[0]).max()
print(f"Max diff PyTorch vs ONNX: {diff:.6e}")
print("OK")

# Upload to juicefs if mounted
juicefs_dir = "/mnt/data/face"
if os.path.isdir(juicefs_dir):
    import shutil
    dst = os.path.join(juicefs_dir, f"{MODEL_NAME}.onnx")
    shutil.copy2(ONNX_PATH, dst)
    print(f"Copied to juicefs: {dst}")
else:
    print(f"Juicefs not mounted at {juicefs_dir}, upload manually")
