#!/usr/bin/env python3
"""Test full YuNet + EdgeFace ONNX pipeline on Jetson — simulates face.py logic."""
import sys, os, time, numpy as np, cv2

EDGEFACE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edgeface_src")
sys.path.insert(0, EDGEFACE_SRC)

import onnxruntime as ort
import torch
torch.set_num_threads(1)

model_dir = os.environ.get("FACE_MODEL_DIR", "/home/develop/dengshiwei/yunet_models")
onnx_path = os.path.join(model_dir, "edgeface_s_gamma_05.onnx")
yunet_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1
sess = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])

detector = cv2.FaceDetectorYN_create(yunet_path, "", (320, 320), score_threshold=0.5, nms_threshold=0.3, top_k=5000)

_ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.3655],
], dtype=np.float32)

_DETECT_TARGET_W = 1280


def detect_and_embed(image):
    H_orig, W_orig = image.shape[:2]
    if W_orig > _DETECT_TARGET_W:
        scale = _DETECT_TARGET_W / W_orig
        new_h, new_w = int(H_orig * scale), _DETECT_TARGET_W
        img_det = cv2.resize(image, (new_w, new_h))
    else:
        scale = 1.0
        new_h, new_w = H_orig, W_orig
        img_det = image

    detector.setInputSize((new_w, new_h))
    _, faces = detector.detect(img_det)
    if faces is None:
        return []

    aligned_faces = []
    results = []
    for f in faces:
        if f[14] < 0.5:
            continue
        landmarks = np.array([
            [f[4], f[5]], [f[6], f[7]], [f[8], f[9]],
            [f[10], f[11]], [f[12], f[13]],
        ], dtype=np.float32)
        M = cv2.estimateAffinePartial2D(landmarks, _ARCFACE_REF)[0]
        if M is None:
            continue
        aligned = cv2.warpAffine(img_det, M, (112, 112), borderValue=0)
        aligned_faces.append(aligned)
        x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
        bbox = [x / scale, y / scale, (x + w) / scale, (y + h) / scale]
        results.append({"bbox": bbox, "confidence": float(f[14])})

    if not aligned_faces:
        return []

    tensors = [f.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0 for f in aligned_faces]
    batch = np.stack(tensors)
    embeddings = sess.run(["embedding"], {"input": batch})[0]
    for i, emb in enumerate(embeddings):
        results[i]["embedding"] = emb
    return results


if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else "/home/develop/dengshiwei/gpt_日系校园穿搭sexiestadjusted7.jpg"
    img = cv2.imread(test_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image: {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    results = detect_and_embed(img_rgb)
    print(f"Warmup: {len(results)} faces")
    if results:
        print(f"  bbox: {results[0]['bbox']}")
        print(f"  conf: {results[0]['confidence']:.3f}")
        print(f"  emb dim: {len(results[0]['embedding'])}")

    times = []
    for i in range(5):
        t0 = time.time()
        results = detect_and_embed(img_rgb)
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    print(f"Full pipeline avg: {avg*1000:.0f}ms ({1/avg:.1f} fps)")
