#!/usr/bin/env python3
"""Benchmark MTCNN + EdgeFace on GPU — measure VRAM and speed.

Tests whether MTCNN + EdgeFace on GPU satisfies the <10% GPU (<1.6GB)
constraint for 10 containers on Jetson Orin 16GB.

Usage:
    cd /tmp/phanthymotus/perception
    python3 bench_mtcnn_gpu.py [test_image]
"""

import os, sys, time, gc, traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGEFACE_SRC = os.path.join(SCRIPT_DIR, "edgeface_src")
sys.path.insert(0, EDGEFACE_SRC)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image

# Suppress the "is" vs "==" SyntaxWarning from align_trans.py
import importlib
import face_alignment.mtcnn_pytorch.src.align_trans  # noqa

test_image_path = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(SCRIPT_DIR, "..", "gpt_日系校园穿搭sexiestadjusted7.jpg")

print(f"[bench] test_image: {test_image_path}")
print(f"[bench] torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[bench] GPU: {torch.cuda.get_device_name(0)}")
print()


# ── Memory tracking ──
def get_gpu_mem():
    if not torch.cuda.is_available():
        return 0, 0
    allocated = torch.cuda.memory_allocated() / 1e6  # MB
    reserved = torch.cuda.memory_reserved() / 1e6
    return allocated, reserved


def get_sys_mem():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"): total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"): avail = int(line.split()[1]) // 1024
        used = total - avail
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"): rss = int(line.split()[1]) // 1024
        return rss, used, total
    except:
        return 0, 0, 0


def mem(label):
    rss, used, total = get_sys_mem()
    alloc, reserved = get_gpu_mem()
    print(f"  [mem] {label}: RSS={rss}MB, sys={used}/{total}MB, "
          f"GPU alloc={alloc:.1f}MB, GPU reserved={reserved:.1f}MB")
    return rss, used, total


# ── ArcFace reference for 112×112 ──
_ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.3655],
], dtype=np.float32)


def bench_mtcnn_edgeface_gpu():
    print(f"\n{'='*70}")
    print(f"MTCNN + EdgeFace on GPU — VRAM & speed")
    print(f"{'='*70}")

    mem("before anything")

    # ── Init CUDA context ──
    torch.cuda.init()
    _ = torch.zeros(1).cuda()
    mem("after CUDA init")

    # ── Load MTCNN ──
    from face_alignment import mtcnn as mtcnn_mod
    mtcnn_detector = mtcnn_mod.MTCNN(device="cuda:0", crop_size=(112, 112))
    mem("after MTCNN load")

    # ── Load EdgeFace ──
    from backbones import get_model
    model_name = "edgeface_s_gamma_05"
    ckpt_path = os.path.join(EDGEFACE_SRC, f"{model_name}.pt")
    model = get_model(model_name)
    state_dict = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model.to("cuda").eval()
    mem("after EdgeFace load")

    # ── Transform ──
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ── Load image ──
    pil_img = Image.open(test_image_path).convert("RGB")
    img_arr = np.array(pil_img)
    H, W = img_arr.shape[:2]
    print(f"  image: {W}×{H}")

    # ── Warmup ──
    print("\n  --- Warmup ---")
    t0 = time.time()
    boxes, landmarks = mtcnn_detector.detect_faces(
        pil_img, mtcnn_detector.min_face_size,
        mtcnn_detector.thresholds, mtcnn_detector.nms_thresholds,
        mtcnn_detector.factor,
    )
    t1 = time.time()
    print(f"  MTCNN detect: {(t1-t0)*1000:.0f}ms, {len(boxes)} faces")

    if len(boxes) > 0:
        from face_alignment.mtcnn_pytorch.src.align_trans import (
            get_reference_facial_points, warp_and_crop_face,
        )
        ref = get_reference_facial_points(default_square=True)
        aligned_faces = []
        for i in range(len(boxes)):
            facial5points = [
                [landmarks[i][j], landmarks[i][j + 5]] for j in range(5)
            ]
            warped = warp_and_crop_face(img_arr, facial5points, ref, crop_size=(112, 112))
            aligned_faces.append(warped)

        tensors = [transform(Image.fromarray(f)) for f in aligned_faces]
        batch = torch.stack(tensors).cuda()
        with torch.no_grad():
            emb = model(batch)
        t2 = time.time()
        print(f"  EdgeFace embed: {(t2-t1)*1000:.0f}ms, shape={emb.shape}")
        print(f"  Total: {(t2-t0)*1000:.0f}ms")

    mem("after warmup")

    # ── Benchmark ──
    print("\n  --- Benchmark (5 iterations) ---")
    times_detect = []
    times_embed = []
    times_total = []

    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.time()

        boxes, landmarks = mtcnn_detector.detect_faces(
            pil_img, mtcnn_detector.min_face_size,
            mtcnn_detector.thresholds, mtcnn_detector.nms_thresholds,
            mtcnn_detector.factor,
        )
        torch.cuda.synchronize()
        t1 = time.time()

        if len(boxes) > 0:
            from face_alignment.mtcnn_pytorch.src.align_trans import (
                get_reference_facial_points, warp_and_crop_face,
            )
            ref = get_reference_facial_points(default_square=True)
            aligned_faces = []
            for j in range(len(boxes)):
                facial5points = [
                    [landmarks[j][k], landmarks[j][k + 5]] for k in range(5)
                ]
                warped = warp_and_crop_face(img_arr, facial5points, ref, crop_size=(112, 112))
                aligned_faces.append(warped)

            tensors = [transform(Image.fromarray(f)) for f in aligned_faces]
            batch = torch.stack(tensors).cuda()
            with torch.no_grad():
                emb = model(batch)
            torch.cuda.synchronize()
            t2 = time.time()
        else:
            t2 = t1

        dt = t2 - t0
        times_detect.append(t1 - t0)
        times_embed.append(t2 - t1)
        times_total.append(dt)
        print(f"  [iter {i}] detect={ (t1-t0)*1000:.0f}ms embed={ (t2-t1)*1000:.0f}ms "
              f"total={dt*1000:.0f}ms faces={len(boxes)}")

    avg_det = sum(times_detect) / len(times_detect)
    avg_emb = sum(times_embed) / len(times_embed)
    avg_tot = sum(times_total) / len(times_total)
    print(f"\n  MTCNN detect avg: {avg_det*1000:.0f}ms")
    print(f"  EdgeFace embed avg: {avg_emb*1000:.0f}ms")
    print(f"  Total avg: {avg_tot*1000:.0f}ms ({1/avg_tot:.1f} fps)")

    mem("after benchmark")

    # ── VRAM estimate for 10 containers ──
    alloc, reserved = get_gpu_mem()
    print(f"\n  ── VRAM estimate for 10 containers ──")
    print(f"  Current GPU allocated: {alloc:.1f}MB")
    print(f"  Current GPU reserved:  {reserved:.1f}MB")
    # CUDA context is the reserved minus allocated (roughly)
    cuda_ctx = reserved - alloc
    print(f"  CUDA context overhead: ~{cuda_ctx:.1f}MB")
    print(f"  Per-container GPU: ~{reserved:.1f}MB (context + models + inference)")
    print(f"  10 containers estimate: ~{reserved * 10:.0f}MB")
    if reserved * 10 > 1600:
        print(f"  ⚠ EXCEEDS 1.6GB limit (10% of 16GB)")
    else:
        print(f"  ✓ Within 1.6GB limit")

    # ── Try with different min_face_size to reduce pyramid ──
    print(f"\n  ── Pyramid level sweep ──")
    for min_face in [20, 40, 60, 80, 100]:
        mtcnn_detector.min_face_size = min_face
        t0 = time.time()
        boxes, _ = mtcnn_detector.detect_faces(
            pil_img, min_face,
            mtcnn_detector.thresholds, mtcnn_detector.nms_thresholds,
            mtcnn_detector.factor,
        )
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  min_face={min_face:3d}: {dt*1000:.0f}ms, {len(boxes)} faces")

    del model, mtcnn_detector
    gc.collect()
    torch.cuda.empty_cache()
    mem("after cleanup")


if __name__ == "__main__":
    try:
        bench_mtcnn_edgeface_gpu()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    print("\nDone.")
