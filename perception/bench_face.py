#!/usr/bin/env python3
"""Benchmark MTCNN + EdgeFace on Jetson Orin — GPU vs CPU, VRAM & speed.

Usage:
    cd /home/develop/dengshiwei/phanthymotus-face/perception
    python3 /home/develop/dengshiwei/bench_face.py [test_image] [device]

    test_image: path to jpg (default: ../gpt_日系校园穿搭sexiestadjusted7.jpg)
    device: gpu | cpu | both (default: both)
"""

import os, sys, time, gc, traceback

# ── Fix numpy 2.x vs torch issue ──
os.environ["PYTHONWARNINGS"] = "ignore"

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGEFACE_SRC = os.path.join(SCRIPT_DIR, "edgeface_src")
sys.path.insert(0, EDGEFACE_SRC)

# ── Suppress numpy warnings ──
import warnings
warnings.filterwarnings("ignore")

# ── Install timm if missing ──
try:
    import timm
except ImportError:
    print("[setup] installing timm...")
    os.system(f"{sys.executable} -m pip install -q --no-deps timm 2>&1")
    os.system(f"{sys.executable} -m pip install -q safetensors huggingface-hub 2>&1")
    import timm

import numpy as np
import torch
from PIL import Image

# ── Test image ──
test_image_path = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(SCRIPT_DIR, "gpt_日系校园穿搭sexiestadjusted7.jpg")
device_arg = sys.argv[2] if len(sys.argv) > 2 else "both"

print(f"[bench] test_image: {test_image_path}")
print(f"[bench] edgeface_src: {EDGEFACE_SRC}")
print(f"[bench] torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
print(f"[bench] numpy: {np.__version__}")
print(f"[bench] timm: {timm.__version__}")
print()

# ── Memory measurement ──
def get_gpu_memory():
    """Get GPU memory in MB on Jetson (unified memory)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                    used = total - avail
                    return used, total
    except:
        pass
    return 0, 0

def get_process_rss():
    """Get process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except:
        pass
    return 0

def measure_memory(label):
    rss = get_process_rss()
    used, total = get_gpu_memory()
    print(f"  [mem] {label}: RSS={rss}MB, system used={used}MB / {total}MB")
    return rss, used, total

# ── Benchmark ──
def bench_device(device_str):
    print(f"\n{'='*60}")
    print(f"[bench] device={device_str}")
    print(f"{'='*60}")

    torch_device = torch.device(device_str)
    torch.set_num_threads(1)

    rss_before, used_before, total = measure_memory("before import")

    # ── Load EdgeFace ──
    from backbones import get_model
    model_name = "edgeface_s_gamma_05"
    model = get_model(model_name)

    ckpt_path = os.path.join(EDGEFACE_SRC, f"{model_name}.pt")
    state_dict = torch.load(ckpt_path, map_location=torch_device)
    model.load_state_dict(state_dict)
    model.to(torch_device).eval()

    rss_after_model, used_after_model, _ = measure_memory("after EdgeFace load")

    # ── Load MTCNN ──
    from face_alignment import mtcnn
    mtcnn_device = "cuda:0" if device_str == "cuda" else "cpu"
    detector = mtcnn.MTCNN(device=mtcnn_device, crop_size=(112, 112))

    rss_after_mtcnn, used_after_mtcnn, _ = measure_memory("after MTCNN load")

    # ── Preprocessing ──
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    from face_alignment.mtcnn_pytorch.src.align_trans import (
        get_reference_facial_points, warp_and_crop_face,
    )
    reference = get_reference_facial_points(default_square=True)

    # ── Load test image ──
    pil_img = Image.open(test_image_path).convert("RGB")
    img_arr = np.array(pil_img)
    print(f"  [info] image size: {pil_img.size}, mode: {pil_img.mode}")

    # ── Warmup ──
    print("  [warmup] running 1 inference...")
    t0 = time.time()
    boxes, landmarks = detector.detect_faces(
        pil_img, detector.min_face_size, detector.thresholds,
        detector.nms_thresholds, detector.factor,
    )
    t1 = time.time()
    print(f"  [warmup] MTCNN detect: {t1-t0:.3f}s, found {len(boxes)} faces")

    if len(boxes) > 0:
        for box, lm in zip(boxes, landmarks):
            facial5points = [[lm[j], lm[j + 5]] for j in range(5)]
            warped = warp_and_crop_face(img_arr, facial5points, reference, crop_size=(112, 112))
            aligned = Image.fromarray(warped)
            tensor = transform(aligned).unsqueeze(0).to(torch_device)
            with torch.no_grad():
                emb = model(tensor)
            emb = emb.cpu().numpy().flatten()
        t2 = time.time()
        print(f"  [warmup] EdgeFace embed: {t2-t1:.3f}s, emb dim={emb.shape[0]}")

    rss_after_warmup, used_after_warmup, _ = measure_memory("after warmup")

    # ── Benchmark 5 iterations ──
    print("  [bench] running 5 iterations...")
    detect_times = []
    embed_times = []
    total_times = []

    for i in range(5):
        t_start = time.time()

        t0 = time.time()
        boxes, landmarks = detector.detect_faces(
            pil_img, detector.min_face_size, detector.thresholds,
            detector.nms_thresholds, detector.factor,
        )
        t1 = time.time()
        detect_times.append(t1 - t0)

        if len(boxes) > 0:
            for box, lm in zip(boxes, landmarks):
                facial5points = [[lm[j], lm[j + 5]] for j in range(5)]
                warped = warp_and_crop_face(img_arr, facial5points, reference, crop_size=(112, 112))
                aligned = Image.fromarray(warped)
                tensor = transform(aligned).unsqueeze(0).to(torch_device)
                with torch.no_grad():
                    emb = model(tensor)
                emb = emb.cpu().numpy().flatten()
        t2 = time.time()
        embed_times.append(t2 - t1)
        total_times.append(t2 - t_start)

        print(f"  [iter {i}] detect={t1-t0:.3f}s embed={t2-t1:.3f}s total={t2-t_start:.3f}s faces={len(boxes)}")

    rss_final, used_final, _ = measure_memory("after 5 iterations")

    # ── Summary ──
    avg_detect = sum(detect_times) / len(detect_times)
    avg_embed = sum(embed_times) / len(embed_times)
    avg_total = sum(total_times) / len(total_times)

    model_mem = used_after_model - used_before
    mtcnn_mem = used_after_mtcnn - used_after_model
    total_model_mem = used_after_mtcnn - used_before

    print(f"\n  {'─'*50}")
    print(f"  [SUMMARY] device={device_str}")
    print(f"  {'─'*50}")
    print(f"  MTCNN detect avg:  {avg_detect*1000:.0f}ms")
    print(f"  EdgeFace embed avg: {avg_embed*1000:.0f}ms")
    print(f"  Total per-image avg: {avg_total*1000:.0f}ms ({1/avg_total:.1f} fps)")
    print(f"  Model load memory: {total_model_mem}MB (EdgeFace={model_mem}MB, MTCNN={mtcnn_mem}MB)")
    print(f"  Final system memory: {used_final}MB / {total}MB")
    print(f"  Process RSS: {rss_final}MB")
    print(f"  Per-container est (x10): {used_final + (used_final-used_before)*9}MB (model mem only)")

    # ── Cleanup ──
    del model, detector
    gc.collect()
    if device_str == "cuda":
        torch.cuda.empty_cache()

    return {
        "device": device_str,
        "detect_ms": avg_detect * 1000,
        "embed_ms": avg_embed * 1000,
        "total_ms": avg_total * 1000,
        "fps": 1 / avg_total,
        "model_mem_mb": total_model_mem,
        "final_rss_mb": rss_final,
        "final_sys_used_mb": used_final,
        "total_sys_mb": total,
    }


results = []
if device_arg in ("gpu", "both"):
    try:
        results.append(bench_device("cuda"))
    except Exception as e:
        print(f"  [ERROR] GPU: {e}")
        traceback.print_exc()
    # cleanup between runs
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

if device_arg in ("cpu", "both"):
    try:
        results.append(bench_device("cpu"))
    except Exception as e:
        print(f"  [ERROR] CPU: {e}")
        traceback.print_exc()

print(f"\n{'='*60}")
print("FINAL COMPARISON")
print(f"{'='*60}")
print(f"{'Device':<8} {'Detect':>8} {'Embed':>8} {'Total':>8} {'FPS':>6} {'ModelMem':>10} {'RSS':>6} {'SysUsed':>8}")
print(f"{'─'*60}")
for r in results:
    print(f"{r['device']:<8} {r['detect_ms']:7.0f}ms {r['embed_ms']:7.0f}ms {r['total_ms']:7.0f}ms {r['fps']:5.1f} {r['model_mem_mb']:8d}MB {r['final_rss_mb']:5d}MB {r['final_sys_used_mb']:7d}MB")
