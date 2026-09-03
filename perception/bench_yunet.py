#!/usr/bin/env python3
"""Benchmark YuNet + EdgeFace on Jetson Orin — CPU only.

YuNet is OpenCV's built-in face detector (cv2.FaceDetectorYN).
Single-stage, no image pyramid, 5-15ms on CPU.

Usage:
    cd /home/develop/dengshiwei
    python3 bench_yunet.py [test_image]
"""

import os, sys, time, gc, traceback

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGEFACE_SRC = os.path.join(SCRIPT_DIR, "edgeface_src")
sys.path.insert(0, EDGEFACE_SRC)

import warnings
warnings.filterwarnings("ignore")

# ── Install timm if missing ──
try:
    import timm
except ImportError:
    os.system(f"{sys.executable} -m pip install -q --no-deps timm 2>&1")
    os.system(f"{sys.executable} -m pip install -q safetensors huggingface-hub 2>&1")
    import timm

import numpy as np
import torch
from PIL import Image
import cv2

test_image_path = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(SCRIPT_DIR, "gpt_日系校园穿搭sexiestadjusted7.jpg")

print(f"[bench] test_image: {test_image_path}")
print(f"[bench] torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
print(f"[bench] numpy: {np.__version__}")
print(f"[bench] cv2: {cv2.__version__}")
print()

# ── Download YuNet model if not present ──
yunet_dir = os.path.join(SCRIPT_DIR, "yunet_models")
os.makedirs(yunet_dir, exist_ok=True)
yunet_path = os.path.join(yunet_dir, "face_detection_yunet_2023mar.onnx")

if not os.path.exists(yunet_path):
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    print(f"[setup] downloading YuNet model...")
    import urllib.request
    urllib.request.urlretrieve(url, yunet_path)
    print(f"[setup] YuNet downloaded: {os.path.getsize(yunet_path)/1e6:.1f}MB")

# ── Memory measurement ──
def get_gpu_memory():
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


def bench_yunet_edgeface():
    print(f"\n{'='*60}")
    print(f"[bench] YuNet + EdgeFace (CPU only)")
    print(f"{'='*60}")

    torch.set_num_threads(1)
    torch_device = torch.device("cpu")

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

    # ── Load YuNet ──
    detector = cv2.FaceDetectorYN_create(
        yunet_path, "", (320, 320), 0.9, 0.3, 5000
    )

    rss_after_yunet, used_after_yunet, _ = measure_memory("after YuNet load")

    # ── Preprocessing ──
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ── ArcFace alignment reference (same as MTCNN uses) ──
    # Standard 5-point reference for 112x112
    reference = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.6963],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.3655],
    ], dtype=np.float32)

    # ── Load test image ──
    pil_img = Image.open(test_image_path).convert("RGB")
    img_arr = np.array(pil_img)
    H, W = img_arr.shape[:2]
    print(f"  [info] image size: {pil_img.size} (W×H), array shape: {img_arr.shape}")

    # ── Face alignment using 5 landmarks (similar to MTCNN's warp_and_crop_face) ──
    def align_face(img, landmarks, ref_pts, crop_size=(112, 112)):
        """Align face using 5 landmarks via affine transform."""
        src_pts = landmarks.astype(np.float32)
        # Compute affine transform
        M = cv2.estimateAffinePartial2D(src_pts, ref_pts)[0]
        if M is None:
            return None
        aligned = cv2.warpAffine(img, M, crop_size, borderValue=0)
        return aligned

    # ── Warmup ──
    print("  [warmup] running 1 inference...")
    detector.setInputSize((W, H))
    t0 = time.time()
    retval, faces = detector.detect(img_arr)
    t1 = time.time()
    print(f"  [warmup] YuNet detect: {t1-t0:.3f}s, found {len(faces) if faces is not None else 0} faces")

    n_faces = 0
    if faces is not None and len(faces) > 0:
        for face in faces:
            # YuNet output: [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, re_mouth_x, re_mouth_y, le_mouth_x, le_mouth_y, score]
            landmarks = np.array([
                [face[4], face[5]],    # right eye
                [face[6], face[7]],    # left eye
                [face[8], face[9]],    # nose
                [face[10], face[11]],  # right mouth
                [face[12], face[13]],  # left mouth
            ], dtype=np.float32)
            aligned = align_face(img_arr, landmarks, reference)
            if aligned is None:
                continue
            aligned_pil = Image.fromarray(aligned)
            tensor = transform(aligned_pil).unsqueeze(0).to(torch_device)
            with torch.no_grad():
                emb = model(tensor)
            emb = emb.cpu().numpy().flatten()
            n_faces += 1
        t2 = time.time()
        print(f"  [warmup] EdgeFace embed: {t2-t1:.3f}s, faces={n_faces}, emb dim={emb.shape[0]}")

    rss_after_warmup, used_after_warmup, _ = measure_memory("after warmup")

    # ── Benchmark 5 iterations ──
    print("  [bench] running 5 iterations...")
    detect_times = []
    embed_times = []
    total_times = []

    for i in range(5):
        t_start = time.time()

        t0 = time.time()
        detector.setInputSize((W, H))
        retval, faces = detector.detect(img_arr)
        t1 = time.time()
        detect_times.append(t1 - t0)

        n_faces = 0
        if faces is not None and len(faces) > 0:
            for face in faces:
                landmarks = np.array([
                    [face[4], face[5]], [face[6], face[7]], [face[8], face[9]],
                    [face[10], face[11]], [face[12], face[13]],
                ], dtype=np.float32)
                aligned = align_face(img_arr, landmarks, reference)
                if aligned is None:
                    continue
                aligned_pil = Image.fromarray(aligned)
                tensor = transform(aligned_pil).unsqueeze(0).to(torch_device)
                with torch.no_grad():
                    emb = model(tensor)
                emb = emb.cpu().numpy().flatten()
                n_faces += 1
        t2 = time.time()
        embed_times.append(t2 - t1)
        total_times.append(t2 - t_start)

        print(f"  [iter {i}] detect={t1-t0:.3f}s embed={t2-t1:.3f}s total={t2-t_start:.3f}s faces={n_faces}")

    rss_final, used_final, _ = measure_memory("after 5 iterations")

    # ── Summary ──
    avg_detect = sum(detect_times) / len(detect_times)
    avg_embed = sum(embed_times) / len(embed_times)
    avg_total = sum(total_times) / len(total_times)

    model_mem = used_after_yunet - used_before

    print(f"\n  {'─'*50}")
    print(f"  [SUMMARY] YuNet + EdgeFace CPU")
    print(f"  {'─'*50}")
    print(f"  YuNet detect avg:   {avg_detect*1000:.0f}ms")
    print(f"  EdgeFace embed avg: {avg_embed*1000:.0f}ms")
    print(f"  Total per-image avg: {avg_total*1000:.0f}ms ({1/avg_total:.1f} fps)")
    print(f"  Model load memory: {model_mem}MB")
    print(f"  Final system memory: {used_final}MB / {total}MB")
    print(f"  Process RSS: {rss_final}MB")
    print(f"  Per-container est (x10): {used_final + model_mem*9}MB")

    del model, detector
    gc.collect()

    return {
        "device": "cpu",
        "detect_ms": avg_detect * 1000,
        "embed_ms": avg_embed * 1000,
        "total_ms": avg_total * 1000,
        "fps": 1 / avg_total,
        "model_mem_mb": model_mem,
        "final_rss_mb": rss_final,
        "final_sys_used_mb": used_final,
        "total_sys_mb": total,
    }


# Also test with different image sizes
def bench_resized():
    """Test YuNet + EdgeFace with resized image (1080p, 720p)."""
    print(f"\n{'='*60}")
    print(f"[bench] YuNet + EdgeFace — resized images")
    print(f"{'='*60}")

    torch.set_num_threads(1)
    torch_device = torch.device("cpu")

    from backbones import get_model
    model = get_model("edgeface_s_gamma_05")
    ckpt_path = os.path.join(EDGEFACE_SRC, "edgeface_s_gamma_05.pt")
    state_dict = torch.load(ckpt_path, map_location=torch_device)
    model.load_state_dict(state_dict)
    model.to(torch_device).eval()

    detector = cv2.FaceDetectorYN_create(
        yunet_path, "", (320, 320), 0.9, 0.3, 5000
    )

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    reference = np.array([
        [38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.3655],
    ], dtype=np.float32)

    pil_img = Image.open(test_image_path).convert("RGB")
    img_arr_orig = np.array(pil_img)

    for target_w in [1920, 1280, 640]:
        h, w = img_arr_orig.shape[:2]
        scale = target_w / w
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img_arr_orig, (new_w, new_h))

        # Warmup
        detector.setInputSize((new_w, new_h))
        retval, faces = detector.detect(img_resized)

        # Bench
        times = []
        for i in range(5):
            t0 = time.time()
            detector.setInputSize((new_w, new_h))
            retval, faces = detector.detect(img_resized)
            t1 = time.time()

            if faces is not None and len(faces) > 0:
                for face in faces:
                    landmarks = np.array([
                        [face[4], face[5]], [face[6], face[7]], [face[8], face[9]],
                        [face[10], face[11]], [face[12], face[13]],
                    ], dtype=np.float32)
                    src_pts = landmarks.astype(np.float32)
                    M = cv2.estimateAffinePartial2D(src_pts, reference)[0]
                    if M is not None:
                        aligned = cv2.warpAffine(img_resized, M, (112, 112), borderValue=0)
                        tensor = transform(Image.fromarray(aligned)).unsqueeze(0).to(torch_device)
                        with torch.no_grad():
                            emb = model(tensor)

            t2 = time.time()
            times.append(t2 - t0)

        avg = sum(times) / len(times)
        n_faces = len(faces) if faces is not None else 0
        print(f"  {new_w}×{new_h}: detect+embed avg={avg*1000:.0f}ms ({1/avg:.1f} fps), faces={n_faces}")


results = []
try:
    results.append(bench_yunet_edgeface())
except Exception as e:
    print(f"  [ERROR] {e}")
    traceback.print_exc()

try:
    bench_resized()
except Exception as e:
    print(f"  [ERROR] resized: {e}")
    traceback.print_exc()

print(f"\n{'='*60}")
print("FINAL COMPARISON")
print(f"{'='*60}")
print(f"{'Method':<25} {'Detect':>8} {'Embed':>8} {'Total':>8} {'FPS':>6} {'RSS':>6}")
print(f"{'─'*60}")
for r in results:
    print(f"{'YuNet+EdgeFace CPU':<25} {r['detect_ms']:7.0f}ms {r['embed_ms']:7.0f}ms {r['total_ms']:7.0f}ms {r['fps']:5.1f} {r['final_rss_mb']:5d}MB")
print(f"{'MTCNN+EdgeFace GPU':<25} {'2081ms':>8} {'26ms':>8} {'2107ms':>8} {'0.5':>6} {'1183MB':>6}")
print(f"{'MTCNN+EdgeFace CPU':<25} {'6628ms':>8} {'189ms':>8} {'6818ms':>8} {'0.1':>6} {'1226MB':>6}")
