#!/usr/bin/env python3
"""Benchmark all approaches on Jetson Orin — YuNet+EdgeFace CPU, ONNX Runtime, MPS.

Usage:
    cd /home/develop/dengshiwei
    python3 bench_all.py [test_image]
"""

import os, sys, time, gc, traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGEFACE_SRC = os.path.join(SCRIPT_DIR, "phanthymotus-face", "perception", "edgeface_src")
sys.path.insert(0, EDGEFACE_SRC)

import warnings
warnings.filterwarnings("ignore")

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
print(f"[bench] numpy: {np.__version__}, cv2: {cv2.__version__}")
print()

# ── YuNet model ──
yunet_dir = os.path.join(SCRIPT_DIR, "yunet_models")
os.makedirs(yunet_dir, exist_ok=True)
yunet_path = os.path.join(yunet_dir, "face_detection_yunet_2023mar.onnx")
if not os.path.exists(yunet_path):
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    import urllib.request
    urllib.request.urlretrieve(url, yunet_path)

# ── Memory ──
def get_mem():
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
    except: return 0, 0, 0

def mem(label):
    rss, used, total = get_mem()
    print(f"  [mem] {label}: RSS={rss}MB, sys={used}MB/{total}MB")
    return rss, used, total

# ── ArcFace reference ──
REFERENCE = np.array([
    [38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.3655],
], dtype=np.float32)

def align_face(img, landmarks):
    M = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), REFERENCE)[0]
    if M is None: return None
    return cv2.warpAffine(img, M, (112, 112), borderValue=0)

def yunet_detect(detector, img, conf=0.5):
    H, W = img.shape[:2]
    detector.setInputSize((W, H))
    _, faces = detector.detect(img)
    results = []
    if faces is not None:
        for f in faces:
            if f[14] >= conf:
                lm = np.array([
                    [f[4], f[5]], [f[6], f[7]], [f[8], f[9]],
                    [f[10], f[11]], [f[12], f[13]],
                ], dtype=np.float32)
                results.append({
                    "bbox": [float(f[0]), float(f[1]), float(f[0]+f[2]), float(f[1]+f[3])],
                    "confidence": float(f[14]),
                    "landmarks": lm,
                })
    return results


# ════════════════════════════════════════════════════════════════════════════
# 1. YuNet + EdgeFace CPU — full res + resized
# ════════════════════════════════════════════════════════════════════════════
def bench_yunet_cpu():
    print(f"\n{'='*70}")
    print(f"[1] YuNet + EdgeFace (CPU) — multiple resolutions")
    print(f"{'='*70}")

    torch.set_num_threads(1)
    device = torch.device("cpu")

    from backbones import get_model
    model = get_model("edgeface_s_gamma_05")
    state_dict = torch.load(os.path.join(EDGEFACE_SRC, "edgeface_s_gamma_05.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    detector = cv2.FaceDetectorYN_create(yunet_path, "", (320,320), 0.5, 0.3, 5000)

    pil_img = Image.open(test_image_path).convert("RGB")
    img_orig = np.array(pil_img)
    H, W = img_orig.shape[:2]
    print(f"  original: {W}×{H}")

    mem("after model load")

    # Test at different resolutions
    for target_w in [2160, 1280, 960, 640]:
        scale = target_w / W
        new_h, new_w = int(H * scale), target_w
        img = cv2.resize(img_orig, (new_w, new_h))

        # Warmup
        dets = yunet_detect(detector, img, conf=0.5)
        n_faces = 0
        for det in dets:
            aligned = align_face(img, det["landmarks"])
            if aligned is None: continue
            tensor = transform(Image.fromarray(aligned)).unsqueeze(0).to(device)
            with torch.no_grad(): emb = model(tensor)
            n_faces += 1

        # Bench
        times = []
        for i in range(5):
            t0 = time.time()
            dets = yunet_detect(detector, img, conf=0.5)
            t1 = time.time()
            for det in dets:
                aligned = align_face(img, det["landmarks"])
                if aligned is None: continue
                tensor = transform(Image.fromarray(aligned)).unsqueeze(0).to(device)
                with torch.no_grad(): emb = model(tensor)
            t2 = time.time()
            times.append(t2 - t0)

        avg = sum(times) / len(times)
        fps = 1 / avg
        print(f"  {new_w}×{new_h}: detect={sum(t for t in times)/len(times)*1000:.0f}ms total={avg*1000:.0f}ms ({fps:.1f} fps) faces={n_faces}")

    rss, used, total = mem("final")

    del model, detector
    gc.collect()
    return rss, used, total


# ════════════════════════════════════════════════════════════════════════════
# 2. ONNX Runtime — EdgeFace on CUDA (smaller context than PyTorch)
# ════════════════════════════════════════════════════════════════════════════
def bench_onnxruntime():
    print(f"\n{'='*70}")
    print(f"[2] ONNX Runtime — EdgeFace CUDA + YuNet CPU")
    print(f"{'='*70}")

    try:
        import onnxruntime as ort
        print(f"  onnxruntime: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"  available providers: {providers}")
    except ImportError:
        print("  onnxruntime not available, skipping")
        return

    torch.set_num_threads(1)

    # ── Export EdgeFace to ONNX ──
    onnx_path = os.path.join(EDGEFACE_SRC, "edgeface_s_gamma_05.onnx")
    if not os.path.exists(onnx_path):
        print("  exporting EdgeFace to ONNX...")
        from backbones import get_model
        device = torch.device("cpu")
        model = get_model("edgeface_s_gamma_05")
        state_dict = torch.load(os.path.join(EDGEFACE_SRC, "edgeface_s_gamma_05.pt"), map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        dummy = torch.randn(1, 3, 112, 112)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["input"], output_names=["embedding"],
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=13,
        )
        print(f"  ONNX exported: {os.path.getsize(onnx_path)/1e6:.1f}MB")
        del model
        gc.collect()

    mem("before ORT session")

    # ── Try CUDA provider ──
    for provider_set in [
        ["CUDAExecutionProvider"],
        ["CPUExecutionProvider"],
    ]:
        provider_name = provider_set[0]
        if provider_name not in providers:
            print(f"  {provider_name} not available, skipping")
            continue

        print(f"\n  --- Provider: {provider_name} ---")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        try:
            session = ort.InferenceSession(onnx_path, sess_options, providers=provider_set)
        except Exception as e:
            print(f"  failed to create session: {e}")
            continue

        mem(f"after ORT session ({provider_name})")

        # ── YuNet detector (CPU) ──
        detector = cv2.FaceDetectorYN_create(yunet_path, "", (320,320), 0.5, 0.3, 5000)

        # ── Load image ──
        pil_img = Image.open(test_image_path).convert("RGB")
        img_orig = np.array(pil_img)
        H, W = img_orig.shape[:2]

        # Resize to 1280 width
        target_w = 1280
        scale = target_w / W
        new_h, new_w = int(H * scale), target_w
        img = cv2.resize(img_orig, (new_w, new_h))

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

        # Warmup
        dets = yunet_detect(detector, img, conf=0.5)
        for det in dets:
            aligned = align_face(img, det["landmarks"])
            if aligned is None: continue
            tensor = transform(Image.fromarray(aligned)).unsqueeze(0).numpy()
            emb = session.run(["embedding"], {"input": tensor})[0]

        mem(f"after warmup ({provider_name})")

        # Bench
        times = []
        for i in range(5):
            t0 = time.time()
            dets = yunet_detect(detector, img, conf=0.5)
            t1 = time.time()
            for det in dets:
                aligned = align_face(img, det["landmarks"])
                if aligned is None: continue
                tensor = transform(Image.fromarray(aligned)).unsqueeze(0).numpy()
                emb = session.run(["embedding"], {"input": tensor})[0]
            t2 = time.time()
            times.append(t2 - t0)
            print(f"  [iter {i}] detect={t1-t0:.3f}s embed={t2-t1:.3f}s total={t2-t0:.3f}s faces={len(dets)}")

        avg = sum(times) / len(times)
        print(f"  {provider_name}: total={avg*1000:.0f}ms ({1/avg:.1f} fps)")

        mem(f"after bench ({provider_name})")

        del session
        gc.collect()

    # Cleanup
    del detector
    gc.collect()


# ════════════════════════════════════════════════════════════════════════════
# 3. PyTorch CUDA — measure actual CUDA context size
# ════════════════════════════════════════════════════════════════════════════
def bench_pytorch_cuda_context():
    print(f"\n{'='*70}")
    print(f"[3] PyTorch CUDA — measuring CUDA context size")
    print(f"{'='*70}")

    rss0, used0, total = get_mem()
    print(f"  [mem] before torch.cuda: RSS={rss0}MB, sys={used0}MB/{total}MB")

    # Just initialize CUDA context
    torch.cuda.init()
    x = torch.zeros(1).cuda()  # Force context creation

    rss1, used1, _ = get_mem()
    ctx_mb = used1 - used0
    print(f"  [mem] after CUDA init: RSS={rss1}MB, sys={used1}MB/{total}MB")
    print(f"  [mem] CUDA context overhead: {ctx_mb}MB")

    # Load model on CUDA
    from backbones import get_model
    model = get_model("edgeface_s_gamma_05")
    state_dict = torch.load(os.path.join(EDGEFACE_SRC, "edgeface_s_gamma_05.pt"), map_location="cuda")
    model.load_state_dict(state_dict)
    model.to("cuda").eval()

    rss2, used2, _ = get_mem()
    print(f"  [mem] after EdgeFace CUDA: RSS={rss2}MB, sys={used2}MB/{total}MB")
    print(f"  [mem] EdgeFace model overhead: {used2-used1}MB")

    # MTCNN on CUDA
    from face_alignment import mtcnn
    detector = mtcnn.MTCNN(device="cuda:0", crop_size=(112, 112))

    rss3, used3, _ = get_mem()
    print(f"  [mem] after MTCNN CUDA: RSS={rss3}MB, sys={used3}MB/{total}MB")
    print(f"  [mem] MTCNN overhead: {used3-used2}MB")
    print(f"  [mem] Total CUDA overhead: {used3-used0}MB")
    print(f"  [mem] x10 containers estimate: {used3-used0+10*0}MB model only, but CUDA ctx = {ctx_mb}MB each")
    print(f"  [mem] x10 total estimate: {used0 + ctx_mb*10 + (used3-used0-ctx_mb)*10}MB (if no sharing)")
    print(f"  [mem] x10 total estimate: {used0 + ctx_mb + (used3-used0-ctx_mb)*10}MB (if MPS sharing)")

    del model, detector, x
    gc.collect()
    torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STARTING BENCHMARKS")
print("="*70)

# 1. YuNet + EdgeFace CPU
try:
    bench_yunet_cpu()
except Exception as e:
    print(f"  [ERROR] YuNet CPU: {e}")
    traceback.print_exc()

gc.collect()
time.sleep(2)

# 2. ONNX Runtime
try:
    bench_onnxruntime()
except Exception as e:
    print(f"  [ERROR] ONNX Runtime: {e}")
    traceback.print_exc()

gc.collect()
time.sleep(2)

# 3. PyTorch CUDA context measurement
try:
    bench_pytorch_cuda_context()
except Exception as e:
    print(f"  [ERROR] PyTorch CUDA: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("ALL BENCHMARKS COMPLETE")
print("="*70)
