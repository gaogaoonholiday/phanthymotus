#!/usr/bin/env python3
"""Export EdgeFace locally, validate FP32, and optionally benchmark Linear INT8.

Example:
    python export_edgeface_onnx.py --checkpoint /path/edgeface_s_gamma_05.pt \
        --output-dir /tmp/edgeface-export --quantize-linear
No artifacts are published or copied outside --output-dir.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np


def artifact_size(path):
    """Count the protobuf and every referenced external tensor file once."""
    import onnx
    from onnx.external_data_helper import _get_all_tensors

    path = Path(path).resolve()
    model = onnx.load(str(path), load_external_data=False)
    files = {path}
    for tensor in _get_all_tensors(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            info = {entry.key: entry.value for entry in tensor.external_data}
            files.add((path.parent / info['location']).resolve())
    sizes = {str(file): file.stat().st_size for file in sorted(files)}
    return {"bytes": sum(sizes.values()), "files": sizes}


def session(path, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def difference(reference, candidate):
    a, b = reference.astype(np.float64).ravel(), candidate.astype(np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite embedding")
    return {
        "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        "max_error": float(np.max(np.abs(a - b))),
    }


def benchmark(path, inputs, threads, warmup, runs):
    sess = session(path, threads)
    for i in range(warmup):
        sess.run(None, {"input": inputs[i % len(inputs)]})
    times = []
    for i in range(runs):
        start = time.perf_counter()
        sess.run(None, {"input": inputs[i % len(inputs)]})
        times.append((time.perf_counter() - start) * 1000)
    return {"threads": threads, "inter_op_threads": 1, "batch": 1,
            "warmup": warmup, "runs": runs, "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="edgeface_s_gamma_05")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantize-linear", action="store_true")
    parser.add_argument("--detector", type=Path, help="Include detector artifacts in the size budget")
    parser.add_argument("--budget-bytes", type=int, default=30_000_000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args(argv)
    if args.samples < 2 or args.warmup < 1 or args.runs < 1:
        parser.error("samples >= 2, warmup >= 1 and runs >= 1 are required")
    if Path(args.model).name != args.model or args.model in (".", ".."):
        parser.error("model must be a model name, not a path")

    import onnx
    import onnxruntime as ort
    import torch
    import timm

    source = Path(__file__).resolve().parent / "edgeface_src"
    sys.path.insert(0, str(source))
    from backbones import get_model

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    # Use the explicit attention graph for opset 13 export.
    timm.layers.set_fused_attn(False)
    model = get_model(args.model).cpu().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    linear_elements = sum(m.weight.numel() for m in model.modules()
                          if isinstance(m, torch.nn.Linear))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32 = args.output_dir / f"{args.model}.onnx"
    int8 = args.output_dir / f"{args.model}.int8.onnx"
    report_path = args.output_dir / f"{args.model}.report.json"
    for output in (fp32, int8, report_path):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}; choose a fresh output directory")
    rng = np.random.default_rng(args.seed)
    inputs = [rng.standard_normal((1, 3, 112, 112)).astype(np.float32)
              for _ in range(args.samples)]
    report = {
        "model": args.model, "checkpoint_bytes": args.checkpoint.stat().st_size,
        "parameter_elements": sum(p.numel() for p in model.parameters()),
        "linear_weight_elements": linear_elements, "seed": args.seed,
        "input_kind": "synthetic standard normal; not recognition accuracy",
        "versions": {"torch": torch.__version__, "timm": timm.__version__,
                     "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    print("Exporting FP32", flush=True)
    torch.onnx.export(
        model, torch.from_numpy(inputs[0]), str(fp32),
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=13, dynamo=False,
    )
    onnx.checker.check_model(str(fp32))
    fp_session = session(fp32, 2)
    fp_outputs = []
    report["pytorch_vs_fp32"] = []
    with torch.inference_mode():
        for data in inputs:
            reference = model(torch.from_numpy(data)).numpy()
            output = fp_session.run(None, {"input": data})[0]
            np.testing.assert_allclose(output, reference, rtol=1e-3, atol=1e-4)
            report["pytorch_vs_fp32"].append(difference(reference, output))
            fp_outputs.append(output)
    report["fp32_allclose"] = {"passed": True, "rtol": 1e-3, "atol": 1e-4}
    report["fp32_artifact"] = artifact_size(fp32)
    print("FP32 allclose passed", flush=True)
    del fp_session

    if args.quantize_linear:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        graph = onnx.load(str(fp32))
        weights = {tensor.name: tensor for tensor in graph.graph.initializer}
        targets = [node for node in graph.graph.node
                   if node.op_type in ("MatMul", "Gemm") and len(node.input) > 1
                   and node.input[1] in weights and len(weights[node.input[1]].dims) == 2]
        target_elements = sum(int(np.prod(weights[name].dims))
                              for name in {node.input[1] for node in targets})
        if not targets or target_elements != linear_elements:
            raise RuntimeError(f"Linear coverage mismatch: ONNX {target_elements}, torch {linear_elements}")
        quantize_dynamic(
            str(fp32), str(int8), op_types_to_quantize=["MatMul", "Gemm"],
            # ORT's dynamic quantizer rewrites Gemm to <name>_MatMul first.
            nodes_to_quantize=[name for node in targets for name in
                               ([node.name, node.name + "_MatMul"]
                                if node.op_type == "Gemm" else [node.name])],
            weight_type=QuantType.QInt8, per_channel=True, use_external_data_format=False,
            extra_options={"MatMulConstBOnly": True},
        )
        quantized = onnx.load(str(int8))
        onnx.checker.check_model(quantized)
        counts = Counter(node.op_type for node in quantized.graph.node)
        qweights = {tensor.name: tensor for tensor in quantized.graph.initializer}
        integer_nodes = [node for node in quantized.graph.node if node.op_type == "MatMulInteger"]
        actual_elements = sum(int(np.prod(qweights[name].dims)) for name in
                              {node.input[1] for node in integer_nodes}
                              if name in qweights and qweights[name].data_type == onnx.TensorProto.INT8)
        if len(integer_nodes) != len(targets) or actual_elements != linear_elements:
            raise RuntimeError("Not all Linear weights were quantized")
        report["quantization"] = {"target_nodes": len(targets), "int8_weight_elements": actual_elements,
                                  "node_counts": dict(sorted(counts.items()))}
        report["int8_artifact"] = artifact_size(int8)
        if len(report["int8_artifact"]["files"]) != 1:
            raise RuntimeError("INT8 must be a single ONNX file")
        int_session = session(int8, 2)
        report["fp32_vs_int8"] = [difference(reference, int_session.run(None, {"input": data})[0])
                                  for data, reference in zip(inputs, fp_outputs)]
        del int_session
        print("INT8 CPU inference and Linear coverage passed", flush=True)

    selected = int8 if args.quantize_linear else fp32
    if args.detector:
        report["detector_artifact"] = artifact_size(args.detector)
        total = artifact_size(selected)["bytes"] + report["detector_artifact"]["bytes"]
        report["size_budget"] = {"total_bytes": total, "limit_bytes": args.budget_bytes,
                                 "passed": total < args.budget_bytes}
    report["latency"] = {}
    for path in ([fp32, int8] if args.quantize_linear else [fp32]):
        report["latency"][path.name] = [benchmark(path, inputs, threads, args.warmup, args.runs)
                                       for threads in (2, 1)]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Report: {report_path.resolve()}", flush=True)
    if args.detector and not report["size_budget"]["passed"]:
        raise RuntimeError("Combined artifact size must be strictly below the budget")


if __name__ == "__main__":
    main()
