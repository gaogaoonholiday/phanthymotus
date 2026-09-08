"""ROS-free contracts for the merged SCRFD-2.5G detector work.

The w600k_mbf/sface recognizer candidates are still in stash@{0}; tests for
_ONNX_ARCFACE_REF/_similarity_transform live with that work.
"""
import ast
import json
import logging
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.request

import numpy as np


def load_face():
    path = Path(__file__).resolve().parents[1] / "plugins" / "face.py"
    tree = ast.parse(path.read_text())
    # Extract production definitions without importing ROS or EdgeFace dependencies.
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and getattr(node, "name", "") != "_FaceNode"]
    nodes += [node for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id in {
                  "_ARCFACE_REF", "_SCRFD_STRIDES", "_SCRFD_NUM_ANCHORS",
                  "_DETECT_TARGET_W",
              } for t in node.targets)]
    ns = dict(np=np, os=os, log=logging.getLogger(__name__), urllib=urllib,
              DEFAULT_MODEL_NAME="edgeface_s_gamma_05", _MODEL_BASE_URL="https://example.invalid")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + sorted(nodes, key=lambda n: n.lineno), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


class FaceContracts(unittest.TestCase):
    def setUp(self):
        self.ns = load_face()

    def test_detector_files(self):
        self.assertEqual(
            self.ns["_ensure_weights"] and True, True)  # module loads
        src = (Path(__file__).resolve().parents[1] / "plugins" / "face.py").read_text()
        for needle in ("scrfd_2.5g_bnkps_hsuyabc.onnx", '"scrfd_2.5g"'):
            self.assertIn(needle, src)

    def test_bbox_json(self):
        detector = self.ns["SCRFDDetector"].__new__(self.ns["SCRFDDetector"])
        detector._confidence = .5
        detector._forward = lambda *a: (
            [np.array([[.9]], np.float32)],
            [np.array([[1, 2, 30, 40]], np.float32)],
            [np.zeros((1, 5, 2), np.float32)],
        )
        result = detector.detect(np.zeros((640, 640, 3), np.uint8))[0]
        # bbox must be plain floats — np.float32 is not JSON serializable
        self.assertTrue(all(type(v) is float for v in result["bbox"]))
        json.dumps({"bbox_relative": [round(v / 640, 6) for v in result["bbox"]]})

    def test_forward_reshape_contract(self):
        """2.5G outputs carry a leading batch dim / named outputs; _forward must
        normalize to 2D and index scores[:, 0]."""
        import types

        detector = self.ns["SCRFDDetector"].__new__(self.ns["SCRFDDetector"])
        detector._confidence = .5
        detector._cache = {}
        detector._input_name = "input"
        # 640x640 input: strides 8/16/32 → 80x80/40x40/20x20 grids, 2 anchors.
        def mk(h, w):
            return np.zeros((1, h * w * 2, 1), np.float32)
        det_out = np.zeros((1, 80 * 80 * 2, 4), np.float32)
        det_out_mid = np.zeros((1, 40 * 40 * 2, 4), np.float32)
        det_out_lo = np.zeros((1, 20 * 20 * 2, 4), np.float32)
        kps_out = np.zeros((1, 80 * 80 * 2, 10), np.float32)
        kps_out_mid = np.zeros((1, 40 * 40 * 2, 10), np.float32)
        kps_out_lo = np.zeros((1, 20 * 20 * 2, 10), np.float32)
        # 9 outputs: scores(3) + bboxes(3) + kps(3); all scores 1.0 on first cell
        outs = [mk(80, 80), mk(40, 40), mk(20, 20),
                det_out, det_out_mid, det_out_lo,
                kps_out, kps_out_mid, kps_out_lo]
        outs[0][0, 0, 0] = 0.9
        outs[3][0, 0] = [1, 2, 30, 40]
        detector._sess = types.SimpleNamespace(run=lambda _, __: outs)
        s, b, k = detector._forward(np.zeros((640, 640, 3), np.uint8), 0.5)
        self.assertEqual(len(s), 3)
        self.assertEqual(s[0].shape, (1,))
        self.assertEqual(b[0].shape, (1, 4))
        self.assertEqual(k[0].shape, (1, 5, 2))


if __name__ == "__main__":
    unittest.main()
