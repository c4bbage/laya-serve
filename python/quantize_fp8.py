"""FP8 (e4m3) PTQ via NVIDIA ModelOpt -> TensorRT Q/DQ ONNX.

Result on RTX 4090 (see docs/BENCHMARKS.md): +7-9% throughput, argmax agreement
97.3% -> 89.5%. Kept for reference / other GPUs, not recommended on Ada.

Why these settings (modelopt 0.46 defaults ran >1.5h on CPU and never finished):
  - calibration_method="max": modelopt's fp8 default is "entropy" (per-tensor
    2048-bin histograms over ~400 tensors x ~10M elems in numpy) -> hours.
    Max calibration is an on-graph min/max reduction, seconds per batch.
  - calibration_eps=[cuda:N] only: the default ['cpu','cuda:0','trt'] puts CPU
    first, so ORT placed the graph on CPU (601 Memcpy nodes, GPU util 0%).
  - source is the fp32 export with masks patched (python patch_mask.py laya.onnx
    laya32s.onnx); modelopt casts non-quantized ops to fp16 itself
    (high_precision_dtype). Calibrating from the fp16 graph is what produced
    the -inf/NaN histograms that needed the np.histogram monkeypatch.
  - disable_mha_qdq: keep attention BMMs in fp16, quantize GEMMs only first.
  - scorer / act_head stay out of FP8 (they produce the calibrated probs).

Usage (pick a GPU with free memory; calibration data = laya_parity.jsonl from export_onnx.py):
  CALIB_GPU=0 SRC=laya32s.onnx DST=laya8.onnx python quantize_fp8.py
  Note: modelopt rewrites SRC in place (adds value_info shape annotations).
"""
import json
import os
import time

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader
from modelopt.onnx.quantization import quantize

SRC = os.environ.get("SRC", "laya32s.onnx")
DST = os.environ.get("DST", "laya8.onnx")
PARITY = "laya_parity.jsonl"
N_BATCH = int(os.environ.get("N_BATCH", "256"))
# BS=1 by default: no padding. Padded positions far (>128 tok) from real tokens
# see a fully-masked local-attention window and produce garbage up to inf; max
# calibration then takes that as amax -> scale=inf -> NaN engine output.
BS = int(os.environ.get("BS", "1"))
GPU = os.environ.get("CALIB_GPU", "1")

# one question per fixture, spread over the whole file (consecutive items are
# questions of the same state, which would make each calibration batch redundant)
items = []
for line in open(PARITY, encoding="utf-8"):
    e = json.loads(line)
    qs = list(e["questions"].values())
    q = qs[len(items) % len(qs)]
    items.append((q["ids"], q["markers"], q["qtype"]))
rng = np.random.default_rng(0)
rng.shuffle(items)
print("calibration pool:", len(items), "using", N_BATCH * BS)


def make_batches(n_batches, bs):
    out = []
    for b in range(n_batches):
        chunk = items[b * bs:(b + 1) * bs]
        if not chunk:
            break
        n = len(chunk)
        L = max(len(a) for a, _, _ in chunk)
        K = max(len(m) for _, m, _ in chunk)
        ids = np.zeros((n, L), np.int64)
        att = np.zeros((n, L), np.int64)
        mpos = np.zeros((n, K), np.int64)
        mmask = np.zeros((n, K), np.int64)
        qt = np.zeros((n,), np.int64)
        for r, (a, m, t) in enumerate(chunk):
            ids[r, :len(a)] = a
            att[r, :len(a)] = 1
            mpos[r, :len(m)] = m
            mmask[r, :len(m)] = 1
            qt[r] = t
        out.append({
            "input_ids": ids, "attention_mask": att,
            "marker_pos": mpos, "marker_mask": mmask, "qtype": qt,
        })
    return out


class Reader(CalibrationDataReader):
    def __init__(self):
        self.rewind()

    def get_next(self):
        if self.i >= len(self.data):
            return None
        d = self.data[self.i]
        self.i += 1
        if self.i % 8 == 0:
            print(f"  calib batch {self.i}/{len(self.data)}", flush=True)
        return d

    def get_first(self):
        return self.data[0]

    def rewind(self):
        self.data = make_batches(N_BATCH, BS)
        self.i = 0


# keep the decision heads (scorer + act_head) in high precision. The 2-layer
# DynHeadLayer head (/m/layers.*) is excluded too: TRT 10.13 Myelin fails on its
# FP8 per-channel weight DQ ("No matching rules found for input operand types").
g = onnx.load(SRC, load_external_data=False).graph
exclude = [n.name for n in g.node
           if n.op_type in ("MatMul", "Gemm")
           and ("scorer" in n.name or "act_head" in n.name or n.name.startswith("/m/layers.")
                or "rotary_emb" in n.name)]  # RoPE angle = pos x inv_freq: 3-bit mantissa wrecks it
# mmBERT has massive activations on the CLS token (tok0): mlp Mul_2 ~3.6e4 at
# ch924 from layer 11 on (fp32, measured on 256 real samples; typical values
# 0.1-100). A per-tensor FP8 scale sized for 3.6e4 flushes everything else to 0,
# so the Wo MatMuls fed by those tensors stay fp16. Layer list via EXCLUDE_WO.
wo_layers = os.environ.get("EXCLUDE_WO", "0-21")
lo, hi = (int(x) for x in wo_layers.split("-"))
exclude += [n.name for n in g.node
            if n.op_type == "MatMul" and "/mlp/Wo/" in n.name
            and lo <= int(n.name.split("/layers.")[1].split("/")[0]) <= hi]
print("excluded heads:", exclude)

t0 = time.time()
quantize(
    onnx_path=SRC,
    quantize_mode="fp8",
    # GEMMs only: by default modelopt also quantizes residual Adds, whose tensors
    # carry the ~1.4e4 CLS outlier -> the whole residual stream collapses to 0.
    op_types_to_quantize=["MatMul", "Gemm"],
    calibration_method="max",
    calibration_data_reader=Reader(),
    calibration_eps=[f"cuda:{GPU}"],
    high_precision_dtype="fp16",
    disable_mha_qdq=os.environ.get("MHA_QDQ", "0") != "1",
    nodes_to_exclude=exclude or None,
    output_path=DST,
    # GEMV probe runs the whole graph with every MatMul output kept alive -> OOMs
    # on a shared 24GB card; all GEMMs here are (B*S)xD, the only small-M ones
    # (act_head) are already excluded above.
    enable_gemv_detection_for_trt=False,
)
print(f"fp8 onnx saved: {os.path.getsize(DST)} bytes in {time.time() - t0:.0f}s")
