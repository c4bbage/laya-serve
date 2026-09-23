"""Replace huge-negative attention-mask constants (<= -60000) with -10000.

Usage: python patch_mask.py laya.onnx laya32s.onnx

finfo(dtype).min (-65504 in fp16, -3.4e38 in fp32) plus negative scores overflows to
-inf inside Add(score, mask) once TensorRT runs the graph in fp16. -inf is
softmax-harmless (exp(-inf)=0) but breaks ModelOpt calibration histograms and is
TRT-unfriendly. -10000 keeps the exp() underflow semantics.
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper

SRC = sys.argv[1] if len(sys.argv) > 1 else "laya.onnx"
DST = sys.argv[2] if len(sys.argv) > 2 else "laya32s.onnx"
NEW = -10000.0

m = onnx.load(SRC)
g = m.graph
patched = 0

for init in g.initializer:
    arr = numpy_helper.to_array(init)
    if np.issubdtype(arr.dtype, np.floating) and (arr <= -60000).any():
        arr = np.where(arr <= -60000, np.float32(NEW).astype(arr.dtype), arr)
        new_init = numpy_helper.from_array(arr, init.name)
        init.CopyFrom(new_init)
        patched += 1

for node in g.node:
    if node.op_type != "Constant":
        continue
    for attr in node.attribute:
        if attr.name != "value" or attr.t is None:
            continue
        arr = numpy_helper.to_array(attr.t)
        if np.issubdtype(arr.dtype, np.floating) and (arr <= -60000).any():
            arr = np.where(arr <= -60000, np.float32(NEW).astype(arr.dtype), arr)
            attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
            patched += 1

onnx.save(m, DST)
print("patched tensors:", patched, "->", DST)
