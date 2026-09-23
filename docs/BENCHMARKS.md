# Benchmarks and engineering notes

All numbers below were measured in September 2026 on one server with 2 × RTX 4090 (24 GB, sm_89). **Both GPUs were shared with other production services**, so absolute numbers are a floor rather than a ceiling, and the same configuration measured at different times varied by about 5%.

- Model: Laya `multilingual` checkpoint (mmBERT-base encoder, 322M parameters), **not fine-tuned**.
- Workload: 2884 real Sichuan-mahjong decision requests (4436 questions, 1.5 questions per request, ~400 tokens per question, 1-14 options). The data is proprietary and not included; `examples/` has synthetic samples in the same format.
- SLO: a concurrency level "passes" if ≥ 99% of requests complete within 600 ms. "Max throughput" is the best passing level.
- Agreement: the share of the 4436 questions whose argmax matches the Python `laya` package on GPU (bf16 autocast). It measures whether a faster stack changes the model's answers, not decision quality.

## 1. Stack evolution

| Stack | 1 GPU | 2 GPUs | c=1 latency | GPU memory / GPU |
|---|---|---|---|---|
| `laya` SDK + FastAPI, fp32 eager, 1 worker | 29 rps | – | 34 ms p50 | ~3 GB |
| same, 4 gunicorn workers | 73 rps | – | – | ~12 GB |
| Go gateway (`layagate`) + 2 Python batch backends / GPU, bf16 | 193 rps | 389 rps | 56 ms | ~6.4 GB |
| `goserve` + ORT CUDA EP, fp16 | – | 336 rps | 27 ms | ~10.5 GB |
| `goserve` + ORT TensorRT EP, fp16 | 402 rps | 774 rps | 16 ms | ~3.2 GB |

Notes:
- With the per-request SDK the GPU was nearly idle (~0% utilization): each request ran its own forward pass and most of the 34 ms was Python-side work. More workers only queued on the same GPU.
- The gateway initially waited for each backend reply before sending the next batch (73 rps); dispatching asynchronously gave 193 rps/GPU at 100% utilization. Three checks confirmed the GPU was saturated: batch cap 48 → 96 (same throughput, double latency), 2 → 3 backends per GPU (same throughput, longer queues), concurrency 32 → 384 (plateau at ~410 rps on 2 GPUs).
- The CUDA EP graph had 0 fused attention nodes and 25 separate Softmax ops: the full attention-score matrix is materialized every layer, which also explains the 10.5 GB arena. TensorRT fuses the graph; memory drops to ~3.2 GB/GPU.
- An earlier TensorRT run, when the other services on the GPUs were lighter, reached 857 rps on 2 GPUs. The 402 / 774 figures come from the final same-period sweep below.

## 2. Precision × concurrency (TensorRT, same period)

One GPU (rps / p95):

| Precision | Agreement | c=1 | c=64 | c=128 | c=192 | c=256 | Max within SLO |
|---|---|---|---|---|---|---|---|
| fp16 | 97.34% | 44 / 16 ms | 386 / 161 ms | 395 / 303 ms | 402 / 457 ms | 408 / 582 ms ✗ | 402 rps |
| bf16 | 97.23% | 45 / 16 ms | 310 / 199 ms | 320 / 379 ms | 318 / 566 ms | 330 / 725 ms ✗ | 318 rps |
| fp32 | 97.50% | 34 / 24 ms | 96 / 670 ms ✗ | 98 ✗ | – | – | 94 rps (c=32) |
| FP8 | 89.52% | 44 / 17 ms | 413 / 153 ms | 421 / 287 ms | 438 / 419 ms | 429 / 566 ms | ~430 rps |

Two GPUs:

| Precision | c=128 | c=256 | c=384 | Max within SLO |
|---|---|---|---|---|
| fp16 | 749 / 243 ms | 774 / 513 ms | 778 / 807 ms ✗ | 774 rps |
| FP8 | 787 / 231 ms | 842 / 457 ms | 854 / 625 ms ✗ | 842 rps |

fp16 and bf16 rows use the RoPE-table export (section 3). Building the fp32 engine needs ~8.5 GB of free GPU memory.

Low-load latency is ~16 ms for every 16-bit and 8-bit variant. With a single request in flight the batcher still waits the full 10 ms window, so most of it is waiting; GPU compute is roughly 5 ms (an estimate, not measured directly). Dispatching immediately when a GPU is idle is the next latency win.

## 3. RoPE precision under TensorRT

The first bf16 engine agreed with Python on only 64.09% of questions (most questions have two options). Keeping LayerNorm in fp32 (`TRT_LN_FP32=1`) changed nothing.

The cause is rotary position embedding. Hugging Face ModernBERT computes `freqs = inv_freq @ position_ids` inside `autocast(enabled=False)` to force fp32. Angles reach ~1000 rad. After ONNX export nothing forces fp32, and TensorRT computes the angles in the engine precision:

- bf16 has 7 mantissa bits: between 512 and 1024 representable values are 4 rad apart, so late positions get effectively random angles;
- fp16 spacing in the same range is 0.5 rad, a smaller but real error.

Fix (in `python/export_onnx.py`): precompute cos/sin for positions 0-1023 in fp32 as constants and gather by position. The table values are in [-1, 1], which fp16 and bf16 represent well.

| | Before | After |
|---|---|---|
| fp16 agreement | 96.82% | 97.34% |
| bf16 agreement | 64.09% | 97.23% |
| fp16 throughput (1 GPU, c=128) | 403-405 rps | 407-411 rps |

## 4. FP8 post-training quantization

Tooling: NVIDIA ModelOpt 0.46 ONNX PTQ → Q/DQ ONNX → TensorRT 10.13 through ONNX Runtime 1.30. Script: `python/quantize_fp8.py`.

Problems hit, in order:

1. Defaults use `entropy` calibration and put the CPU first in `calibration_eps`: numpy histograms over hundreds of large tensors on CPU; still running after 1 h 46 min with 0% GPU. `max` calibration on `cuda:N` finishes in ~3 minutes.
2. The GEMV-detection pass keeps every MatMul output alive and ran out of memory on a shared GPU → `enable_gemv_detection_for_trt=False`.
3. fp32 calibration at batch 8 still ran out of memory → batch 1, which also avoids garbage values in padded positions.
4. Calibrating the fp16 graph produced -inf/NaN histograms; calibrate the fp32 graph and let ModelOpt convert the rest to fp16.
5. FP8 per-channel weight DQ in the 2-layer head fails in TensorRT's Myelin compiler (`No matching rules found for input operand types`) → exclude `/m/layers.*`.
6. By default residual `Add`s are quantized too; the engine output was all NaN → `op_types_to_quantize=["MatMul", "Gemm"]`.
7. The RoPE MatMul was quantized; agreement 54% → exclude (gone entirely with the RoPE-table export).

Why FP8 does not pay off here: mmBERT has a massive activation on the first token. From layer 11 on, one channel of the MLP output sits at ~3.56e4 (and ~1.39e4 in the residual stream) on every input we measured, while typical activations are 0.1-100. Ada's FP8 path uses one scale per tensor; E4M3 tops out at 448, so the scale must be ~80×, and the smallest normal FP8 value then maps to ~1.2 — most ordinary activations underflow. Excluding the MLP output projections leaves 66 FP8 GEMMs (Wqkv, attention Wo, MLP Wi in all 22 layers).

| FP8 variant | FP8 GEMMs | Agreement | 1 GPU, c=128 |
|---|---|---|---|
| + MLP Wo in layers 0-10 | 77 | 89.29% | 434-443 rps |
| MLP Wo excluded (default) | 66 | 89.52-90.15% | 432-447 rps |
| everything incl. attention Q/DQ | – | NaN | 439-454 rps |

Result: +7-9% throughput, no latency change, agreement 97.3% → 89.5%. The loss comes from the FP8 GEMMs themselves: running the same ModelOpt pipeline with zero quantized nodes gives 96.89%. Block-scaled formats (MXFP8 / NVFP4 on Blackwell) or SmoothQuant-style outlier migration are the options left; neither is tested here.

Side note: ModelOpt's recorded activation amax was consistently 12.44× the fp32 amax we measured directly (66 tensors); overwriting the scales with the measured values did not improve agreement (89.09%).

## 5. Reproducing

`scripts/bench_sweep.sh` (Linux) and `windows/run_sweep.ps1` (Windows) run parity plus a concurrency sweep for each precision. Use your own request log in the `examples/laya_smoke.jsonl` format for meaningful numbers: throughput depends heavily on tokens per question and questions per request.
