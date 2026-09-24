# laya-serve

High-throughput inference server and quantization toolkit for [Laya](https://huggingface.co/convaiinnovations/laya), the open System-1 typed-decision model (mmBERT / ModernBERT encoder + option-marker scorer).

> Unofficial community project, not affiliated with Convai Innovations.

A single Go binary does tokenization, sequence building, dynamic batching and post-processing; the model runs on ONNX Runtime with the TensorRT (or CUDA) execution provider. On one RTX 4090 it serves **402 requests/s within a 600 ms p99 budget** (vs. 73 rps for a Python SDK + FastAPI service on the same card), with **16 ms** single-request latency.

中文说明见 [docs/README_zh.md](docs/README_zh.md)。

## What's inside

| Path | What it is |
|---|---|
| `go/layago` | Pure-Go Laya stack: SentencePiece-BPE tokenizer, `build_sequence`, ONNX Runtime engine (CUDA / TensorRT, fp16 / bf16 / fp32), answer post-processing for `choice` / `score` / `noul` |
| `go/goserve` | HTTP server: `/predict` (same request shape as `laya.predict`), dynamic batching, multi-GPU dispatch |
| `go/paritycheck` | Verifies the Go stack against Python dumps: tokens, sequences, and model outputs |
| `go/bench` | Load generator with SLO check |
| `go/layagate` + `python/batch_backend.py` | Alternative: Go batching gateway in front of Python PyTorch backends |
| `python/export_onnx.py` | ONNX export with the fixes TensorRT needs (see below) + all parity dumps |
| `python/patch_mask.py` | Replaces overflow-prone attention-mask constants |
| `python/quantize_fp8.py` | FP8 PTQ with NVIDIA ModelOpt (documented, not recommended on Ada) |
| `scripts/bench_sweep.sh`, `windows/*.ps1` | Precision × concurrency sweeps (Linux / Windows) |
| `examples/basic.jsonl` | Small mixed-language request set covering all question types (default fixtures) |
| `examples/mahjong/` | The use case the benchmarks came from: data converter, evaluator and hand-written samples |
| `docs/BENCHMARKS.md` | Full measurements and the debugging notes behind them |

## Supported models

| Laya checkpoint | Status |
|---|---|
| `multilingual` (mmBERT-base, 322M, 1024 tokens, 100+ languages incl. English and Chinese), base or your own fine-tune | **Supported and parity-tested** |
| English base (ModernBERT-large, 421M, 512 tokens) | Not supported yet: it uses a byte-level BPE tokenizer that `layago` does not implement |
| `typed-decisions` fine-tune | Not tested |

Nothing in the server is task-specific. Any state/questions payload that `laya.predict` accepts works, with all three question types (`choice`, `score`, `noul`).

## Results (RTX 4090, 2884 real decision requests from a mahjong game, ~400 tokens each)

Max throughput within a 600 ms p99 budget:

| Stack | 1 GPU | 2 GPUs | Single request |
|---|---|---|---|
| Python SDK + FastAPI, 4 workers | 73 rps | – | 34 ms |
| Go gateway + Python batch backends | 193 rps | 389 rps | 56 ms |
| goserve + ONNX Runtime CUDA EP (fp16) | – | 336 rps | 27 ms |
| **goserve + TensorRT fp16** | **402 rps** | **774 rps** | **16 ms** |

Precision (TensorRT, one GPU, argmax agreement with the Python bf16 reference on 4436 questions):

| Precision | Agreement | Throughput |
|---|---|---|
| fp32 | 97.50% | 94 rps |
| **fp16** | **97.34%** | **402 rps** |
| bf16 | 97.23% | 318 rps |
| FP8 (ModelOpt PTQ) | 89.52% | ~430 rps |

GPUs were shared with other services; same-config run-to-run noise was about 5%.

On a dedicated **RTX 5090 D** (Windows) the same fp16 setup serves **571 rps** on one GPU (+42%), with 97.59% agreement and ~15 ms single-request latency. Details and caveats: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quick start (Linux, NVIDIA GPU)

Requirements: Go 1.26+, Python with `laya` (0.3.5) and PyTorch with a CUDA GPU for the export (the fp16 graph and the bf16 reference answers are produced on GPU), ONNX Runtime 1.30 GPU build, and TensorRT 10.x libs for the TensorRT provider.

```bash
# 1. export the model + tokenizer + parity dumps (from the examples, or your own jsonl)
mkdir run && cd run
python ../python/export_onnx.py ../examples/basic.jsonl   # model from HF: convaiinnovations/laya
python ../python/patch_mask.py laya.onnx laya32s.onnx

# 2. build
(cd ../go && go build -o ../run/goserve ./goserve && go build -o ../run/paritycheck ./paritycheck && go build -o ../run/bench ./bench)

# 3. verify the Go stack matches Python
./paritycheck -mode tokens -dir .
./paritycheck -mode seq    -dir . -fixtures ../examples/basic.jsonl
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:/path/to/tensorrt/lib:/path/to/cuda/libs
export ORT_LIB=/path/to/onnxruntime/lib/libonnxruntime.so
TRT_CACHE=$PWD/trtcache ./paritycheck -mode full -provider trt -gpu 0 -onnx laya32s.onnx -fixtures ../examples/basic.jsonl

# 4. serve (first start builds the TensorRT engine, a few minutes; cached afterwards)
GOSERVE_PROVIDER=trt GOSERVE_GPUS=0 ONNX_PATH=laya32s.onnx TRT_CACHE=$PWD/trtcache LISTEN=:8320 ./goserve

# 5. call it
curl -s localhost:8320/predict -d '{
  "state": "Customer says the invoice total is wrong and asks for a refund.",
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team handles this?", "criteria": ["billing", "support", "sales"]},
    "urgency": {"type": "score",  "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
    "refund":  {"type": "noul",   "instructions": "Should we refund?"}
  }}'

# 6. load test
./bench -u http://127.0.0.1:8320/predict -c 128 -d 20s -f ../examples/basic.jsonl -slo 600ms
```

The base checkpoint is not tuned for any task (Laya's model card reports near-chance zero-shot accuracy); fine-tune it for your workflow, then re-export and re-run `paritycheck`.

### goserve configuration

| Env | Default | Meaning |
|---|---|---|
| `ONNX_PATH` | `laya.onnx` | model file |
| `GOSERVE_PROVIDER` | `cuda` | `cuda` or `trt` |
| `TRT_PRECISION` | `fp16` | `fp16`, `bf16` or `fp32` (TensorRT provider) |
| `TRT_CACHE` | `trtcache` | TensorRT engine/timing cache dir (bound to GPU model + TRT version) |
| `TRT_SEQ_MIN/OPT/MAX`, `TRT_KMAX` | 16/512/1024, 32 | TensorRT dynamic-shape profile |
| `TRT_LN_FP32`, `TRT_WORKSPACE` | off, TRT default | keep LayerNorm in fp32; cap builder workspace (bytes) |
| `GOSERVE_GPUS` | `0,1` | GPUs, one session each (repeat an id for more sessions) |
| `BATCH_MAX` | 48 | max **questions** per batch (TensorRT profile batch max is 64) |
| `BATCH_WINDOW_MS` | 10 | batching window |
| `BUILD_WORKERS` | NumCPU | parallel tokenization/sequence building |
| `GOSERVE_MEM_LIMIT` | 6 GiB | CUDA arena limit per session, comma-separated per session |
| `ORT_LIB` | – | path to `libonnxruntime.so` / `onnxruntime.dll` |
| `LISTEN` | `:8320` | listen address |

Endpoints: `POST /predict`, `GET /stats`, `GET /healthz`.

## Export fixes you probably need for any ModernBERT-family model on TensorRT

1. **RoPE angles lose precision.** Hugging Face computes RoPE angles (position × inv_freq, up to ~1000 rad) with autocast disabled to force fp32. That intent does not survive ONNX export: TensorRT recomputes them in fp16/bf16. In bf16 the spacing between representable values at 512-1024 is 4 rad, so positions become noise (our argmax agreement dropped to 64%). `export_onnx.py` bakes a fp32 cos/sin table and gathers by position; bf16 went back to 97.2%, fp16 improved from 96.8% to 97.3%, at no speed cost.
2. **Attention-mask overflow.** `finfo(dtype).min` plus negative scores overflows to -inf in fp16; `0 * -inf = NaN`. `patch_mask.py` rewrites those constants to -10000.
3. **`nn.MultiheadAttention` bakes shapes into the graph.** The head layers are re-implemented with symbolic shapes (same weights, same math).
4. **Single-option questions.** The `k == 1` branch traces into `topk(2)` and crashes at runtime; the export pads a zero column instead.

## Why not FP8

mmBERT has a massive activation on the first token: from layer 11, one channel sits at ~3.5e4 while typical values are 0.1-100. Ada's FP8 uses one scale per tensor (E4M3 max 448), so a scale sized for 3.5e4 flushes most normal values to zero. After excluding the affected GEMMs, FP8 gave +7-9% throughput and dropped agreement to 89.5%. Block-scaled formats on Blackwell (MXFP8 / NVFP4) may behave better; not tested yet.

## Windows

`goserve.exe`, `bench.exe` and `paritycheck.exe` cross-compile with mingw-w64:

```bash
cd go && for t in goserve bench paritycheck; do
  CGO_ENABLED=1 GOOS=windows GOARCH=amd64 CC=x86_64-w64-mingw32-gcc \
    go build -trimpath -ldflags "-s -w -extldflags=-static" -o ../windows/$t.exe ./$t; done
```

`windows/setup.ps1` downloads ONNX Runtime 1.30 (CUDA 12), TensorRT 10.13, CUDA 12.9 runtime/cuBLAS/cuFFT and cuDNN 9 next to the exes (and skips the download if the DLLs are already there, e.g. copied from another machine); `windows/run_sweep.ps1` runs the precision × concurrency sweep. Validated on an RTX 5090 D (driver 591.86); needs the VC++ 2015-2022 x64 runtime.

## License

Apache-2.0. The sequence building and post-processing mirror the Apache-2.0 [`laya`](https://pypi.org/project/laya/) package; see [NOTICE](NOTICE).
