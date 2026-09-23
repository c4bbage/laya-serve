#!/bin/bash
# Precision x concurrency sweep for goserve + TensorRT on one GPU (Linux).
#
#   ORT_DIR=/path/to/onnxruntime/lib TRT_DIR=/path/to/tensorrt/lib CUDA_LIBS=/path/a:/path/b \
#   GPU=0 FIXTURES=../examples/laya_smoke.jsonl ./bench_sweep.sh
#
# Expects in the working directory: goserve, bench, paritycheck binaries, laya_vocab.json,
# laya_merges.json, laya_go_cfg.json, laya_parity.jsonl and the ONNX files listed in VARIANTS.
# First run of each variant builds a TensorRT engine (minutes), cached in trtcache_<name>/.
set -u
GPU=${GPU:-0}
FIXTURES=${FIXTURES:-laya_smoke.jsonl}
CONC=${CONC:-"1 32 64 128 192 256 384"}
PORT=${PORT:-8329}
export LD_LIBRARY_PATH=${ORT_DIR:-}:${TRT_DIR:-}:${CUDA_LIBS:-}:${LD_LIBRARY_PATH:-}
export ORT_LIB=${ORT_LIB:-${ORT_DIR:-.}/libonnxruntime.so}
export GOSERVE_PROVIDER=trt GOSERVE_GPUS=$GPU BATCH_MAX=${BATCH_MAX:-48} GOSERVE_MEM_LIMIT=${GOSERVE_MEM_LIMIT:-$((4*1024*1024*1024))}
# name:TRT precision:onnx
VARIANTS=${VARIANTS:-"fp16:fp16:laya32s.onnx bf16:bf16:laya32s.onnx fp32:fp32:laya32s.onnx fp8:fp16:laya8.onnx"}

for v in $VARIANTS; do
  IFS=: read -r name prec onnx <<< "$v"
  [ -f "$onnx" ] || { echo "$name: skip, $onnx not found"; continue; }
  mkdir -p trtcache_$name
  r=$(TRT_PRECISION=$prec TRT_CACHE=$PWD/trtcache_$name ./paritycheck -mode full -provider trt -gpu $GPU -onnx $onnx -fixtures $FIXTURES 2>&1)
  echo "$name parity: $(echo "$r" | grep -E '^full:|^engine:' | head -1)"
  TRT_PRECISION=$prec ONNX_PATH=$onnx TRT_CACHE=$PWD/trtcache_$name LISTEN=:$PORT ./goserve > goserve_$name.log 2>&1 &
  pid=$!
  for _ in $(seq 1 300); do curl -s 127.0.0.1:$PORT/healthz >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 3; done
  if ! curl -s 127.0.0.1:$PORT/healthz >/dev/null; then echo "$name: goserve failed, see goserve_$name.log"; kill $pid 2>/dev/null; continue; fi
  for c in $CONC; do
    d=20s; [ "$c" = 1 ] && d=10s
    r=$(./bench -u http://127.0.0.1:$PORT/predict -c $c -d $d -warm 3s -f $FIXTURES -slo 600ms 2>&1)
    printf "%-5s c=%-4s %-9s %s %s\n" "$name" "$c" "$(echo "$r" | grep -o 'rps=[0-9.]*')" \
      "$(echo "$r" | grep -o 'p50=[0-9.]*ms\|p95=[0-9.]*ms\|p99=[0-9.]*ms' | tr '\n' ' ')" "$(echo "$r" | grep -o '[0-9.]*% within -> [A-Z]*')"
  done
  kill $pid; wait $pid 2>/dev/null; sleep 3
done
