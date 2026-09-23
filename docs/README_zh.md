# laya-serve（中文说明）

[Laya](https://huggingface.co/convaiinnovations/laya) 的高性能推理服务与量化工具套件。Laya 是开源的 System 1 决策模型：给定现状和候选项，一次前向返回每个选项的校准概率，不生成文本。

> 非官方社区项目，与 Convai Innovations 无关联。

## 它解决什么问题

Laya 官方只提供 Python SDK。直接用 SDK + FastAPI 做服务，一张 RTX 4090 只能跑 73 rps，GPU 利用率接近 0：每个请求单独做一次前向，时间大部分花在 Python 侧。

laya-serve 用一个 Go 二进制完成分词、序列构建、动态攒批、多卡调度和后处理，模型由 ONNX Runtime + TensorRT 执行：

| 方案 | 单卡 | 双卡 | 单次延迟 |
|---|---|---|---|
| Python SDK + FastAPI（4 进程） | 73 rps | – | 34ms |
| Go 网关 + Python 批处理后端 | 193 rps | 389 rps | 56ms |
| **laya-serve + TensorRT fp16** | **402 rps** | **774 rps** | **16ms** |

吞吐指 99% 请求在 600ms 内返回时的最大值，测试数据为 2884 条真实麻将决策请求（每题约 400 token），GPU 与其他服务共享。

## 主要内容

- **Go 推理栈**（`go/layago`）：SentencePiece BPE 分词、序列构建、choice / score / noul 三种题型的后处理，与 Python `laya` 逐 token 一致（`paritycheck` 验证）
- **导出修复**（`python/export_onnx.py`）：RoPE 角度在 TensorRT 低精度下算错（bf16 一致率 64% → 修复后 97%）、fp16 掩码溢出成 NaN、`nn.MultiheadAttention` 形状写死、单选项题导出崩溃
- **精度实测**：fp32 / fp16 / bf16 / FP8 全部实测，推荐 TensorRT fp16（一致率 97.34%，吞吐是 fp32 的 4 倍）
- **FP8 量化脚本与结论**（`python/quantize_fp8.py`）：mmBERT 首 token 有 3.5 万量级的离群激活，4090 的逐张量 FP8 只换来 +7-9% 吞吐，一致率降到 89.5%，不推荐
- **压测工具**：`go/bench`、`scripts/bench_sweep.sh`、Windows 版 `windows/*.ps1`

快速开始和配置项见 [README](../README.md)，完整测试数据与排查过程见 [BENCHMARKS](BENCHMARKS.md)。

## 支持的模型

- **Laya multilingual**（mmBERT 322M，1024 上下文，100+ 语言含中英文）：支持，底座或自己微调的版本都可以，已做 parity 验证
- Laya 英文版（ModernBERT-large 421M）：暂不支持，它用 byte-level BPE 分词器，Go 端未实现
- typed-decisions 微调版：未测试

服务本身与具体业务无关，`examples/basic.jsonl` 是通用示例；麻将只是我们的测试场景，相关脚本在 `examples/mahjong/`。

## 注意

- 底座模型未针对任何任务微调，零样本接近随机水平。请先按 Laya 的 RLCD 流程微调，再重新导出并跑 `paritycheck`。
- TensorRT 引擎缓存绑定显卡型号和 TRT 版本，换卡需要重建。
