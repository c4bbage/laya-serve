"""Cross-request batch inference backend for laya.

POST /predict_batch
  {"requests": [{"id": "...", "state": "...|{...}", "questions": {qid: qdef}}, ...]}
  -> {"answers": {id: {qid: {...}}}, "batch_size": N, "forward_ms": x, "seq_len_avg": y}

One forward pass for ALL questions across ALL requests (dynamic batching).
Logic mirrors laya Agent.system_one (Apache-2.0, convaiinnovations/laya).
"""
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import laya
from laya.common import (
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
    QTYPES,
)
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


MODEL_PATH = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
SUBFOLDER = os.environ.get("LAYA_SUBFOLDER", "multilingual")
PORT = int(os.environ.get("PORT", "8302"))
MAX_REQ_PER_BATCH = int(os.environ.get("MAX_REQ_PER_BATCH", "64"))

agent = laya.load(MODEL_PATH, subfolder=SUBFOLDER)
MAX_LEN = agent.cfg.get("max_len", 512)
HEAD_MAX_LEN = agent.cfg.get("head_max_len", 192)

app = FastAPI()

STATS = {"batches": 0, "requests": 0, "questions": 0, "forward_ms": 0.0, "build_ms": 0.0, "seq_len_sum": 0, "t0": time.time()}


def _to_internal(qdef: Dict) -> Dict:
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def _answer_one(q: Dict, logits_row: np.ndarray, k: int, qt: int) -> Dict:
    t_scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
    z = logits_row[:k] / t_scale
    p = np.exp(z - z.max())
    p = p / p.sum()
    conf_score = round(confidence_from_probs(p, k), 4)
    if q["t"] == "choice":
        keys = list(q["crit"].keys())
        return {
            "type": "choice",
            "choice": keys[int(p.argmax())],
            "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
            "confidence": conf_score,
        }
    if q["t"] == "score":
        exp_score = float((np.arange(k) * p).sum())
        return {
            "type": "score",
            "score": round(exp_score, 4),
            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
            "confidence": conf_score,
        }
    return {
        "type": "noul",
        "noul": round(float(p[1]), 4),
        "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
    }


class PredictItem(BaseModel):
    id: str
    state: Any
    questions: Dict[str, Dict[str, Any]]


class PredictBatch(BaseModel):
    requests: List[PredictItem]


@torch.inference_mode()
def run_batch(requests: List[PredictItem]) -> Dict:
    t_all = time.perf_counter()

    items = []
    meta: List[Tuple[int, str, Dict, int]] = []  # (req_idx, qid, q_internal, qtype)
    for ri, r in enumerate(requests):
        for qid, qdef in r.questions.items():
            q = _to_internal(qdef)
            seq, markers = build_sequence(agent.tok, r.state, q, MAX_LEN, HEAD_MAX_LEN)
            if len(markers) != len(render_options(q)):
                raise ValueError(f"question {qid!r} options exceed head_max_len={HEAD_MAX_LEN}")
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
            meta.append((ri, qid, q, QTYPES[q["t"]]))
    if not items:
        return {"answers": {r.id: {} for r in requests}, "forward_ms": 0.0}

    t_build = time.perf_counter()
    b = collate_items([items], agent.tok.pad_token_id)
    use_amp = agent.device.type == "cuda"
    with torch.autocast(device_type=agent.device.type, dtype=agent.dtype, enabled=use_amp):
        logits, _act = agent.model(
            b["input_ids"].to(agent.device),
            b["attention_mask"].to(agent.device),
            b["marker_pos"].to(agent.device),
            b["marker_mask"].to(agent.device),
            b["qtype"].to(agent.device),
        )
    if agent.device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.perf_counter()

    logits = logits.float().cpu().numpy()

    answers: Dict[str, Dict] = {r.id: {} for r in requests}
    for row, (ri, qid, q, qt) in enumerate(meta):
        k = len(items[row]["markers"])
        answers[requests[ri].id][qid] = _answer_one(q, logits[row], k, qt)

    t_end = time.perf_counter()
    STATS["batches"] += 1
    STATS["requests"] += len(requests)
    STATS["questions"] += len(items)
    STATS["forward_ms"] += (t_fwd - t_build) * 1000
    STATS["build_ms"] += (t_build - t_all) * 1000
    STATS["seq_len_sum"] += sum(len(it["ids"]) for it in items)

    return {
        "answers": answers,
        "batch_size": len(requests),
        "n_questions": len(items),
        "forward_ms": round((t_fwd - t_build) * 1000, 1),
        "total_ms": round((t_end - t_all) * 1000, 1),
    }


@app.post("/predict_batch")
def predict_batch(pb: PredictBatch):
    return run_batch(pb.requests[:MAX_REQ_PER_BATCH])


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "subfolder": SUBFOLDER}


@app.get("/stats")
def stats():
    n = max(1, STATS["batches"])
    up = time.time() - STATS["t0"]
    return {
        "uptime_s": round(up, 1),
        "batches": STATS["batches"],
        "requests_total": STATS["requests"],
        "questions_total": STATS["questions"],
        "avg_batch_size": round(STATS["requests"] / n, 1),
        "avg_questions_per_batch": round(STATS["questions"] / n, 1),
        "avg_forward_ms": round(STATS["forward_ms"] / n, 1),
        "avg_build_ms": round(STATS["build_ms"] / n, 1),
        "avg_seq_len": round(STATS["seq_len_sum"] / max(1, STATS["questions"]), 1),
        "rps": round(STATS["requests"] / up, 1),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
