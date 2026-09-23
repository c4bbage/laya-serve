"""Export laya to ONNX + dump everything the Go reimplementation needs.

Usage:
  python export_onnx.py [fixtures.jsonl]        # default: ../examples/basic.jsonl
  LAYA_MODEL=/local/path python export_onnx.py  # default: convaiinnovations/laya (HF hub)
  ONLY_FP32=1 ONNX_OUT=laya32.onnx python export_onnx.py   # just the fp32 graph

Outputs (in cwd):
  laya.onnx            DecisionModel, fp32 (marker_mask as int64, bool cast inside graph)
  laya16.onnx          same graph with fp16 weights (needs CUDA)
  laya_go_cfg.json     max_len / head_max_len / temperature / specials
  laya_vocab.json      token piece -> id
  laya_merges.json     BPE merges as a JSON list of pairs (pieces may contain newlines)
  laya_token_probe.json  tokenizer probe for `paritycheck -mode tokens`
  laya_parity.jsonl    per fixture per question: ids, markers, qtype + bf16-GPU answers
  laya_logits_sample.json  fp32 logits/act_logits sample for ONNX numeric check

Recommended for TensorRT: export fp32 (laya.onnx), run patch_mask.py on it and let
TensorRT pick fp16 (TRT_PRECISION=fp16) instead of shipping the fp16-weight graph.
"""
import json
import os
import sys
import torch
import laya
from laya.common import build_sequence, collate_items, QTYPES

MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
SUB = os.environ.get("LAYA_SUBFOLDER", "multilingual")
FIXTURES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "examples", "basic.jsonl")
N_LOGIT_SAMPLE = 64
ONNX_OUT = os.environ.get("ONNX_OUT", "laya.onnx")
ONLY_FP32 = os.environ.get("ONLY_FP32") == "1"  # stop after the fp32 export (no cfg/vocab/parity rewrite)

agent = laya.load(MODEL, subfolder=SUB, device="cpu")
agent.model.encoder.config.reference_compile = False
tok = agent.tok

# ---------- replace head layers with a shape-dynamic reimplementation ----------
# nn.MultiheadAttention bakes dummy reshape constants into the ONNX graph
# (e.g. Reshape to [504, 768, 64]); reimplement the same math with symbolic shapes.
import torch.nn as nn
import torch.nn.functional as F

class DynHeadLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        sa = layer.self_attn
        d = sa.embed_dim
        self.H, self.hd = sa.num_heads, d // sa.num_heads
        self.qkv = nn.Linear(d, 3 * d)
        with torch.no_grad():
            self.qkv.weight.copy_(sa.in_proj_weight)
            self.qkv.bias.copy_(sa.in_proj_bias)
        self.out_proj = sa.out_proj
        self.norm1, self.norm2 = layer.norm1, layer.norm2
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        self.act = layer.activation

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=False):
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.H, self.hd).transpose(1, 2)
        k = k.view(B, S, self.H, self.hd).transpose(1, 2)
        v = v.view(B, S, self.H, self.hd).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) / (self.hd ** 0.5)
        if src_key_padding_mask is not None:
            pad = src_key_padding_mask.to(att.dtype).view(B, 1, 1, S)
            att = att + pad * torch.finfo(att.dtype).min
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v)  # B,H,S,hd
        o = o.transpose(1, 2).reshape(B, S, D)
        o = self.out_proj(o)
        x = x + o
        h2 = self.norm2(x)
        x = x + self.linear2(self.act(self.linear1(h2)))
        return x

for i, layer in enumerate(agent.model.head.layers):
    agent.model.head.layers[i] = DynHeadLayer(layer)
print("head layers replaced with dynamic-shape reimplementation")

# ---------- RoPE as a precomputed fp32 table ----------
# HF computes the RoPE angles (pos x inv_freq, up to ~1024 rad) under
# autocast(enabled=False) to force fp32. In ONNX that intent is lost: TRT in
# fp16/bf16 mode recomputes the angles in reduced precision (bf16 ulp at 512-1024
# is 4 rad -> random angles; FP8 on this MatMul dropped parity to 54%).
# Bake cos/sin for every position in fp32 and gather by position_ids; the table
# values are in [-1, 1], which fp16/bf16 represent fine.
from transformers.models.modernbert import modeling_modernbert as _mb

ROPE_MAX_POS = int(agent.cfg["max_len"])

def _rope_table_forward(self, x, position_ids):
    if not hasattr(self, "_cos_tbl"):
        pos = torch.arange(ROPE_MAX_POS, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(pos, self.inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_tbl = emb.cos() * self.attention_scaling
        self._sin_tbl = emb.sin() * self.attention_scaling
    cos = self._cos_tbl.to(x.device)[position_ids]
    sin = self._sin_tbl.to(x.device)[position_ids]
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

_mb.ModernBertRotaryEmbedding.forward = _rope_table_forward
print("rotary embedding patched (fp32 cos/sin table, max_pos=%d)" % ROPE_MAX_POS)

# ---------- patch forward: graph-safe TopK ----------
# DecisionModel.forward's k==1 branch (topk(1)+pad) traces to topk(2) on the dummy
# (k>=2), which explodes at runtime for single-option questions. Pad with a zero
# column first so topk(2) is always valid and matches the k==1 semantics exactly.
import types

def dyn_forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
    h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    if detach_encoder:
        h = h.detach()
    h = h + self.type_emb(qtype)[:, None, :]
    if self.head is not None:
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    m = torch.gather(h, 1, idx)
    logits = self.scorer(m).squeeze(-1)
    logits = logits.masked_fill(~marker_mask, -1e4)
    p = torch.softmax(logits.detach(), -1)
    k = marker_mask.sum(-1).clamp(min=2).to(p.dtype)
    ent = -(p * torch.log(p.clamp_min(6e-5))).sum(-1) / torch.log(k)
    ppad = torch.cat([p, torch.zeros_like(p[..., :1])], dim=-1)
    top2 = ppad.topk(2, -1).values
    feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
    pooled = h[:, 0]
    act_logits = self.act_head(torch.cat([pooled, feats], -1))
    return logits.float(), act_logits.float()

agent.model.forward = types.MethodType(dyn_forward, agent.model)
print("forward patched (graph-safe TopK)")

# ---------- cfg ----------
tj = json.loads(tok.backend_tokenizer.to_str())
cfg = {
    "max_len": int(agent.cfg["max_len"]),
    "head_max_len": int(agent.cfg["head_max_len"]),
    "temperature": list(agent.temperature),
    "temperature_by_options": agent.temperature_by_options,
    "cls_token_id": tok.cls_token_id,
    "sep_token_id": tok.sep_token_id,
    "pad_token_id": tok.pad_token_id,
    "mask_token_id": tok.mask_token_id,
    "mask_token": tok.mask_token,
    "unk_token_id": tok.unk_token_id,
    "qtypes": QTYPES,
}
if not ONLY_FP32:
    json.dump(cfg, open("laya_go_cfg.json", "w"), ensure_ascii=False)
    print("cfg:", cfg)

    # ---------- vocab + merges ----------
    json.dump(tj["model"]["vocab"], open("laya_vocab.json", "w"), ensure_ascii=False)
    # older tokenizers serialize merges as "a b" strings, newer ones as [a, b] pairs
    merges = [m.split(" ", 1) if isinstance(m, str) else m for m in tj["model"]["merges"]]
    json.dump(merges, open("laya_merges.json", "w"), ensure_ascii=False)
    print("vocab:", len(tj["model"]["vocab"]), "merges:", len(merges))

# ---------- ONNX export ----------
class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask_int, qtype):
        return self.m(input_ids, attention_mask, marker_pos, marker_mask_int.bool(), qtype)

def _to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}

fixtures = [json.loads(l) for l in open(FIXTURES, encoding="utf-8") if l.strip()]

# ---------- tokenizer probe (for `paritycheck -mode tokens`) ----------
if not ONLY_FP32:
    from laya.common import render_options, serialize_state

    def _enc(text):
        return tok(text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]

    probe = []
    for fx in fixtures[:150]:
        qs = {}
        for qid, qdef in fx["questions"].items():
            q = _to_internal(qdef)
            qs[qid] = dict(qdef)
            qs[qid]["head"] = _enc("%s question: %s" % (q["t"], q["ins"]))
            qs[qid]["opts"] = [_enc(" " + o)[:48] for o in render_options(q)]
        probe.append({"id": fx["id"], "state": fx["state"],
                      "state_ids": _enc(serialize_state(fx["state"])), "questions": qs})
    json.dump(probe, open("laya_token_probe.json", "w"), ensure_ascii=False)
    print("token probe:", len(probe), "fixtures")

items = []
for fx in fixtures[:8]:
    for qid, qdef in fx["questions"].items():
        q = _to_internal(qdef)
        seq, markers = build_sequence(tok, fx["state"], q, cfg["max_len"], cfg["head_max_len"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

b = collate_items([items], tok.pad_token_id)
ids = b["input_ids"][:4]
att = b["attention_mask"][:4]
mpos = b["marker_pos"][:4]
mmask = b["marker_mask"][:4].long()
qt = b["qtype"][:4]

wrap = Wrap(agent.model).eval()
# nn.TransformerEncoderLayer eval fast path (aten::_transformer_encoder_layer_fwd) is not
# ONNX-exportable; disable the MHA fast path globally (slow path is mathematically identical).
torch.backends.mha.set_fastpath_enabled(False)
with torch.inference_mode():
    torch.onnx.export(
        wrap, (ids, att, mpos, mmask, qt), ONNX_OUT,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "marker_pos": {0: "batch", 1: "kmax"},
            "marker_mask": {0: "batch", 1: "kmax"},
            "qtype": {0: "batch"},
            "logits": {0: "batch", 1: "kmax"},
            "act_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
print("onnx exported:", ONNX_OUT)
if ONLY_FP32:
    sys.exit(0)

# ---------- fp16 export (weights in fp16, no autocast cast-nodes) ----------
agent.model.half().to("cuda:0")
wrap16 = Wrap(agent.model).eval()
ids16, att16, mpos16, mmask16, qt16 = ids.to("cuda:0"), att.to("cuda:0"), mpos.to("cuda:0"), mmask.to("cuda:0"), qt.to("cuda:0")
with torch.inference_mode():
    torch.onnx.export(
        wrap16, (ids16, att16, mpos16, mmask16, qt16), "laya16.onnx",
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "marker_pos": {0: "batch", 1: "kmax"},
            "marker_mask": {0: "batch", 1: "kmax"},
            "qtype": {0: "batch"},
            "logits": {0: "batch", 1: "kmax"},
            "act_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
agent.model.float().to("cpu")
print("fp16 export saved:", os.path.getsize("laya16.onnx"))

# ---------- fp32 logits sample ----------
with torch.inference_mode():
    lg, act = agent.model(ids, att, mpos, b["marker_mask"][:4], qt)
json.dump({
    "input_ids": ids.tolist(), "attention_mask": att.tolist(),
    "marker_pos": mpos.tolist(), "marker_mask": mmask.tolist(), "qtype": qt.tolist(),
    "logits": lg.tolist(), "act_logits": act.tolist(),
}, open("laya_logits_sample.json", "w"))
print("logits sample dumped")

# ---------- parity dump: all fixtures, items + bf16-GPU answers ----------
gpu = laya.load(MODEL, subfolder=SUB, device="cuda:0")
gpu.model.encoder.config.reference_compile = False

with open("laya_parity.jsonl", "w", encoding="utf-8") as f, torch.inference_mode():
    n_q = 0
    for i, fx in enumerate(fixtures):
        entry = {"id": fx["id"], "questions": {}, "answers": {}}
        for qid, qdef in fx["questions"].items():
            q = _to_internal(qdef)
            seq, markers = build_sequence(tok, fx["state"], q, cfg["max_len"], cfg["head_max_len"])
            entry["questions"][qid] = {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}
            n_q += 1
        out = gpu.predict(fx["state"], fx["questions"])
        entry["answers"] = out["answers"]
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"parity dump: {len(fixtures)} fixtures, {n_q} questions")
