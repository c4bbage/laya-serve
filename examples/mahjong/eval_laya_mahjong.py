"""Evaluate Laya on converted mahjong typed-decisions data.

Usage:
    python eval_laya_mahjong.py --data laya_smoke.jsonl [--url http://localhost:8000]

Metrics per question id:
    - top-1 accuracy (argmax == gold label)
    - gold probability mass (how much prob the model put on the right answer)
    - ECE (calibration)
    - latency stats
Prints a confusion matrix for action_type and per-workflow breakdown.
"""
import argparse
import json
import statistics
import urllib.request
from collections import Counter, defaultdict


def call(url, state, questions):
    data = json.dumps({"state": state, "questions": questions}).encode()
    req = urllib.request.Request(f"{url}/predict", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def ece(bins_acc_conf):
    """Expected calibration error from (correct, confidence) pairs."""
    n = len(bins_acc_conf)
    if n == 0:
        return 0.0
    ece_val, bins = 0.0, defaultdict(list)
    for correct, conf in bins_acc_conf:
        bins[min(int(conf * 10), 9)].append((correct, conf))
    for b, items in bins.items():
        w = len(items) / n
        acc = sum(c for c, _ in items) / len(items)
        avg_conf = sum(cf for _, cf in items) / len(items)
        ece_val += w * abs(acc - avg_conf)
    return ece_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    print(f"cases: {len(rows)}")

    stats = defaultdict(lambda: {"n": 0, "top1": 0, "gold_p": 0.0, "lat": [], "ac": []})
    confusion = defaultdict(Counter)
    per_workflow = defaultdict(lambda: {"n": 0, "top1": 0})
    tokens = []

    for r in rows:
        res = call(args.url, r["state"], r["questions"])
        tokens.append(res.get("usage", {}).get("input_tokens", 0))
        for qid, q in r["questions"].items():
            gold = r["gold"].get(qid)
            if not gold:
                continue
            pred = res["answers"][qid]
            gold_label = max(gold["probabilities"], key=gold["probabilities"].get)
            probs = pred.get("probabilities", {})
            top = max(probs, key=probs.get) if probs else None
            s = stats[qid]
            s["n"] += 1
            correct = top == gold_label
            s["top1"] += correct
            s["gold_p"] += probs.get(gold_label, 0.0)
            s["lat"].append(res.get("latency_ms", 0))
            conf = pred.get("confidence", probs.get(top, 0.0))
            s["ac"].append((correct, conf))
            if qid == "action_type":
                confusion[gold_label][top] += 1
            per_workflow[r["workflow"]]["n"] += 1
            per_workflow[r["workflow"]]["top1"] += correct

    print(f"{'question':16s} {'n':>4s} {'top1':>7s} {'gold_p':>7s} {'ECE':>6s} {'p50_ms':>7s}")
    for qid, s in sorted(stats.items()):
        lats = sorted(s["lat"])
        print(f"{qid:16s} {s['n']:4d} {s['top1']/s['n']:7.1%} "
              f"{s['gold_p']/s['n']:7.3f} {ece(s['ac']):6.3f} "
              f"{statistics.median(lats):7.1f}")

    if confusion:
        labels = sorted(set(confusion) | {l for c in confusion.values() for l in c})
        print("\naction_type confusion (rows=gold, cols=pred):")
        print("        " + "".join(f"{l:>6s}" for l in labels))
        for g in labels:
            print(f"{g:>7s} " + "".join(f"{confusion[g].get(p,0):6d}" for p in labels))

    print("\nper workflow:")
    for wf, s in sorted(per_workflow.items()):
        print(f"  {wf:20s} n={s['n']:4d}  top1={s['top1']/s['n']:.1%}")

    if tokens:
        print(f"\ninput tokens: avg={statistics.mean(tokens):.0f} "
              f"max={max(tokens)} (context budget 1024)")


if __name__ == "__main__":
    main()
