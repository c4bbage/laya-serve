"""Convert alpaca-style Sichuan mahjong samples into Laya typed-decisions JSONL.

Input  : alpaca JSON/JSONL, each item {"instruction", "input", "output"}
         - input : rendered board state (v2/v3/v4 templates)
         - output: SAL action, e.g. "出牌 3万" / "碰" / "定缺 条"
                   (v2 CoT: "...。最终决策：出牌 3万")
Output : JSONL rows {id, workflow, state, questions, gold} in the schema used by
         upstream laya RLCD fine-tune notebook (LocalLLaMA/typed-decisions style).

Usage:
    python convert_mahjong_to_laya.py --input alpaca_train.json --output laya_train.jsonl
"""
import argparse
import json
import re
import sys

ACTION_DESCRIPTIONS = {
    "过": "放弃当前可执行的操作，保持手牌不变",
    "抓牌": "从牌墙摸一张牌",
    "出牌": "从手牌中选择一张打出",
    "吃": "用两张手牌与上家打出的牌组成顺子",
    "碰": "用两张相同手牌与场上牌组成刻子",
    "明杠": "亮出三张相同手牌杠场上第四张",
    "暗杠": "手牌中四张相同牌直接开杠，不亮明",
    "补杠": "在已有碰子上补第四张成杠",
    "吃胡": "和别家打出的牌，结束本轮计分",
    "自摸": "自己摸到的牌成和，番数更高",
    "换牌": "游戏初期选牌交换",
    "定缺": "指定一门花色本局不要",
}

V4_ACT_LINE = re.compile(r"^操作:(.+)$", re.M)
V4_HAND_LINE = re.compile(r"^手:(.+)$", re.M)
V23_ACT_LINE = re.compile(r"(?:当前可选操作|当前可执行的操作)\s*\n(.+)", re.M)
V23_HAND_LINE = re.compile(r"(?:我的手牌|手牌)[:：]\s*(.+)")
COT_SUFFIX = re.compile(r"最终决策[:：]\s*")
SAL_V0 = re.compile(r"^<(.+?)>\s*(.*)$")
TILE = re.compile(r"[一二三四五六七八九1-9][万筒条]")


def parse_actlist(text: str) -> list[str]:
    m = V4_ACT_LINE.search(text) or V23_ACT_LINE.search(text)
    if not m:
        return []
    raw = m.group(1).strip()
    return [o.strip() for o in re.split(r"[,，;；]", raw) if o.strip()]


def parse_hand(text: str) -> list[str]:
    m = V4_HAND_LINE.search(text) or V23_HAND_LINE.search(text)
    if not m:
        return []
    return TILE.findall(m.group(1))


SUIT = re.compile(r"[万筒条]")


def parse_sal(output: str) -> tuple[str, list[str], list[str]]:
    out = COT_SUFFIX.split(output.strip())[-1].strip()
    m = SAL_V0.match(out)
    if m:
        action, cards_str = m.group(1), m.group(2)
    else:
        parts = out.split(None, 1)
        action = parts[0] if parts else ""
        cards_str = parts[1] if len(parts) > 1 else ""
    cards = TILE.findall(cards_str)
    suits = SUIT.findall(cards_str)
    return action, cards, suits


def unique_tiles(tiles: list[str]) -> list[str]:
    seen, out = set(), []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_item(idx: int, sample: dict) -> dict | None:
    state_text = sample.get("input", "").strip()
    output = sample.get("output", "")
    actlist = parse_actlist(state_text)
    hand = parse_hand(state_text)
    action, cards, suits = parse_sal(output)

    if not action:
        return None

    questions: dict = {}
    gold: dict = {}

    if actlist and action in actlist:
        questions["action_type"] = {
            "type": "choice",
            "instructions": "当前牌局下我应该执行哪个操作，收益最大？",
            "criteria": {a: ACTION_DESCRIPTIONS.get(a, a) for a in actlist},
        }
        gold["action_type"] = {
            "probabilities": {a: (1.0 if a == action else 0.0) for a in actlist}
        }
    elif action not in ("出牌", "定缺"):
        return None

    if action == "出牌" and cards and hand:
        tile = cards[0]
        options = unique_tiles(hand)
        if tile not in options:
            return None
        questions["discard_tile"] = {
            "type": "choice",
            "instructions": "选择打出手牌中的哪一张？",
            "criteria": {t: f"打出{t}" for t in options},
        }
        gold["discard_tile"] = {
            "probabilities": {t: (1.0 if t == tile else 0.0) for t in options}
        }
    elif action == "定缺" and suits:
        suit = suits[0]
        suit_options = ["万", "筒", "条"]
        if suit not in suit_options:
            return None
        questions["dingque_suit"] = {
            "type": "choice",
            "instructions": "我应该定缺哪一门花色？",
            "criteria": {
                "万": "定缺万子，本局不要万",
                "筒": "定缺筒子，本局不要筒",
                "条": "定缺条子，本局不要条",
            },
        }
        gold["dingque_suit"] = {
            "probabilities": {s: (1.0 if s == suit else 0.0) for s in suit_options}
        }
    elif action == "换牌":
        # 换牌需选3张，choice 单选表达不了，跳过（后续可用多次单选或多标签扩展）
        return None

    if not questions:
        return None

    return {
        "id": f"mahjong-{idx}",
        "workflow": f"mahjong_{action}",
        "state": {"board": state_text},
        "questions": questions,
        "gold": gold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="alpaca json/jsonl file")
    ap.add_argument("--output", required=True, help="output jsonl")
    args = ap.parse_args()

    samples = []
    with open(args.input, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        sys.exit("empty input")
    if content.startswith("["):
        samples = json.loads(content)
    else:
        samples = [json.loads(line) for line in content.splitlines() if line.strip()]

    kept, skipped = [], 0
    for i, s in enumerate(samples):
        item = build_item(i, s)
        if item:
            kept.append(item)
        else:
            skipped += 1

    with open(args.output, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    from collections import Counter
    wf = Counter(x["workflow"] for x in kept)
    print(f"converted: {len(kept)}  skipped: {skipped}")
    for k, v in wf.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
