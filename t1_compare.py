"""Build compare.md from frozen T1 summaries. Does not score audio."""
from __future__ import annotations

import json
from pathlib import Path

from t1_score import PART_A_NAMES, PART_B_DIALECTS

GEN_TAGS = ("s45190", "s5482_11labs", "s1370_11labs", "s45190_rerun")
TAGS = GEN_TAGS + ("real",)
A_ORDER = ("ar_msa", "ar_egy", "ar_sau", "ar_uae", "ar_irq", "ar_dza", "ar_mar")
B_NAMES = {
    "emirati": "Emirati",
    "kuwaiti": "Kuwaiti",
    "najdi": "Najdi",
    "hijazi": "Hijazi",
    "egyptian": "Egyptian",
    "levantine": "Levantine",
    "msa": "MSA",
}
MODES = (("xvec", "x_vector_only"), ("icl", "in-context"))
METRICS = (
    ("wer", "WER"),
    ("ler", "letter"),
    ("dropped_ending", "dropped endings"),
    ("dropped_words", "dropped words"),
    ("utmos", "UTMOS"),
    ("sim", "same speaker"),
    ("sim_other", "different speaker"),
    ("broken", "broken"),
    ("broken_silent", "silent"),
    ("broken_cut_off", "cut off"),
    ("broken_too_long", "too long"),
    ("broken_repeating", "repeating"),
)


def _load(root: Path, tag: str) -> dict | None:
    path = root / tag / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _block(doc: dict | None, part: str, mode: str | None = None) -> dict | None:
    if not doc:
        return None
    block = doc.get(part)
    if part == "partB":
        if not isinstance(block, dict):
            return None
        return block.get(mode or "xvec")
    return block


def _node(block: dict | None, dialect: str | None) -> dict:
    if not block:
        return {}
    if dialect is None:
        return block
    return (block.get("by_dialect") or {}).get(dialect) or {}


def _triple(block: dict | None, dialect: str | None, key: str):
    node = _node(block, dialect)
    return node.get(key), node.get(f"{key}_min"), node.get(f"{key}_max")


def _overlap(left, right) -> bool:
    if any(value is None for value in (left[1], left[2], right[1], right[2])):
        return False
    return not (left[2] < right[1] or right[2] < left[1])


def _fmt(value, low, high, mark: bool = False) -> str:
    if value is None:
        return ""
    text = f"{float(value):.3f}"
    if low is not None and high is not None:
        text += f" ({float(low):.3f}–{float(high):.3f})"
    if mark:
        text += " no real difference"
    return text


def _cell(block, dialect, key, base) -> str:
    got = _triple(block, dialect, key)
    mark = bool(base) and _overlap(got, base)
    return _fmt(*got, mark)


def _best(docs: dict, part: str, dialect: str | None, key: str, mode: str | None = None):
    found = []
    for tag in GEN_TAGS:
        block = _block(docs.get(tag), part, mode)
        mean, low, high = _triple(block, dialect, key)
        if mean is not None:
            found.append((mean, tag, (mean, low, high)))
    if not found:
        return None
    found.sort()
    return found[0]


def _plain(docs: dict) -> list[str]:
    lines = []
    best = _best(docs, "partA", None, "priority_wer")
    real = _triple(_block(docs.get("real"), "partA"), None, "priority_wer")
    if best:
        lines.append(
            f"Best checkpoint overall on Part A priority WER is {best[1]} at {_fmt(*best[2])}."
        )
    else:
        lines.append("No generation checkpoint has a Part A score.")
    per = []
    for dialect in A_ORDER:
        winner = _best(docs, "partA", dialect, "wer")
        if winner:
            per.append(f"{PART_A_NAMES[dialect]} {winner[1]} {_fmt(*winner[2])}")
    if per:
        lines.append("Best Part A checkpoint per dialect: " + "; ".join(per) + ".")
    per_b = []
    for dialect in PART_B_DIALECTS:
        winner = _best(docs, "partB", dialect, "wer", "xvec")
        if winner:
            per_b.append(f"{B_NAMES[dialect]} {winner[1]} {_fmt(*winner[2])}")
    if per_b:
        lines.append("Best Part B x_vector_only checkpoint per dialect: " + "; ".join(per_b) + ".")
    if best and real[0] is not None:
        gap = best[0] - real[0]
        lines.append(
            f"The gap from {best[1]} to real recordings is {gap:.3f} WER. "
            f"Real-recording WER is {_fmt(*real)}, and that is the part the ASR itself contributes."
        )
    stage = _block(docs.get("s45190"), "partA")
    helped, hurt, same = [], [], []
    for label, tag in (("step 5482", "s5482_11labs"), ("step 1370", "s1370_11labs")):
        other = _block(docs.get(tag), "partA")
        if not stage or not other:
            continue
        for dialect in A_ORDER:
            left = _triple(other, dialect, "wer")
            right = _triple(stage, dialect, "wer")
            if left[0] is None or right[0] is None:
                continue
            name = PART_A_NAMES[dialect]
            if _overlap(left, right):
                same.append(f"{label} {name}")
            elif left[0] < right[0]:
                helped.append(f"{label} {name}")
            else:
                hurt.append(f"{label} {name}")
    if helped or hurt or same:
        lines.append(
            "ElevenLabs-data training versus Stage B1 on Part A: helped "
            + (", ".join(helped) or "none")
            + "; hurt "
            + (", ".join(hurt) or "none")
            + "; no real difference on "
            + (", ".join(same) or "none")
            + "."
        )
    xvec = _triple(_block(docs.get("s45190"), "partB", "xvec"), None, "overall_wer")
    icl = _triple(_block(docs.get("s45190"), "partB", "icl"), None, "overall_wer")
    if xvec[0] is not None and icl[0] is not None:
        if _overlap(xvec, icl):
            verdict = "no real difference"
        elif icl[0] < xvec[0]:
            verdict = "in-context cloning is lower"
        else:
            verdict = "x_vector_only is lower"
        lines.append(
            f"On Stage B1 Part B, in-context cloning is {_fmt(*icl)} and x_vector_only is {_fmt(*xvec)}: {verdict}."
        )
    focus = best[1] if best else "s45190"
    broken_bits = []
    for part, mode, label in (("partA", None, "Part A"), ("partB", "xvec", "Part B x_vector_only")):
        block = _block(docs.get(focus), part, mode)
        if not block:
            continue
        order = A_ORDER if part == "partA" else PART_B_DIALECTS
        names = PART_A_NAMES if part == "partA" else B_NAMES
        worst_broken = _worst(block, order, names, "broken")
        worst_end = _worst(block, order, names, "dropped_ending")
        if worst_broken:
            broken_bits.append(f"{label} broken {worst_broken[0]} {_fmt(*worst_broken[1])}")
        if worst_end:
            broken_bits.append(f"{label} dropped endings {worst_end[0]} {_fmt(*worst_end[1])}")
    if broken_bits:
        lines.append(f"On {focus}, the highest rates are " + "; ".join(broken_bits) + ".")
    return lines[:10]


def _worst(block: dict, order, names, key: str):
    found = []
    for dialect in order:
        got = _triple(block, dialect, key)
        if got[0] is not None:
            found.append((got[0], names[dialect], got))
    if not found:
        return None
    found.sort(reverse=True)
    return found[0][1], found[0][2]


def _metric_table(title: str, docs: dict, part: str, dialect: str | None, mode: str | None, note: str) -> list[str]:
    lines = [f"### {title}", ""]
    if note:
        lines.append(note)
        lines.append("")
    headers = ["checkpoint", *[label for _, label in METRICS]]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(METRICS))) + " |")
    base_doc = docs.get("s45190")
    base = _block(base_doc, part, mode)
    tags = TAGS if part == "partA" else GEN_TAGS
    for tag in tags:
        block = _block(docs.get(tag), part, mode)
        if block is None:
            lines.append(f"| {tag} | missing |")
            continue
        cells = []
        for key, _label in METRICS:
            reference = None if tag == "s45190" else _triple(base, dialect, key)
            cells.append(_cell(block, dialect, key, reference))
        lines.append("| " + " | ".join([tag, *cells]) + " |")
    lines.append("")
    lines.append("A number is marked `no real difference` when its seed range overlaps step 45190. The range is the min and max of the three seeds.")
    lines.append("")
    return lines


def _cost(docs: dict) -> list[str]:
    lines = ["## Cost", ""]
    lines.append("| step | dollars | GPU hours |")
    lines.append("| --- | ---: | ---: |")
    total_d = 0.0
    total_h = 0.0
    for tag in TAGS:
        doc = docs.get(tag)
        if not doc:
            lines.append(f"| {tag} | missing | |")
            continue
        dollars = float(doc.get("dollars") or 0)
        hours = float(doc.get("gpu_wall_s") or 0) / 3600
        total_d += dollars
        total_h += hours
        lines.append(f"| {tag} | {dollars:.2f} | {hours:.2f} |")
    lines.append(f"| total of finished steps | {total_d:.2f} | {total_h:.2f} |")
    lines.append("")
    lines.append("GPU hours are the sum of L40S container time. Dollars are those hours at $1.95.")
    lines.append("Part B different-speaker prompts use fixed seed 13. Najdi and Kuwaiti each have one speaker, so a different-speaker clip is from another dialect.")
    lines.append("The Kuwaiti prompt transcript has CER 0.10, so its in-context-cloning column may read slightly low.")
    lines.append("The listening pack uses the same seed and copies wavs that were already generated.")
    lines.append("")
    return lines


def render(docs: dict) -> str:
    lines = ["# T1 comparison", "", "Quality metrics use non-broken clips. Broken rates count every clip and are split by reason.", ""]
    lines.extend(_plain(docs))
    lines.append("")
    lines.append("## Part A")
    lines.append("")
    for dialect in A_ORDER:
        lines.extend(_metric_table(PART_A_NAMES[dialect], docs, "partA", dialect, None, ""))
    lines.append("## Part B")
    lines.append("")
    lines.append("Kuwaiti and Najdi Part B rows are 1 speaker. The Kuwaiti prompt transcript has CER 0.10, so its in-context-cloning column may read slightly low.")
    lines.append("")
    for dialect in PART_B_DIALECTS:
        note = "1 speaker." if dialect in {"najdi", "kuwaiti"} else ""
        if dialect == "kuwaiti":
            note += " The Kuwaiti prompt transcript has CER 0.10, so its in-context-cloning column may read slightly low."
        for mode, label in MODES:
            title = f"{B_NAMES[dialect]} {label}"
            if dialect in {"najdi", "kuwaiti"}:
                title += " (1 speaker)"
            lines.extend(_metric_table(title, docs, "partB", dialect, mode, note))
    lines.extend(_cost(docs))
    return "\n".join(lines).rstrip() + "\n"


def write_compare(root: Path, dest: Path) -> str:
    docs = {tag: loaded for tag in TAGS if (loaded := _load(root, tag))}
    text = render(docs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return text
