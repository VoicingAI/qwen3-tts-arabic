"""Decoding sweep for step 5482. Reads the checkpoint and writes only under /data/tasks/t2_decode."""
from __future__ import annotations

import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

from t1_score import (
    PART_A_NAMES,
    PART_A_PRIORITY,
    PART_B_DIALECTS,
    _mean,
    _pack_rows,
    _usable,
    _value,
    aggregate,
)

CKPT = f"{_DATA}/exp_11labs/runs/step-005482-base"
ROOT = Path(f"{_DATA}/tasks/t2_decode")
RESULTS = ROOT / "results"
EXAM = ROOT / "exam.jsonl"
REPORT = ROOT / "report.md"
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
CURRENT = {
    "do_sample": True,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_temperature": 0.9,
    "subtalker_top_p": 1.0,
    "subtalker_top_k": 50,
}
LANG = {
    "emirati": "ar_uae",
    "kuwaiti": "ar_gulf",
    "najdi": "ar_sau",
    "hijazi": "ar_sau",
    "egyptian": "ar_egy",
    "levantine": "ar_lev",
    "msa": "ar_msa",
}


def with_changes(changes: dict) -> dict:
    """One knob moves. Subtalker temperature, top-p, and top-k follow unless set explicitly."""
    decode = dict(CURRENT)
    decode.update(changes)
    if "temperature" in changes and "subtalker_temperature" not in changes:
        decode["subtalker_temperature"] = changes["temperature"]
    if "top_p" in changes and "subtalker_top_p" not in changes:
        decode["subtalker_top_p"] = changes["top_p"]
    if "top_k" in changes and "subtalker_top_k" not in changes:
        decode["subtalker_top_k"] = changes["top_k"]
    return decode


def temperature_jobs() -> list[dict]:
    jobs = []
    for value in (0.9, 0.8, 0.7, 0.6, 0.5):
        jobs.append({
            "tag": f"temp_{value:.1f}",
            "stage": "temperature",
            "decode": with_changes({"temperature": value}),
        })
    return jobs


def top_p_jobs(temperature: float) -> list[dict]:
    jobs = []
    for value in (0.8, 0.9):
        jobs.append({
            "tag": f"topp_{value:.1f}",
            "stage": "top_p",
            "decode": with_changes({"temperature": temperature, "top_p": value}),
        })
    return jobs


def repetition_jobs(temperature: float) -> list[dict]:
    jobs = []
    for value in (1.0, 1.1, 1.2):
        jobs.append({
            "tag": f"rep_{value:.1f}",
            "stage": "repetition_penalty",
            "decode": with_changes({"temperature": temperature, "repetition_penalty": value}),
        })
    return jobs


def subtalker_jobs(temperature: float) -> list[dict]:
    jobs = []
    for value in (0.6, 0.9):
        jobs.append({
            "tag": f"sub_{value:.1f}",
            "stage": "subtalker_temperature",
            "decode": with_changes({"temperature": temperature, "subtalker_temperature": value}),
        })
    return jobs


def combined_job(chosen: dict) -> dict:
    decode = with_changes({
        "temperature": chosen["temperature"],
        "top_p": chosen["top_p"],
        "repetition_penalty": chosen["repetition_penalty"],
        "subtalker_temperature": chosen["subtalker_temperature"],
    })
    return {"tag": "combined", "stage": "combined", "decode": decode}


def same_decode(left: dict, right: dict) -> bool:
    keys = (
        "temperature", "top_k", "top_p", "repetition_penalty",
        "subtalker_temperature", "subtalker_top_p", "subtalker_top_k",
    )
    return all(float(left[key]) == float(right[key]) for key in keys)


def select_part_a(rows: list[dict], n: int = 300, seed: int = 13) -> tuple[list[dict], dict]:
    """Stratified by dialect. Seed 13 breaks the ties and picks the clips."""
    by = defaultdict(list)
    for row in rows:
        by[row["dialect"]].append(row)
    total = sum(len(items) for items in by.values())
    raw = {dialect: n * len(by[dialect]) / total for dialect in A_ORDER if dialect in by}
    alloc = {dialect: int(raw[dialect]) for dialect in raw}
    left = n - sum(alloc.values())
    groups = []
    for dialect in sorted(raw, key=lambda name: -(raw[name] - alloc[name])):
        fraction = raw[dialect] - alloc[dialect]
        if not groups or abs(fraction - groups[-1][0]) > 1e-9:
            groups.append((fraction, [dialect]))
        else:
            groups[-1][1].append(dialect)
    rng = random.Random(seed)
    order = []
    for _fraction, group in groups:
        group = sorted(group)
        rng.shuffle(group)
        order.extend(group)
    for dialect in order:
        if left <= 0:
            break
        alloc[dialect] += 1
        left -= 1
    chosen = []
    for dialect in A_ORDER:
        if dialect not in by:
            continue
        pool = sorted(by[dialect], key=lambda row: row["id"])
        rng.shuffle(pool)
        chosen.extend(pool[:alloc[dialect]])
    chosen.sort(key=lambda row: row["id"])
    if len(chosen) != n:
        raise RuntimeError(f"Part A slice is {len(chosen)}")
    return chosen, alloc


def exam_rows(part_a: list[dict], part_b: list[dict]) -> list[dict]:
    rows = []
    for row in part_a:
        rows.append({
            "id": row["id"],
            "part": "A",
            "dialect": row["dialect"],
            "text": row["text"],
            "ref_audio": row["ref_audio"],
            "language": LANG.get(row["dialect"], row["dialect"]),
        })
    for row in part_b:
        rows.append({
            "id": row["id"],
            "part": "B",
            "dialect": row["dialect"],
            "text": row["text"],
            "ref_audio": row["ref_audio"],
            "prompt_text": row.get("prompt_text") or "",
            "prompt_clip": row.get("prompt_clip") or "",
            "language": LANG.get(row["dialect"], row["dialect"]),
        })
    rows.sort(key=lambda row: (row["part"], row["id"]))
    for index, row in enumerate(rows):
        row["i"] = index
    return rows


def generation_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "i": row["i"],
            "clip_id": row["id"],
            "dialect": row["language"],
            "text": row["text"],
            "ref_audio": row["ref_audio"],
            "mode": "xvec",
            "part": row["part"],
        }
        for row in rows
    ]


def _overlap(left_min, left_max, right_min, right_max) -> bool:
    if None in (left_min, left_max, right_min, right_max):
        return False
    return not (left_max < right_min or right_max < left_min)


def _distance(decode: dict) -> tuple:
    return (
        abs(float(decode["temperature"]) - 0.9),
        abs(float(decode["top_p"]) - 1.0),
        abs(float(decode["repetition_penalty"]) - 1.05),
        abs(float(decode["subtalker_temperature"]) - 0.9),
        abs(float(decode["subtalker_top_p"]) - 1.0),
    )


def primary_spread(rows: list[dict]) -> tuple:
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row.get("seed"), []).append(row)
    vals = []
    for group in by_seed.values():
        by = {}
        for row in group:
            if row.get("part") == "A":
                by.setdefault(row.get("dialect"), []).append(row)
        dialect_means = []
        for name in PART_A_PRIORITY:
            got = _mean([_value(row, "wer") for row in by.get(name, []) if _usable(row)])
            if got is not None:
                dialect_means.append(got)
        priority = _mean(dialect_means)
        ending = _mean([_value(row, "dropped_ending") for row in group if _usable(row)])
        if priority is not None and ending is not None:
            vals.append(priority + ending)
    if not vals:
        return None, None, None
    return _mean(vals), min(vals), max(vals)


def summarize(rows: list[dict], tag: str, decode: dict, dollars: float, gpu_wall_s: float) -> dict:
    priority, low, high = primary_spread(rows)
    part_b = [row for row in rows if row.get("part") == "B"]
    return {
        "tag": tag,
        "decode": decode,
        "partA": aggregate(rows, "A"),
        "partB": aggregate(part_b, "B", "xvec"),
        "joint": _pack_rows(rows),
        "primary": priority,
        "primary_min": low,
        "primary_max": high,
        "n": len(rows),
        "dollars": round(dollars, 4),
        "gpu_wall_s": round(gpu_wall_s, 1),
    }


def accepts(candidate: dict, baseline: dict) -> tuple[bool, str]:
    """A setting must not raise the broken rate, or drop similarity by more than 0.01.

    UTMOS is reported in the tables and is not part of this decision.
    """
    base = baseline["joint"]
    got = candidate["joint"]
    if got.get("broken") is not None and base.get("broken") is not None:
        if got["broken"] > base["broken"] and not _overlap(
            got.get("broken_min"), got.get("broken_max"), base.get("broken_min"), base.get("broken_max"),
        ):
            return False, "broken rate rose"
    if got.get("sim") is not None and base.get("sim") is not None:
        if base["sim"] - got["sim"] > 0.01 + 1e-9:
            return False, "similarity dropped more than 0.01"
    return True, ""


def pick_best(candidates: list[dict], baseline: dict) -> dict:
    pool = [baseline]
    for item in candidates:
        if item["tag"] == baseline["tag"]:
            continue
        ok, _reason = accepts(item, baseline)
        if ok:
            pool.append(item)
    pool.sort(key=lambda item: (item.get("primary") is None, item.get("primary") if item.get("primary") is not None else 9))
    leader = pool[0]
    tied = [
        item for item in pool
        if _overlap(item.get("primary_min"), item.get("primary_max"), leader.get("primary_min"), leader.get("primary_max"))
        or item["tag"] == leader["tag"]
    ]
    tied.sort(key=lambda item: (_distance(item["decode"]), item.get("primary") or 9))
    return tied[0]


def _fmt(value, low=None, high=None) -> str:
    if value is None:
        return ""
    text = f"{float(value):.3f}"
    if low is not None and high is not None:
        text += f" ({float(low):.3f}–{float(high):.3f})"
    return text


def _cell(summary: dict | None, part: str, dialect: str, key: str) -> tuple:
    if not summary:
        return None, None, None
    block = summary["partA"] if part == "A" else summary["partB"]
    node = (block.get("by_dialect") or {}).get(dialect) or {}
    return node.get(key), node.get(f"{key}_min"), node.get(f"{key}_max")


def render(docs: dict[str, dict], order: list[str], winner: str, baseline: str, alloc: dict, stopped: str, spend: dict, prior: dict | None = None) -> str:
    base = docs.get(baseline)
    win = docs.get(winner)
    lines = ["# T2 decoding sweep", ""]
    lines.extend(_plain(docs, order, winner, baseline, stopped)[:8])
    lines.append("")
    lines.append(f"Checkpoint `{CKPT}`, read only. Mode x_vector_only. Part A slice seed 13: " + ", ".join(f"{dialect} {alloc.get(dialect, 0)}" for dialect in A_ORDER) + ". Part B is all 270 rows. Seeds 1–3.")
    lines.append("Primary score is Part A priority WER plus the dropped-ending rate on the whole slice. Broken rate must not rise. Similarity must not drop by more than 0.01. UTMOS is shown in the tables and is not used to pick a winner. A tie inside the seed spread keeps the setting closer to temperature 0.9, top-p 1.0, repetition penalty 1.05.")
    lines.append("")
    if win and base:
        lines.append("Winning sampler: " + ", ".join(f"{key} {win['decode'][key]}" for key in ("temperature", "top_p", "top_k", "repetition_penalty", "subtalker_temperature", "subtalker_top_p", "subtalker_top_k")) + ".")
        lines.append("")
    lines.append("## Part A")
    lines.append("")
    for dialect in A_ORDER:
        lines.extend(_table(PART_A_NAMES[dialect], docs, order, "A", dialect, baseline))
    lines.append("## Part B")
    lines.append("")
    lines.append("Kuwaiti and Najdi are 1 speaker. The Kuwaiti prompt transcript has CER 0.10, so these rows are speaker-vector only and are not the in-context column.")
    lines.append("")
    for dialect in PART_B_DIALECTS:
        title = B_NAMES[dialect]
        if dialect in {"kuwaiti", "najdi"}:
            title += " (1 speaker)"
        lines.extend(_table(title, docs, order, "B", dialect, baseline))
    lines.extend(_cost(order, spend, prior or {}, docs))
    return "\n".join(lines).rstrip() + "\n"


def _plain(docs, order, winner, baseline, stopped) -> list[str]:
    base = docs.get(baseline)
    win = docs.get(winner)
    lines = []
    if not win or not base:
        lines.append("No paired current-and-winner scores were finished.")
    else:
        same = winner == baseline or same_decode(win["decode"], base["decode"])
        if same:
            lines.append(
                f"The current sampler stays best: temperature {win['decode']['temperature']}, top-p {win['decode']['top_p']}, "
                f"repetition penalty {win['decode']['repetition_penalty']}, subtalker temperature {win['decode']['subtalker_temperature']}."
            )
        else:
            lines.append(
                f"Winning sampler: temperature {win['decode']['temperature']}, top-p {win['decode']['top_p']}, "
                f"repetition penalty {win['decode']['repetition_penalty']}, subtalker temperature {win['decode']['subtalker_temperature']}. "
                f"Primary score {_fmt(win.get('primary'), win.get('primary_min'), win.get('primary_max'))} "
                f"against current {_fmt(base.get('primary'), base.get('primary_min'), base.get('primary_max'))}."
            )
        better, worse, tied = [], [], []
        for part, dialects, names in (
            ("A", A_ORDER, PART_A_NAMES),
            ("B", PART_B_DIALECTS, B_NAMES),
        ):
            for dialect in dialects:
                left = _cell(win, part, dialect, "wer")
                right = _cell(base, part, dialect, "wer")
                if left[0] is None or right[0] is None:
                    continue
                label = f"Part {part} {names[dialect]}"
                if _overlap(left[1], left[2], right[1], right[2]):
                    tied.append(label)
                elif left[0] < right[0]:
                    better.append(f"{label} {right[0] - left[0]:.3f}")
                else:
                    worse.append(f"{label} {left[0] - right[0]:.3f}")
        lines.append("WER gains versus current: " + (", ".join(better) or "none") + ".")
        lines.append("WER got worse on: " + (", ".join(worse) or "none") + ". No real difference on " + (", ".join(tied) or "none") + ".")
    finished = [tag for tag in order if tag in docs]
    lines.append("Finished settings: " + ", ".join(finished) + ".")
    if stopped:
        lines.append(stopped)
    return lines


def _table(title, docs, order, part, dialect, baseline) -> list[str]:
    lines = [f"### {title}", ""]
    headers = ["setting", "WER", "letter", "dropped endings", "dropped words", "UTMOS", "same speaker", "broken"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    base = docs.get(baseline)
    for tag in order:
        summary = docs.get(tag)
        if summary is None:
            lines.append(f"| {tag} | missing |")
            continue
        cells = []
        for key in ("wer", "ler", "dropped_ending", "dropped_words", "utmos", "sim", "broken"):
            got = _cell(summary, part, dialect, key)
            ref = _cell(base, part, dialect, key) if tag != baseline else (None, None, None)
            mark = tag != baseline and _overlap(got[1], got[2], ref[1], ref[2])
            text = _fmt(*got)
            if mark and text:
                text += " no real difference"
            cells.append(text)
        lines.append("| " + " | ".join([tag, *cells]) + " |")
    lines.append("")
    return lines


def _cost_rows(tags, dollars_for, docs) -> tuple[list[str], float, float]:
    lines = ["| setting | dollars | GPU hours |", "| --- | ---: | ---: |"]
    total_d = 0.0
    total_h = 0.0
    for tag in tags:
        dollars = float(dollars_for.get(tag) if tag in dollars_for else (docs.get(tag) or {}).get("dollars") or 0)
        hours = float((docs.get(tag) or {}).get("gpu_wall_s") or 0) / 3600
        total_d += dollars
        total_h += hours
        lines.append(f"| {tag} | {dollars:.2f} | {hours:.2f} |")
    lines.append(f"| total | {total_d:.2f} | {total_h:.2f} |")
    return lines, total_d, total_h


def _cost(order, spend, prior, docs) -> list[str]:
    lines = ["## Cost", ""]
    paid = [tag for tag in order if tag in prior]
    fresh = [tag for tag in order if tag in spend]
    if paid:
        lines.append("Already paid, not counted against this $15 stop.")
        lines.append("")
        rows, _paid_d, _paid_h = _cost_rows(paid, prior, docs)
        lines.extend(rows)
        lines.append("")
    lines.append("This run. Budget $12, hard stop $15.")
    lines.append("")
    rows, _fresh_d, _fresh_h = _cost_rows(fresh, spend, docs)
    lines.extend(rows)
    lines.append("")
    lines.append("GPU hours are L40S container time. Dollars are those hours at $1.95.")
    lines.append("")
    return lines


def listen_members(exam: list[dict], docs_tags: list[str], seed: int = 13) -> list[tuple[str, str, int, str]]:
    """3 Part B clips per dialect, generation seed 1, the same clips for every tag."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for row in exam:
        if row["part"] == "B":
            by[row["dialect"]].append(row["id"])
    chosen = []
    for dialect in PART_B_DIALECTS:
        ids = sorted(set(by[dialect]))
        pick = ids if len(ids) <= 3 else rng.sample(ids, 3)
        for clip_id in sorted(pick):
            for tag in docs_tags:
                chosen.append((tag, dialect, 1, clip_id))
    return chosen


