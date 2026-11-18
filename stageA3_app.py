"""Plan 8 training. One epoch from step 16,000 into /data/runs/stageA3.

Does not call modal_app.py::train or ::e0. Does not write into /data/runs/stageA or stageA2.
"""
import csv
import io
import json
import os
import shlex
import signal
import subprocess
import threading
import time

import modal

from modal_app import DIALECTS, RECIPE, image as tts_image, vol
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

app = modal.App("qwen3-arabic-stageA3")
gpu_image = tts_image.add_local_python_source("modal_app")

INIT = f"{_DATA}/runs/stageA/step-016000-base"
OUT = f"{_DATA}/runs/stageA3"
TRAIN = f"{_DATA}/tts_stageA/stageA3/train_codes.jsonl"
VAL = f"{_DATA}/tts_stageA/stageA3/a2_val_codes.jsonl"
EXTRA = f"{_DATA}/tts_stageA/stageA3/uae100k_val_codes.jsonl"
SMOKE = f"{_DATA}/runs/stageA3/smoke"


def _watch(log_path, proc, pause_for_guard):
    """Stop on sub > 6, or when 3 val-main evals in a row sit more than 0.01 above the lowest so far.

    The floor includes Stage A2 val main losses already in the log. The streak starts fresh.
    A save of step-004000 pauses the run only when pause_for_guard is set.
    """
    import re

    reason = {"why": None, "detail": ""}
    step = 0
    lowest = None
    streak = 0
    seen = 0
    sub_re = re.compile(r"\bsub ([0-9]+\.[0-9]+)")
    step_re = re.compile(r"\bstep (\d+)/")
    val_re = re.compile(r"val main loss ([0-9]+\.[0-9]+)")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            prior = f.read()
            seen = f.tell()
        for line in prior.splitlines():
            if "val extra" in line:
                continue
            m = val_re.search(line)
            if m:
                v = float(m.group(1))
                lowest = v if lowest is None else min(lowest, v)
        if lowest is not None:
            print(f"WATCH floor {lowest:.4f}", flush=True)

    def parse(chunk):
        nonlocal step, lowest, streak
        for line in chunk.splitlines():
            m = step_re.search(line)
            if m:
                step = int(m.group(1))
            m = sub_re.search(line)
            if m and float(m.group(1)) > 6:
                return "stopped", f"sub {m.group(1)} at step {step}"
            if "val extra" in line:
                continue
            m = val_re.search(line)
            if m:
                v = float(m.group(1))
                if lowest is not None and v > lowest + 0.01:
                    streak += 1
                else:
                    streak = 0
                if lowest is None or v < lowest:
                    lowest = v
                if streak >= 3:
                    return "stopped", (
                        f"val {v:.4f} more than 0.01 above lowest {lowest:.4f} "
                        f"for 3 evals at step {step}"
                    )
            if pause_for_guard and "saved step-004000" in line:
                return "paused", "saved step-004000"
        return None, ""

    while proc.poll() is None:
        time.sleep(10)
        if not os.path.exists(log_path):
            continue
        with open(log_path, encoding="utf-8", errors="replace") as f:
            f.seek(seen)
            chunk = f.read()
            seen = f.tell()
        why, detail = parse(chunk)
        if why:
            reason["why"] = why
            reason["detail"] = detail
            print("WATCH", detail, flush=True)
            if why == "paused":
                time.sleep(15)
            try:
                vol.commit()
            except Exception as e:
                print("commit before stop failed", e, flush=True)
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return reason
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            f.seek(seen)
            why, detail = parse(f.read())
        if why == "stopped":
            reason["why"] = why
            reason["detail"] = detail
            print("WATCH", detail, flush=True)
    return reason


def _run(n_gpu, train_jsonl, out, pause_for_guard, diac_prob):
    import torch

    vol.reload()
    if not os.path.isdir(INIT):
        raise SystemExit(f"missing start checkpoint {INIT}")
    for path in (train_jsonl, VAL):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
    n = torch.cuda.device_count()
    if n != n_gpu:
        raise SystemExit(f"expected {n_gpu} GPUs, found {n}")
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "train.log")
    extra = f"--extra_val_jsonl {EXTRA} " if os.path.exists(EXTRA) else ""
    multi = "--multi_gpu " if n > 1 else ""
    inner = (
        f"cd {RECIPE} && "
        f"accelerate launch {multi}--num_processes {n} --num_machines 1 "
        f"--mixed_precision bf16 --dynamo_backend no train_dialects.py "
        f"--init_model_path {INIT} --dialects {DIALECTS} "
        f"--train_jsonl {train_jsonl} --val_jsonl {VAL} {extra}"
        f"--output_dir {out} --lr 5e-6 --epochs 1 --batch_size 4 --grad_accum 1 "
        f"--eval_every 500 --save_every 2000 --resume 1 --export_speakers 0 "
        f"--id_dropout 0.1 --warmup_frac 0.03 --sub_talker_weight 0.3 "
        f"--diac_prob {diac_prob} "
        f"2>&1 | tee -a {log_path}"
    )
    print("+", inner, flush=True)
    proc = subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", inner],
        cwd=RECIPE,
        start_new_session=True,
    )
    stop_commit = threading.Event()

    def loop():
        while not stop_commit.wait(300):
            try:
                vol.commit()
            except Exception as e:
                print("volume commit failed (will retry):", e, flush=True)

    threading.Thread(target=loop, daemon=True).start()
    reason = _watch(log_path, proc, pause_for_guard)
    rc = proc.wait()
    stop_commit.set()
    vol.commit()
    if reason.get("why") == "stopped":
        raise SystemExit(reason.get("detail") or "stop rule")
    if reason.get("why") == "paused":
        return "paused"
    if rc not in (0, -15, 143):
        raise RuntimeError(f"train failed ({rc})")
    return "done"


def _subset(src, dest, n=800, seed=0):
    import random

    if os.path.exists(dest):
        with open(dest, encoding="utf-8") as f:
            if sum(1 for line in f if line.strip()) == n:
                print(f"subset exists {dest}", flush=True)
                return
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randrange(seen)
                if j < n:
                    reservoir[j] = line
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as out:
        out.writelines(reservoir)
    print(f"subset {len(reservoir)} from {seen}", flush=True)


@app.function(
    image=gpu_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=3 * 3600,
    volumes={"/data": vol},
)
def smoke_run():
    vol.reload()
    dest = os.path.join(SMOKE, "subset.jsonl")
    _subset(TRAIN, dest)
    vol.commit()
    return _run(1, dest, SMOKE, pause_for_guard=0, diac_prob=0.5)


@app.function(
    image=gpu_image,
    gpu="H100:8",
    region=["us", "eu"],
    cpu=64,
    memory=393216,
    timeout=18 * 3600,
    volumes={"/data": vol},
)
def train_stagea3(pause_for_guard: int = 1):
    return _run(8, TRAIN, OUT, pause_for_guard=bool(pause_for_guard), diac_prob=0.5)


@app.function(
    image=gpu_image,
    gpu="L40S",
    cpu=8,
    memory=65536,
    timeout=3600,
    volumes={"/data": vol},
)
def guard4000():
    """Ten short sentences from step 4,000. No ASR. Duration and the wavs only."""
    import zlib
    from pathlib import Path

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    vol.reload()
    ckpt = f"{OUT}/step-004000-base"
    dest = Path(OUT) / "guard-step004000"
    dest.mkdir(parents=True, exist_ok=True)
    want = ["ar_uae", "ar_gulf", "ar_sau", "ar_egy", "ar_msa"]
    picked = {d: [] for d in want}
    with open(f"{_DATA}/tts_stageA/stageA2/eval.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            dialect = row.get("dialect")
            if row.get("kind") != "short" or dialect not in picked or len(picked[dialect]) >= 2:
                continue
            if row.get("text") and row.get("ref_audio"):
                picked[dialect].append(row)
    rows = [row for dialect in want for row in picked[dialect]]
    if len(rows) != 10:
        raise SystemExit(f"expected 10 short rows, found {len(rows)}")
    tts = Qwen3TTSModel.from_pretrained(ckpt, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    manifest = []
    for i, row in enumerate(rows):
        text = row["text"]
        chars = len(text.replace(" ", ""))
        cap_s = min(60.0, 2.5 * chars / 12.0 + 3.0)
        torch.manual_seed(zlib.crc32(f"{text}|{row['dialect']}|guard".encode()))
        wavs, sr = tts.generate_voice_clone(
            text=text, language=row["dialect"], ref_audio=row["ref_audio"],
            x_vector_only_mode=True, non_streaming_mode=True,
            max_new_tokens=int(cap_s * 12.5),
        )
        y = wavs[0]
        dur = len(y) / sr
        cps = chars / max(dur, 1e-3)
        path = dest / f"{i:02d}_{row['dialect']}.wav"
        sf.write(path, y, sr)
        item = {
            "file": path.name, "dialect": row["dialect"], "text": text,
            "dur": round(dur, 2), "cps": round(cps, 2),
            "cap_s": round(cap_s, 2), "near_cap": dur >= 0.95 * cap_s, "empty": dur < 0.3,
            "rate_out": not (5 <= cps <= 12),
        }
        manifest.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    vol.commit()
    bad = [it["file"] for it in manifest if it["empty"] or it["near_cap"] or it["rate_out"]]
    print(f"wrote {dest} bad {bad or 'none'}", flush=True)
    return "fail" if bad else "pass"


SCORE_CKPTS = [
    f"{_DATA}/runs/stageA/step-016000-base",
    f"{_DATA}/runs/stageA3/step-002000-base",
    f"{_DATA}/runs/stageA3/step-004000-base",
    f"{_DATA}/runs/stageA3/step-006000-base",
    f"{_DATA}/runs/stageA3/step-008000-base",
    f"{_DATA}/runs/stageA3/step-010000-base",
    f"{_DATA}/runs/stageA3/step-012000-base",
    f"{_DATA}/runs/stageA3/step-014000-base",
    f"{_DATA}/runs/stageA3/step-016000-base",
    f"{_DATA}/runs/stageA3/step-018000-base",
    f"{_DATA}/runs/stageA3/step-020000-base",
    f"{_DATA}/runs/stageA3/step-020408-base",
]
SCORE_DIALECTS = ["ar_uae", "ar_gulf", "ar_sau", "ar_egy", "ar_msa"]
SCORE_STORE = "qwen3-stageA3-val-scores"
A2_VAL = f"{_DATA}/tts_stageA/stageA3/a2_val_codes.jsonl"


def _tag(ckpt):
    parts = ckpt.rstrip("/").split("/")
    return parts[-2] + "_" + parts[-1]


def _col(ckpt):
    step = ckpt.rstrip("/").split("/")[-1].replace("step-", "").replace("-base", "")
    step = step.lstrip("0") or "0"
    parent = ckpt.rstrip("/").split("/")[-2]
    if parent == "stageA":
        return "A" + step
    return step


def _store():
    return modal.Dict.from_name(SCORE_STORE, create_if_missing=True)


def _run_score(ckpt, data, per_dialect, dialects):
    out = f"/tmp/{_tag(ckpt)}-{per_dialect}.csv"
    cmd = " ".join([
        "python", "val_by_dialect.py",
        "--ckpts", shlex.quote(ckpt),
        "--data", shlex.quote(data),
        "--per_dialect", str(per_dialect),
        "--seed", "0",
        "--dialects", dialects,
        "--out", shlex.quote(out),
    ])
    print("+", cmd, flush=True)
    rc = subprocess.call(cmd, shell=True, cwd=RECIPE)
    if rc:
        raise RuntimeError(f"val_by_dialect failed ({rc}) for {ckpt}")
    with open(out, encoding="utf-8") as f:
        return f.read()


def _csv_rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def _metric_map(payload, which, metric):
    out = {d: {} for d in SCORE_DIALECTS}
    counts = {d: None for d in SCORE_DIALECTS}
    prefix = set()
    for ckpt in SCORE_CKPTS:
        rows = _csv_rows(payload[_tag(ckpt)][which])
        col = _col(ckpt)
        for row in rows:
            d = row["dialect"]
            if d not in out:
                continue
            out[d][col] = float(row[metric])
            counts[d] = int(row["n"])
            prefix.add(row.get("prefix", ""))
    return out, counts, prefix


def _md_table(cells, counts, dialects=None):
    cols = [_col(c) for c in SCORE_CKPTS]
    header = "| dialect | n | " + " | ".join(cols) + " | 20408−A16000 |"
    rule = "|---|---:|" + "|".join(["---:"] * (len(cols) + 1)) + "|"
    lines = [header, rule]
    for d in (dialects or SCORE_DIALECTS):
        vals = cells[d]
        base = vals.get("A16000")
        last = vals.get("20408")
        delta = "" if base is None or last is None else f"{last - base:+.4f}"
        nums = ["" if vals.get(c) is None else f"{vals[c]:.4f}" for c in cols]
        n = "" if counts[d] is None else str(counts[d])
        lines.append(f"| {d} | {n} | " + " | ".join(nums) + f" | {delta} |")
    return "\n".join(lines)


def render_report(payload):
    main, n_main, prefix = _metric_map(payload, "a2", "main")
    sub, _, _ = _metric_map(payload, "a2", "sub")
    uae_main, uae_n, uae_prefix = _metric_map(payload, "uae", "main")
    uae_sub, _, _ = _metric_map(payload, "uae", "sub")
    moves = []
    for d in SCORE_DIALECTS:
        a, b = main[d].get("A16000"), main[d].get("20408")
        if a is None or b is None:
            continue
        moves.append(f"{d} {a:.4f} to {b:.4f} ({b - a:+.4f})")
    uae_line = ""
    ua, ub = uae_main["ar_uae"].get("A16000"), uae_main["ar_uae"].get("20408")
    if ua is not None and ub is not None:
        uae_line = f"UAE_100K main {ua:.4f} to {ub:.4f} ({ub - ua:+.4f})."
    n_uae = uae_n.get("ar_uae")
    body = "\n".join([
        "# Stage A3 — vowel epoch scored",
        "",
        f"One epoch finished at step 20,408. The start weights were `{_DATA}/runs/stageA/step-016000-base`. The last checkpoint is `{_DATA}/runs/stageA3/step-020408-base`. Pooled Stage A2 validation loss during training ended at 1.366.",
        "",
        "These tables are teacher-forced loss on the plain `text` field. `val_by_dialect.py` keeps vowel text off (`diac_prob` 0). Lower is better. This is not a listen and not an Audar score.",
        "",
        f"Stage A2 val uses `{_DATA}/tts_stageA/stageA3/a2_val_codes.jsonl`, 150 clips per priority dialect, seed 0. The same draw is used for every checkpoint. Priority dialects: {', '.join(SCORE_DIALECTS)}.",
        "",
        "## Main loss",
        "",
        _md_table(main, n_main),
        "",
        "Step 20,408 against the step 16,000 baseline: " + "; ".join(moves) + ".",
        "",
        "## Sub loss",
        "",
        _md_table(sub, n_main),
        "",
        f"## UAE_100K val",
        "",
        f"`{_DATA}/tts_stageA/stageA3/uae100k_val_codes.jsonl`, scored on its own. Plain text, seed 0, up to 300 clips. Rows scored: {n_uae}.",
        "",
        "### Main",
        "",
        _md_table(uae_main, uae_n, ["ar_uae"]),
        "",
        "### Sub",
        "",
        _md_table(uae_sub, uae_n, ["ar_uae"]),
        "",
        uae_line,
        "",
        f"Dialect-id prefix on the Stage A2 table: {', '.join(sorted(prefix)) or 'none'}. On the UAE table: {', '.join(sorted(uae_prefix)) or 'none'}.",
        "",
        "Train is 653,064 clips, 1,517.21 h. Stage A2 val is 4,208 clips, 10.17 h. `uae100k_val` is 300 clips, 0.55 h. The other 9,335 kept UAE validation clips are unused.",
        "",
        "Plan 5 is unchanged. Listening page: https://tausifiqbal10--qwen3-arabic-plan5-serve.us-east.modal.run . Private results: https://tausifiqbal10--qwen3-arabic-plan5-serve.us-east.modal.run/results/c7e1a94b60d2835f . No one was scored again here.",
        "",
    ])
    return body


def _combined_csv(payload, which):
    buf = io.StringIO()
    writer = None
    for ckpt in SCORE_CKPTS:
        rows = _csv_rows(payload[_tag(ckpt)][which])
        if writer is None and rows:
            writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return buf.getvalue()


@app.function(
    image=gpu_image,
    gpu="L40S",
    cpu=8,
    memory=65536,
    timeout=4 * 3600,
    volumes={"/data": vol},
    max_containers=10,
    retries=1,
)
def score_one(ckpt: str):
    """One checkpoint, plain text. Reads the volume and does not commit it."""
    tag = _tag(ckpt)
    store = _store()
    if store.contains(tag):
        print(f"cached {tag}", flush=True)
        return tag
    vol.reload()
    print(f"scoring {ckpt} plain text seed 0", flush=True)
    a2 = _run_score(ckpt, A2_VAL, 150, ",".join(SCORE_DIALECTS))
    uae = _run_score(ckpt, EXTRA, 300, "ar_uae")
    store.put(tag, {"a2": a2, "uae": uae})
    print(f"stored {tag}", flush=True)
    return tag


def _persist_report(body, files):
    vol.reload()
    root = f"{_DATA}/runs/stageA3/val_by_dialect"
    os.makedirs(root, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(root, name), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {root}/{name}", flush=True)
    path = f"{_DATA}/runs/stageA3/report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    vol.commit()
    print(f"wrote {path}", flush=True)


@app.function(image=gpu_image, cpu=1, memory=2048, timeout=6 * 3600, volumes={"/data": vol})
def finish_scores():
    """One cloud call owns the fan-out, then writes the report once."""
    tags = list(score_one.map(SCORE_CKPTS))
    store = _store()
    payload = {tag: store.get(tag) for tag in tags}
    missing = [tag for tag, row in payload.items() if not row]
    if missing:
        raise RuntimeError(f"missing scores: {missing}")
    body = render_report(payload)
    files = {
        "a2_val.csv": _combined_csv(payload, "a2"),
        "uae100k_val.csv": _combined_csv(payload, "uae"),
    }
    _persist_report(body, files)
    return body


@app.local_entrypoint()
def score():
    body = finish_scores.remote()
    print(body, flush=True)
    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "reports", "latest_update.md")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"wrote {dest}", flush=True)


@app.local_entrypoint()
def smoke():
    first = smoke_run.remote()
    print("smoke first", first, flush=True)
    second = smoke_run.remote()
    print("smoke second", second, flush=True)


@app.local_entrypoint()
def resume():
    print(train_stagea3.remote(pause_for_guard=0), flush=True)


@app.local_entrypoint()
def main():
    status = train_stagea3.remote(pause_for_guard=1)
    print("train", status, flush=True)
    if status != "paused":
        return
    verdict = guard4000.remote()
    print("guard", verdict, flush=True)
    if verdict != "pass":
        raise SystemExit("guard failed; epoch not resumed")
    rest = train_stagea3.remote(pause_for_guard=0)
    print("train resumed", rest, flush=True)
