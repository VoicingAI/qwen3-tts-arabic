"""Plan 14 training and scoring. Writes only under /data/exp_11labs/.

Starts from step 45,190, which is read only. Every checkpoint gets a TAG file
containing internal_11labs. Does not call ElevenLabs and does not touch Stage 2.
Launch only after the step and cost report is approved.
"""
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import modal

from modal_app import RECIPE, image as tts_image
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

app = modal.App("qwen3-arabic-plan14-train")
vol = modal.Volume.from_name("qwen3-arabic-tts-v2")
secret = modal.Secret.from_name("huggingface")
STOP = modal.Dict.from_name("plan14-train-stop", create_if_missing=True)

train_image = tts_image.add_local_python_source("modal_app")
gen_image = tts_image.add_local_python_source("modal_app", "plan13_eval")
score_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.6.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=5.4.0", "accelerate", "soundfile", "librosa", "numpy",
        "sentencepiece", "protobuf", "huggingface_hub", "rapidfuzz", "safetensors",
    )
    .env({"HF_HOME": "/data/hf_cache"})
    .add_local_python_source("modal_app", "plan13_eval", "plan12_text")
)
cpu_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("modal_app")

ROOT = Path(f"{_DATA}/exp_11labs")
OUT = ROOT / "runs"
EVAL = ROOT / "eval"
TRAIN = str(ROOT / "train_codes.jsonl")
VAL = str(ROOT / "val_codes.jsonl")
VAL_MANIFEST = ROOT / "manifest" / "val.jsonl"
INIT = f"{_DATA}/runs/stageB1/step-045190-base"
NOISE_BASE = Path(f"{_DATA}/plan13/eval_noise")
REAL_SUMMARY = Path(f"{_DATA}/plan13/eval/a3-020408/summary.json")
DIALECTS = "ar_msa,ar_egy,ar_sau,ar_irq,ar_tun,ar_mar,ar_omn,ar_jor,ar_syr,ar_sdn"
PRIORITY = ("ar_egy", "ar_msa", "ar_sau", "ar_uae")
TAG = "internal_11labs"
SAVE_EVERY = 1370
SAVE_AT = (2741, 4000, 5482, 8223)
SCORE_STEPS = (1370, 2740, 2741, 4110, 5480, 5482, 6850, 8220, 8223)
EARLY = (1370, 2740)
PASS_STEP = 2741
H100_RATE = 3.95
A100_RATE = 2.50
L40S_RATE = 1.95
CAP = 60.0
PAUSE_AT = 72.0
FLAGS = ROOT / "flags.json"


def _commit():
    try:
        vol.commit()
    except Exception as exc:
        print("commit", exc, flush=True)


def _reload():
    try:
        vol.reload()
    except Exception as exc:
        print("reload", exc, flush=True)


def _read(path, default=None):
    path = Path(path)
    if not path.is_file():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {} if default is None else default


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _flags():
    return _read(FLAGS, {})


def _write_flags(payload):
    _write(FLAGS, payload)


def _stopped():
    try:
        return bool(STOP["stop"])
    except Exception:
        return False


def _encode_spent():
    return float(_read(ROOT / "encode.json", {}).get("gpu_dollars") or 0)


def _train_spent():
    doc = _read(OUT / "spend_train.json", {})
    gpus = int(doc.get("gpus") or 8)
    rate = float(doc.get("rate") or H100_RATE)
    return float(doc.get("total_s") or 0) / 3600 * gpus * rate


def _trial_spent():
    return float(_read(ROOT / "trial" / "spend.json", {}).get("dollars") or 0)


def _combined():
    flags = _flags()
    return _encode_spent() + _train_spent() + _trial_spent() + float(flags.get("eval_spent") or 0)


def _tag_tree():
    if not OUT.is_dir():
        return
    for path in OUT.glob("step-*-base"):
        if not path.is_dir():
            continue
        dest = path / "TAG"
        if not dest.is_file():
            dest.write_text(TAG + "\n", encoding="utf-8")


def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _load_jsonl(path):
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _score_rows(folder):
    rows = []
    folder = Path(folder)
    for shard in (0, 1):
        rows.extend(_load_jsonl(folder / f"score_{shard}.jsonl"))
    return rows


def _priority(rows):
    """Unweighted mean of Egyptian, MSA, Saudi and Emirati, averaged across seeds."""
    by_seed = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("wer") is None or row.get("seed") is None:
            continue
        by_seed[int(row["seed"])][row.get("dialect")].append(float(row["wer"]))
    seed_priority = []
    for seed in sorted(by_seed):
        vals = [_mean(by_seed[seed][dialect]) for dialect in PRIORITY]
        if any(v is None for v in vals):
            continue
        seed_priority.append(sum(vals) / len(vals))
    if not seed_priority:
        return None, []
    return sum(seed_priority) / len(seed_priority), seed_priority


def _by_dialect(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("wer") is None:
            continue
        grouped[row.get("dialect") or "?"].append(float(row["wer"]))
    return {dialect: {"n": len(vals), "wer": _mean(vals)} for dialect, vals in sorted(grouped.items())}


def _watch(log_path, proc):
    """Sub-loss above 6, NaN, the val stop after step 2,000, and the step-4,000 pause."""
    reason = {"why": None, "detail": ""}
    step = 0
    lowest = None
    streak = 0
    seen = 0
    sub_re = re.compile(r"\bsub ([0-9]+\.[0-9]+)")
    step_re = re.compile(r"\bstep (\d+)/")
    val_re = re.compile(r"val main loss ([0-9]+\.[0-9]+)")
    bad_re = re.compile(r"\b(loss|main|sub) (nan|inf|-inf)\b", re.I)
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            prior = handle.read()
            seen = handle.tell()
        for line in prior.splitlines():
            matched = step_re.search(line)
            if matched:
                step = int(matched.group(1))
            if "val extra" in line:
                continue
            matched = val_re.search(line)
            if matched:
                value = float(matched.group(1))
                lowest = value if lowest is None else min(lowest, value)

    def parse(chunk):
        nonlocal step, lowest, streak
        for line in chunk.splitlines():
            matched = re.search(r"saved (step-\d+)", line)
            if matched:
                ckpt = OUT / f"{matched.group(1)}-base"
                if ckpt.is_dir():
                    (ckpt / "TAG").write_text(TAG + "\n", encoding="utf-8")
            matched = step_re.search(line)
            if matched:
                step = int(matched.group(1))
            if bad_re.search(line):
                return "stopped", f"nan or inf at step {step}"
            matched = sub_re.search(line)
            if matched and float(matched.group(1)) > 6:
                return "stopped", f"sub {matched.group(1)} at step {step}"
            if "val extra" in line or "val main loss" not in line:
                if "saved step-004000" in line:
                    return "paused", "saved step-004000"
                continue
            matched = val_re.search(line)
            if not matched:
                continue
            value = float(matched.group(1))
            if step >= 2000:
                if lowest is not None and value > lowest + 0.01:
                    streak += 1
                else:
                    streak = 0
                if streak >= 3:
                    return "stopped", (
                        f"val {value:.4f} more than 0.01 above lowest {lowest:.4f} "
                        f"for 3 evals at step {step}"
                    )
            if lowest is None or value < lowest:
                lowest = value
        return None, ""

    while proc.poll() is None:
        time.sleep(10)
        if _stopped():
            try:
                why = str(STOP["why"])
            except Exception:
                why = "stop flag"
            reason["why"] = "stopped"
            reason["detail"] = why
            print("WATCH", why, flush=True)
            _commit()
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return reason
        if not os.path.exists(log_path):
            continue
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            handle.seek(seen)
            chunk = handle.read()
            seen = handle.tell()
        why, detail = parse(chunk)
        if why:
            reason["why"] = why
            reason["detail"] = detail
            print("WATCH", detail, flush=True)
            if why == "paused":
                time.sleep(15)
            _commit()
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return reason
    return reason


def _beat(log_path, started, stop_event, gpus, rate, batch, accum, gpu_name):
    while not stop_event.wait(60):
        try:
            _reload()
            text = ""
            if os.path.exists(log_path):
                with open(log_path, "rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 80000))
                    text = handle.read().decode("utf-8", "replace")
            step = None
            for line in text.splitlines():
                matched = re.search(r"step (\d+)/(\d+)", line)
                if matched:
                    step = int(matched.group(1))
            spend_path = OUT / "spend_train.json"
            base = float(_read(spend_path).get("base_s") or 0)
            train_s = base + (time.time() - started)
            _write(spend_path, {
                "base_s": base, "total_s": train_s, "gpus": gpus, "rate": rate,
                "batch": batch, "accum": accum, "gpu_name": gpu_name,
            })
            _tag_tree()
            spent = _combined()
            print(json.dumps({"step": step, "train_s": round(train_s, 1), "combined": round(spent, 2)}), flush=True)
            if spent >= PAUSE_AT and not _stopped():
                STOP["stop"] = True
                STOP["why"] = f"spend ${spent:.2f} passed ${PAUSE_AT:.0f}"
            _commit()
        except Exception as exc:
            print("beat", exc, flush=True)


def _run_train(gpus, accum, batch=4, gpu_name="H100", rate=H100_RATE, max_steps=0, out_dir="", bill=True):
    import torch

    _reload()
    if not os.path.isdir(INIT):
        raise SystemExit(f"missing init checkpoint {INIT}")
    if not os.path.isfile(TRAIN) or not os.path.isfile(VAL):
        raise SystemExit("missing coded train or val rows")
    found = torch.cuda.device_count()
    if found != gpus:
        raise SystemExit(f"expected {gpus} GPUs, found {found}")
    rows = batch * accum * gpus
    if bill and rows != 32:
        raise SystemExit(f"expected 32 rows per step, found {rows}")
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if bill:
        spend_path = OUT / "spend_train.json"
        prev = _read(spend_path)
        _write(spend_path, {
            "base_s": float(prev.get("total_s") or prev.get("base_s") or 0),
            "total_s": float(prev.get("total_s") or 0),
            "gpus": gpus, "accum": accum, "batch": batch,
            "gpu_name": gpu_name, "rate": rate,
        })
        _commit()
    log_path = str(out / "train.log")
    save_at = ",".join(str(step) for step in SAVE_AT)
    step_arg = f"--max_steps {int(max_steps)} " if max_steps else ""
    resume = 0 if max_steps else 1
    if gpus == 1:
        launcher = "python -u train_dialects.py "
    else:
        launcher = (
            f"accelerate launch --multi_gpu --num_processes {gpus} --num_machines 1 "
            f"--mixed_precision bf16 --dynamo_backend no train_dialects.py "
        )
    inner = (
        f"cd {RECIPE} && {launcher}"
        f"--init_model_path {INIT} --dialects {DIALECTS} "
        f"--train_jsonl {TRAIN} --val_jsonl {VAL} "
        f"--require_split train --output_dir {out} --lr 2e-6 --epochs 3 "
        f"--batch_size {batch} --grad_accum {accum} --eval_every 500 --save_every {SAVE_EVERY} "
        f"--save_at {save_at} --resume {resume} --export_speakers 0 "
        f"--id_dropout 0.1 --warmup_frac 0.03 --sub_talker_weight 0.3 "
        f"--diac_prob 0.5 --lang_lr_mult 20 --fp32_master 1 {step_arg}"
        f"2>&1 | tee -a {log_path}"
    )
    print(
        f"SHAPE gpu={gpu_name} gpus={gpus} accum={accum} batch={batch} "
        f"rows_per_step={rows} lr=2e-6 max_steps={max_steps or 'full'}",
        flush=True,
    )
    print("+", inner, flush=True)
    proc = subprocess.Popen(["bash", "-o", "pipefail", "-c", inner], cwd=RECIPE, start_new_session=True)
    stop_beat = threading.Event()
    if bill:
        threading.Thread(
            target=_beat,
            args=(log_path, started, stop_beat, gpus, rate, batch, accum, gpu_name),
            daemon=True,
        ).start()
    reason = _watch(log_path, proc)
    rc = proc.wait()
    stop_beat.set()
    if not bill:
        elapsed = time.time() - started
        path = ROOT / "trial" / "spend.json"
        prev = _read(path, {})
        dollars = float(prev.get("dollars") or 0) + elapsed / 3600 * gpus * rate
        _write(path, {
            "dollars": round(dollars, 4), "last_s": round(elapsed, 1),
            "gpu_name": gpu_name, "gpus": gpus, "batch": batch, "accum": accum,
        })
    _tag_tree()
    _commit()
    if reason.get("why") == "paused":
        return "paused"
    if reason.get("why") == "stopped":
        return "stopped:" + (reason.get("detail") or "")
    if rc not in (0, -15, 143):
        raise RuntimeError(f"train failed ({rc})")
    return "done"


@app.function(
    image=train_image, gpu="H100:8", cpu=64, memory=393216, timeout=8 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def train_remote():
    return _run_train(8, 1)


@app.function(
    image=train_image, gpu="H100:4", cpu=32, memory=196608, timeout=8 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def train4():
    return _run_train(4, 2)


@app.function(
    image=train_image, gpu="H100:2", cpu=16, memory=98304, timeout=8 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def train2():
    return _run_train(2, 4)


@app.function(
    image=train_image, gpu="A100-80GB:8", cpu=32, memory=131072, timeout=8 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def train_a100():
    return _run_train(8, 1, batch=4, gpu_name="A100-80GB", rate=A100_RATE)


@app.function(
    image=train_image, gpu="L40S:8", cpu=32, memory=131072, timeout=10 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def train_l40s():
    return _run_train(8, 1, batch=4, gpu_name="L40S", rate=L40S_RATE)


@app.function(
    image=train_image, gpu="L40S:8", cpu=32, memory=131072, timeout=10 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def train_l40s_b2():
    return _run_train(8, 2, batch=2, gpu_name="L40S", rate=L40S_RATE)


@app.function(
    image=train_image, gpu="L40S", cpu=8, memory=65536, timeout=3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def trial_l40s(batch: int = 4, accum: int = 1):
    """20 optimizer steps on one L40S. Peak memory is the per-GPU batch."""
    return _run_train(
        1, accum, batch=batch, gpu_name="L40S", rate=L40S_RATE,
        max_steps=20, out_dir=str(ROOT / "trial"), bill=False,
    )


@app.function(
    image=train_image, gpu="L40S", cpu=8, memory=65536, timeout=3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def guard4000():
    """Ten short real sentences. Speaking rate 5–12 characters per second."""
    import zlib

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    _reload()
    ckpt = str(OUT / "step-004000-base")
    dest = OUT / "guard-step004000"
    dest.mkdir(parents=True, exist_ok=True)
    want = ["ar_uae", "ar_gulf", "ar_sau", "ar_egy", "ar_msa"]
    picked = {dialect: [] for dialect in want}
    with open(f"{_DATA}/tts_stageA/stageA2/eval.jsonl", encoding="utf-8") as handle:
        for line in handle:
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
            max_new_tokens=max(1, int(cap_s * 12.5)),
        )
        y = wavs[0]
        dur = len(y) / sr
        cps = chars / max(dur, 1e-3)
        path = dest / f"{i:02d}_{row['dialect']}.wav"
        sf.write(path, y, sr)
        item = {
            "file": path.name, "dialect": row["dialect"], "text": text,
            "dur": round(dur, 2), "cps": round(cps, 2), "cap_s": round(cap_s, 2),
            "near_cap": dur >= 0.95 * cap_s, "empty": dur < 0.3,
            "rate_out": not (5 <= cps <= 12),
        }
        manifest.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tag = Path(ckpt) / "TAG"
    if not tag.is_file():
        tag.write_text(TAG + "\n", encoding="utf-8")
    _commit()
    bad = [it["file"] for it in manifest if it["empty"] or it["near_cap"] or it["rate_out"]]
    return "fail" if bad else "pass"


def _val_rows():
    rows = []
    with VAL_MANIFEST.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({
                "i": i, "clip_id": row.get("clip_id"), "dialect": row.get("dialect"),
                "text": row.get("text") or "", "ref_audio": row.get("ref_audio") or row.get("audio"),
                "audio": row.get("audio"),
            })
    if len(rows) != 1024:
        raise RuntimeError(f"expected 1024 val rows, found {len(rows)}")
    return rows


@app.function(
    image=gen_image, gpu="L40S", cpu=8, memory=65536, timeout=4 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def gen_job(ckpt: str, tag: str, shard: int, kind: str):
    from plan13_eval import noise_generate

    _reload()
    if kind == "val":
        meta = noise_generate(
            ckpt, tag, shard, root=str(EVAL), seeds=(1,), rows_in=_val_rows(), budget_s=7200,
        )
    else:
        meta = noise_generate(ckpt, tag, shard, root=str(EVAL), budget_s=7200)
    _commit()
    return meta


@app.function(
    image=score_image, gpu="L40S", cpu=8, memory=65536, timeout=4 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol}, secrets=[secret],
)
def score_job(tag: str, shard: int):
    from plan13_eval import noise_score

    _reload()
    meta = noise_score(tag, shard, root=str(EVAL))
    _commit()
    return meta


def _poll(call):
    try:
        return "done", call.get(timeout=0)
    except TimeoutError:
        return "wait", None
    except Exception as exc:
        if exc.__class__.__name__ == "TimeoutError":
            return "wait", None
        return "error", exc


def _add_spent(amount):
    flags = _flags()
    flags["eval_spent"] = round(float(flags.get("eval_spent") or 0) + float(amount or 0), 4)
    _write_flags(flags)
    return flags


def _baseline(tag):
    rows = _score_rows(NOISE_BASE / tag)
    priority, seeds = _priority(rows)
    return {"tag": tag, "n": len(rows), "priority_wer": priority, "seeds": seeds, "by_dialect": _by_dialect(rows)}


def _real_row():
    doc = _read(REAL_SUMMARY, {})
    cells = {}
    for dialect, cell in (doc.get("by_dialect") or {}).items():
        if cell.get("ref_wer") is not None:
            cells[dialect] = {"n": cell.get("n"), "wer": cell.get("ref_wer")}
    vals = [cells[d]["wer"] for d in PRIORITY if d in cells]
    priority = sum(vals) / len(vals) if len(vals) == len(PRIORITY) else None
    return {"priority_wer": priority, "by_dialect": cells, "source": str(REAL_SUMMARY)}


def _fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _rel(new, base):
    if new is None or base is None or base <= 0:
        return None
    return (new - base) / base


def _shape_line():
    doc = _read(OUT / "run.json", {})
    gpus = int(doc.get("gpus") or 0)
    accum = int(doc.get("accum") or 0)
    gpu_name = doc.get("gpu_name") or "H100"
    batch = int(doc.get("batch_size") or 4)
    rate = float(doc.get("rate") or H100_RATE)
    if not gpus:
        return "No region pin. H100 at $3.95 per GPU-hour. L40S scoring at $1.95."
    return (
        f"No region pin. {gpus} {gpu_name}, batch {batch}, accumulation {accum}, "
        f"{batch * accum * gpus} rows per step, ${rate:.2f} per GPU-hour. "
        "L40S scoring at $1.95."
    )


def write_report(why):
    _reload()
    flags = _flags()
    base = _baseline("b1-045190")
    older = _baseline("a3-020408")
    real = _real_row()
    summaries = []
    if EVAL.is_dir():
        for path in sorted(EVAL.glob("p14-*/summary.json")):
            if path.parent.name.endswith("-val"):
                continue
            doc = _read(path, {})
            if doc.get("tag") and doc.get("kind") != "val":
                summaries.append(doc)
    lines = [
        "# Plan 14",
        "",
        why,
        "",
        "Internal only. Checkpoints and this data stay out of Stage 2, the polish set, and any shipping run.",
        "Started from step 45,190, read only. Learning rate 2e-6, cosine, 3% warmup.",
        _shape_line(),
        "ElevenLabs was not called. Testers were not contacted.",
        "",
        f"Encode ${_encode_spent():.2f}. Trial ${_trial_spent():.2f}. Train ${_train_spent():.2f}. Scoring ${float(flags.get('eval_spent') or 0):.2f}.",
        f"Combined ${_combined():.2f}. Cap ${CAP:.0f}. Pause ${PAUSE_AT:.0f}.",
        "",
        "## Step 45,190 and the real recordings",
        "",
        f"Step 45,190 priority word error {_fmt(base.get('priority_wer'))} "
        f"across seeds {', '.join(_fmt(v) for v in base.get('seeds') or [])}. "
        "Read from the existing noise study. Not regenerated.",
        f"Step 20,408 priority word error {_fmt(older.get('priority_wer'))}. "
        "Reported from one full pass on. It does not stop the run.",
        f"Real recordings priority word error {_fmt(real.get('priority_wer'))}.",
        "",
        "## Checkpoints",
        "",
    ]
    for doc in summaries:
        rel_now = _rel(doc.get("priority_wer"), base.get("priority_wer"))
        rel_old = _rel(doc.get("priority_wer"), older.get("priority_wer"))
        old_note = ""
        if int(doc.get("step") or 0) >= PASS_STEP:
            old_note = f" Against step 20,408 {_fmt(rel_old)}."
        lines.append(
            f"Step {doc.get('step')} priority {_fmt(doc.get('priority_wer'))} "
            f"({', '.join(_fmt(v) for v in doc.get('seeds') or [])}). "
            f"Against step 45,190 {_fmt(rel_now)}.{old_note} "
            f"Val {_fmt((doc.get('val') or {}).get('wer'))}."
        )
        for dialect, cell in (doc.get("by_dialect") or {}).items():
            lines.append(f"  {dialect} n {cell.get('n')} wer {_fmt(cell.get('wer'))}")
    skipped = flags.get("skipped_val") or []
    if skipped:
        lines.append("")
        lines.append("Val scores skipped so the total would stay at the cap: " + ", ".join(str(s) for s in skipped) + ".")
    decision = flags.get("early") or {}
    lines += ["", "## Early stop", ""]
    if decision.get("pairs"):
        for pair in decision["pairs"]:
            lines.append(
                f"Step {pair['step']} is {pair['rel']:+.1%} against step 45,190 "
                f"({pair['new']:.3f} vs {pair['base']:.3f})."
            )
        lines.append("Stop fired. Both of the first two checkpoints are more than 10% worse." if decision.get("stop") else "Stop did not fire.")
    else:
        lines.append("The first two checkpoints were not both scored.")
    if flags.get("guard_fail"):
        lines.append("The step-4,000 speaking-rate guard failed.")
    elif flags.get("guard") == "pass":
        lines.append("The step-4,000 speaking-rate guard passed.")
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote report", why, flush=True)


def _publish(tag, step, kind):
    folder = EVAL / tag
    rows = _score_rows(folder)
    priority, seeds = _priority(rows)
    summary_path = folder / "summary.json"
    if kind == "val":
        wers = [float(row["wer"]) for row in rows if row.get("wer") is not None]
        doc = {
            "tag": tag, "step": int(step), "kind": "val", "n": len(rows),
            "wer": _mean(wers), "by_dialect": _by_dialect(rows),
        }
        _write(summary_path, doc)
        return doc
    doc = {
        "tag": tag, "step": int(step), "kind": "noise", "n": len(rows),
        "priority_wer": priority, "seeds": seeds, "by_dialect": _by_dialect(rows),
        "ckpt": str(OUT / f"step-{int(step):06d}-base"),
    }
    _write(summary_path, doc)
    return doc


def _early(base_priority, summaries):
    pairs = []
    by_step = {int(doc["step"]): doc for doc in summaries.values() if doc.get("kind") == "noise"}
    for step in EARLY:
        doc = by_step.get(step)
        if not doc or doc.get("priority_wer") is None or base_priority is None:
            return None
        rel = _rel(doc["priority_wer"], base_priority)
        if rel is None:
            return None
        pairs.append({"step": step, "new": doc["priority_wer"], "base": base_priority, "rel": rel})
    return {"stop": all(item["rel"] > 0.10 for item in pairs), "pairs": pairs}


@app.function(
    image=cpu_image, cpu=2, memory=8192, timeout=24 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def run(adopt_id: str = "", gpus: int = 8, accum: int = 1, gpu_name: str = "H100", batch: int = 4, rate: float = H100_RATE):
    why = "finished"
    try:
        why = _orchestrate(adopt_id, gpus, accum, gpu_name, batch, rate) or "finished"
    except Exception as exc:
        why = "error: " + str(exc)[:800]
        print(why, flush=True)
    try:
        write_report(why)
        _commit()
    except Exception as exc:
        print("report failed", exc, flush=True)
    return why


def _spawn_train(gpus, gpu_name="H100", batch=4):
    if gpu_name == "A100-80GB" and gpus == 8:
        return train_a100.spawn()
    if gpu_name == "L40S" and gpus == 8 and batch == 2:
        return train_l40s_b2.spawn()
    if gpu_name == "L40S" and gpus == 8:
        return train_l40s.spawn()
    if gpu_name == "H100" and gpus == 4:
        return train4.spawn()
    if gpu_name == "H100" and gpus == 2:
        return train2.spawn()
    if gpu_name == "H100" and gpus == 8:
        return train_remote.spawn()
    raise RuntimeError(f"unsupported shape {gpus} {gpu_name} batch {batch}")


def _orchestrate(adopt_id="", gpus=8, accum=1, gpu_name="H100", batch=4, rate=H100_RATE):
    _reload()
    if not os.path.isfile(TRAIN):
        raise RuntimeError("coded rows are not ready")
    if int(batch) * int(accum) * int(gpus) != 32:
        raise RuntimeError(f"{gpus} {gpu_name} batch {batch} accum {accum} is not 32 rows per step")
    STOP["stop"] = False
    STOP["why"] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "run.json", {
        "init": INIT, "lr": 2e-6, "warmup_frac": 0.03, "dialects": DIALECTS,
        "tag": TAG, "cap": CAP, "pause_at": PAUSE_AT, "region": None,
        "gpus": int(gpus), "accum": int(accum), "batch_size": int(batch),
        "rows_per_step": 32, "gpu_name": gpu_name, "rate": float(rate),
        "steps": 8223,
        "save_every": SAVE_EVERY, "save_at": list(SAVE_AT), "score_steps": list(SCORE_STEPS),
    })
    flags = {
        "eval_spent": 0, "early_stop": False, "early": {}, "guard": "",
        "guard_fail": False, "skipped_val": [], "errors": [],
    }
    _write_flags(flags)
    _commit()
    if adopt_id:
        train_call = modal.FunctionCall.from_id(adopt_id)
        print("ADOPT", adopt_id, "gpus", gpus, "accum", accum, flush=True)
    else:
        train_call = _spawn_train(int(gpus), gpu_name, int(batch))
        print("TRAIN", train_call.object_id, gpu_name, "gpus", gpus, "batch", batch, "accum", accum, flush=True)
    base = None
    summaries = {}
    seen = set()
    queue = []
    active = []
    train_done = False
    guard_done = False
    why = "finished"
    early_fired = False

    def room():
        return 1 if not train_done else 2

    def spend():
        _reload()
        return _combined()

    while True:
        if train_call is not None and not train_done:
            state, value = _poll(train_call)
            if state == "done":
                text = str(value)
                print("train returned", text, flush=True)
                if text == "paused" and not guard_done:
                    result = guard4000.remote()
                    guard_done = True
                    flags = _flags()
                    flags["guard"] = result
                    if result != "pass":
                        flags["guard_fail"] = True
                        _write_flags(flags)
                        STOP["stop"] = True
                        STOP["why"] = "step 4000 guard failed"
                        train_done = True
                        why = "guard failed"
                        train_call = None
                    else:
                        _write_flags(flags)
                        train_call = _spawn_train(int(gpus), gpu_name, int(batch))
                        print("TRAIN resume", train_call.object_id, gpu_name, "gpus", gpus, flush=True)
                    _commit()
                else:
                    train_done = True
                    train_call = None
                    if text.startswith("stopped:"):
                        why = text
                    elif text == "done":
                        why = "training finished the planned passes"
                    elif text != "paused":
                        why = text
            elif state == "error":
                train_done = True
                train_call = None
                why = "train error: " + str(value)[:500]

        if base is None:
            _reload()
            base = _baseline("b1-045190")
            if base.get("priority_wer") is None:
                raise RuntimeError("step 45,190 noise-study scores are missing")
            print("baseline", json.dumps({"priority": base["priority_wer"], "seeds": base["seeds"]}), flush=True)

        if not _stopped() and spend() < PAUSE_AT:
            _reload()
            _tag_tree()
            for step in SCORE_STEPS:
                if step in seen:
                    continue
                ckpt = OUT / f"step-{step:06d}-base"
                if (ckpt / "model.safetensors").is_file():
                    seen.add(step)
                    queue.append(step)

        still = []
        for job in active:
            states = [_poll(call) for call in job["calls"]]
            if any(state == "wait" for state, _value in states):
                still.append(job)
                continue
            if any(state == "error" for state, _value in states):
                err = "; ".join(str(value)[:200] for state, value in states if state == "error")
                flags = _flags()
                flags["errors"] = (flags.get("errors") or []) + [f"{job['tag']} {err}"]
                _write_flags(flags)
                print("score error", job["tag"], err[:300], flush=True)
                continue
            metas = [value for _state, value in states]
            spent = sum(float((meta or {}).get("dollars") or 0) for meta in metas)
            _add_spent(spent)
            _reload()
            if job["phase"] == "gen":
                if not all((meta or {}).get("done") for meta in metas):
                    flags = _flags()
                    flags["errors"] = (flags.get("errors") or []) + [f"{job['tag']} generation incomplete"]
                    _write_flags(flags)
                    continue
                job["phase"] = "score"
                job["calls"] = [score_job.spawn(job["tag"], shard) for shard in (0, 1)]
                still.append(job)
                print("score", job["tag"], flush=True)
                continue
            doc = _publish(job["tag"], job["step"], job["kind"])
            if job["kind"] == "noise":
                summaries[job["step"]] = doc
                parent = EVAL / f"p14-{job['step']:06d}" / "summary.json"
                full = _read(parent, {})
                full.update(doc)
                _write(parent, full)
                print("priority", job["step"], doc.get("priority_wer"), flush=True)
            else:
                parent = EVAL / f"p14-{job['step']:06d}" / "summary.json"
                full = _read(parent, {})
                full["val"] = {"wer": doc.get("wer"), "n": doc.get("n"), "by_dialect": doc.get("by_dialect")}
                _write(parent, full)
                if job["step"] in summaries:
                    summaries[job["step"]]["val"] = full["val"]
            decision = _early(base.get("priority_wer"), summaries)
            if decision:
                flags = _flags()
                flags["early"] = decision
                if decision.get("stop") and not early_fired:
                    early_fired = True
                    flags["early_stop"] = True
                    text = "; ".join(
                        f"step {p['step']} is {p['rel']:+.1%} vs 45,190" for p in decision["pairs"]
                    )
                    STOP["stop"] = True
                    STOP["why"] = "early stop " + text
                    why = "early stop " + text
                    print(STOP["why"], flush=True)
                _write_flags(flags)
            write_report(why)
            _commit()

        active = still
        while queue and len(active) < room() and not _stopped() and spend() < PAUSE_AT:
            step = queue.pop(0)
            ckpt = str(OUT / f"step-{step:06d}-base")
            tag = f"p14-{step:06d}"
            active.append({
                "step": step, "tag": tag, "kind": "noise", "phase": "gen",
                "calls": [gen_job.spawn(ckpt, tag, shard, "noise") for shard in (0, 1)],
            })
            print("gen", tag, flush=True)
        flags = _flags()
        val_started = set(flags.get("val_started") or [])
        for step in list(summaries):
            if step in val_started or step in (flags.get("skipped_val") or []):
                continue
            if any(job["step"] == step and job["kind"] == "val" for job in active):
                continue
            if len(active) >= room() or _stopped() or spend() >= PAUSE_AT:
                break
            if spend() + 1.2 > CAP:
                flags = _flags()
                skipped = list(flags.get("skipped_val") or [])
                if step not in skipped:
                    skipped.append(step)
                flags["skipped_val"] = skipped
                _write_flags(flags)
                print("skip val", step, flush=True)
                continue
            tag = f"p14-{step:06d}-val"
            ckpt = str(OUT / f"step-{step:06d}-base")
            active.append({
                "step": step, "tag": tag, "kind": "val", "phase": "gen",
                "calls": [gen_job.spawn(ckpt, tag, shard, "val") for shard in (0, 1)],
            })
            val_started.add(step)
            flags = _flags()
            flags["val_started"] = sorted(val_started)
            _write_flags(flags)
            print("val", tag, flush=True)

        if spend() >= PAUSE_AT and not _stopped():
            STOP["stop"] = True
            STOP["why"] = f"spend ${spend():.2f} passed ${PAUSE_AT:.0f}"
            why = STOP["why"]
            print(why, flush=True)

        if train_done and not active:
            blocked = _stopped() or spend() >= PAUSE_AT
            flags = _flags()
            need_val = False
            if not blocked and spend() + 1.2 <= CAP:
                for step, doc in summaries.items():
                    if step in (flags.get("skipped_val") or []) or doc.get("val"):
                        continue
                    if step not in (flags.get("val_started") or []):
                        need_val = True
            if blocked or (not queue and not need_val):
                return why
        time.sleep(20)


@app.local_entrypoint()
def launch4():
    call = train4.spawn()
    print("TRAIN4", call.object_id, flush=True)


@app.local_entrypoint()
def launch2():
    call = train2.spawn()
    print("TRAIN2", call.object_id, flush=True)


@app.local_entrypoint()
def launch_a100():
    call = train_a100.spawn()
    print("TRAIN_A100", call.object_id, flush=True)


@app.local_entrypoint()
def launch_l40s(batch: int = 4):
    call = train_l40s_b2.spawn() if int(batch) == 2 else train_l40s.spawn()
    print("TRAIN_L40S", call.object_id, flush=True)


@app.local_entrypoint()
def trial(batch: int = 4, accum: int = 1):
    print(trial_l40s.remote(int(batch), int(accum)), flush=True)


@app.local_entrypoint()
def adopt(call_id: str, gpus: int = 8, accum: int = 1, gpu_name: str = "H100", batch: int = 4, rate: float = H100_RATE):
    call = run.spawn(call_id, gpus, accum, gpu_name, batch, rate)
    print("ADOPT", call.object_id, flush=True)
