"""Plan 13 Track 5 and Track 7.

Eight H100s train from Base. Two L40S score checkpoints. No region pin.
The volume does not require one. Pause past $396. No ElevenLabs and no Stage 2.
"""
import html
import json
import math
import os
import random
import re
import signal
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import modal

from modal_app import BASE, RECIPE, TOKENIZER, image as tts_image
from plan13_live import add_event, snapshot, write_json
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

app = modal.App("qwen3-arabic-plan13-train")
vol = modal.Volume.from_name("qwen3-arabic-tts-v2")
secret = modal.Secret.from_name("huggingface")
STOP = modal.Dict.from_name("plan13-train-stop", create_if_missing=True)

train_image = tts_image.add_local_python_source("modal_app", "plan13_live", "plan13_devloss")
gen_image = tts_image.add_local_python_source("modal_app", "plan13_live", "plan13_eval")
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
        "tensorboard",
    )
    .env({"HF_HOME": "/data/hf_cache"})
    .add_local_python_source(
        "modal_app", "plan13_live", "plan13_eval", "plan12_text", "plan12_letter", "retx_jais",
    )
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_python_source("modal_app", "plan13_live")
)

OUT = Path(f"{_DATA}/runs/stageB1")
ROOT = Path(f"{_DATA}/plan13")
TRAIN = f"{_DATA}/plan13/tier1_v1.jsonl"
DEV = f"{_DATA}/plan13/tier1_dev.jsonl"
A2 = f"{_DATA}/tts_stageA/stageA3/a2_val_codes.jsonl"
UAE = f"{_DATA}/tts_stageA/stageA3/uae100k_val_codes.jsonl"
A3_STEPS = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 20408]
DIALECTS = "ar_msa,ar_egy,ar_sau,ar_uae,ar_gulf,ar_omn,ar_lev,ar_irq,ar_mar,ar_tun,ar_dza,unknown"
APPROVED = 330.0
PAUSE_AT = 396.0
H100_RATE = 3.95
FLAGS = ROOT / "live" / "train_flags.json"
PLAN = ROOT / "train_plan.json"
TB_URL = "https://tausifiqbal10--qwen3-arabic-plan13-status-tensorboard.us-east.modal.run"
PAGE_URL = "https://tausifiqbal10--qwen3-arabic-plan13-status-serve.us-east.modal.run"
PRIORITY = ("ar_egy", "ar_msa", "ar_sau", "ar_uae")
DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
CANON = {
    "ar_egy": "ar_egy", "EGY": "ar_egy", "ar_msa": "ar_msa", "MSA": "ar_msa",
    "ar_sau": "ar_sau", "SAU": "ar_sau", "ar_uae": "ar_uae", "UAE": "ar_uae",
    "ALG": "ar_dza", "ar_dza": "ar_dza", "IRQ": "ar_irq", "ar_irq": "ar_irq",
    "MAR": "ar_mar", "ar_mar": "ar_mar",
}


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


def _flags():
    return _read(FLAGS, {})


def _write_flags(payload):
    write_json(FLAGS, payload)


def _stopped():
    try:
        return bool(STOP["stop"])
    except KeyError:
        return False


def steps_for(n_rows, batch=4, world=8, accum=1):
    n_batches = math.ceil(n_rows / batch)
    per_rank = (n_batches // world) // accum * accum
    return per_rank // accum


def a3_path(step):
    return f"{_DATA}/runs/stageA3/step-{int(step):06d}-base"


def nearest_a3(step):
    ordered = sorted(A3_STEPS, key=lambda item: (abs(item - step), item))
    for item in ordered:
        if os.path.isdir(a3_path(item)):
            return item
    return ordered[0]


def _plain(text):
    return " ".join(DIAC.sub("", text or "").split())


def build_subset():
    """Fixed seed 13. Emirati is split evenly between Habibi and Casablanca."""
    pools = defaultdict(list)
    with open(f"{_DATA}/plan13/testset_v1.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dialect = CANON.get(row.get("dialect"))
            text = _plain(row.get("text"))
            if not dialect or not text:
                continue
            row = dict(row)
            row["dialect"] = dialect
            row["text"] = text
            pools[(dialect, row.get("source") or "")].append(row)
    rng = random.Random(13)

    def take(key, n):
        pool = sorted(pools.get(key) or [], key=lambda item: item.get("clip_id") or "")
        rng.shuffle(pool)
        return pool[:n], len(pool)

    notes = ["Seed 13.", "Gulf is not in the human-checked set."]
    picked = []
    for dialect, n in (("ar_egy", 250), ("ar_msa", 250), ("ar_sau", 250)):
        got, have = take((dialect, "habibi"), n)
        notes.append(f"{dialect} habibi {len(got)} of {n}, pool {have}")
        picked.extend(got)
    for source, n in (("habibi", 125), ("casablanca", 125)):
        got, have = take(("ar_uae", source), n)
        notes.append(f"ar_uae {source} {len(got)} of {n}, pool {have}")
        picked.extend(got)
    for dialect, n in (("ar_irq", 100), ("ar_dza", 100), ("ar_mar", 100)):
        got, have = take((dialect, "habibi"), n)
        notes.append(f"{dialect} habibi {len(got)} of {n}, pool {have}")
        picked.extend(got)
    picked.sort(key=lambda item: (item["dialect"], item.get("clip_id") or ""))
    path = ROOT / "testset_score.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(picked):
            handle.write(json.dumps({
                "i": i, "clip_id": row.get("clip_id"), "source": row.get("source"),
                "dialect": row["dialect"], "text": row["text"],
                "audio": row.get("audio"), "ref_audio": row.get("ref_audio") or row.get("audio"),
                "speaker": row.get("speaker") or "", "never_train": True,
            }, ensure_ascii=False) + "\n")
    (ROOT / "testset_score.md").write_text(
        "# Track 7 subset\n\n" + "\n".join(f"- {line}" for line in notes) + f"\n\nClips {len(picked)}.\n",
        encoding="utf-8",
    )
    print("\n".join(notes), flush=True)
    return notes, len(picked)


def build_dev():
    marker = ROOT / "tier1_dev.done"
    if marker.is_file() and os.path.isfile(DEV):
        saved = _read(marker, {})
        if saved.get("train") and saved.get("dev"):
            print(f"dev split already written, train {saved['train']} dev {saved['dev']}", flush=True)
            return int(saved["train"]), int(saved["dev"])
    tmp = Path(DEV + ".tmp")
    n_train = n_dev = bad = 0
    with open(TRAIN, encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as out:
        for i, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not row.get("audio_codes") or not str(row.get("text") or "").strip():
                continue
            if row.get("split") == "dev":
                out.write(line if line.endswith("\n") else line + "\n")
                n_dev += 1
            elif row.get("split") == "train":
                n_train += 1
            if i % 100000 == 0:
                print(f"scanned {i:,} train {n_train:,} dev {n_dev:,}", flush=True)
                snapshot("track5", {
                    "state": "running", "approved": APPROVED, "spent": 0,
                    "note": f"Preparing the dev split. Scanned {i:,} rows.",
                })
                _commit()
    os.replace(tmp, DEV)
    write_json(marker, {"train": n_train, "dev": n_dev, "bad": bad})
    print(f"dev split train {n_train} dev {n_dev} bad {bad}", flush=True)
    return n_train, n_dev


def _watch(log_path, proc):
    """Sub-loss above 6, NaN, the dev stop after step 2,000, the step-4,000 pause, and the stop flag."""
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
            m = step_re.search(line)
            if m:
                step = int(m.group(1))
            if "val extra" in line:
                continue
            m = val_re.search(line)
            if m:
                value = float(m.group(1))
                lowest = value if lowest is None else min(lowest, value)
        if lowest is not None:
            print(f"WATCH floor {lowest:.4f} at step {step}", flush=True)

    def parse(chunk):
        nonlocal step, lowest, streak
        for line in chunk.splitlines():
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
            why = ""
            try:
                why = str(STOP["why"])
            except KeyError:
                why = "stop flag"
            reason["why"] = "stopped"
            reason["detail"] = why or "stop flag"
            print("WATCH", reason["detail"], flush=True)
            try:
                vol.commit()
            except Exception as exc:
                print("commit before stop failed", exc, flush=True)
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
            try:
                vol.commit()
            except Exception as exc:
                print("commit before stop failed", exc, flush=True)
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return reason
    return reason


def _gpu_line():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        )
        return " ".join(out.split())
    except Exception:
        return ""


def _tail(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 120000))
        return handle.read().decode("utf-8", "replace")


def _beat(log_path, started, steps_per_pass, stop_event):
    posted = False
    while not stop_event.wait(60):
        try:
            flags = _flags()
            text = _tail(log_path)
            step = loss = sub = lr = grad = eta = None
            for line in text.splitlines():
                matched = re.search(
                    r"step (\d+)/(\d+) loss (\S+) main \S+ sub (\S+) lr (\S+)", line,
                )
                if matched:
                    step = int(matched.group(1))
                    loss = matched.group(3)
                    sub = matched.group(4)
                    lr = matched.group(5)
                matched = re.search(r"grad ([0-9.]+)", line)
                if matched:
                    grad = matched.group(1)
                matched = re.search(r"ETA ([0-9.]+) h", line)
                if matched:
                    eta = float(matched.group(1))
            elapsed = time.time() - started
            spend_path = OUT / "spend_train.json"
            base = float(_read(spend_path).get("base_s") or 0)
            train_s = base + elapsed
            train_spent = train_s / 3600 * 8 * H100_RATE
            eval_spent = float(flags.get("eval_spent") or 0)
            spent = train_spent + eval_spent
            if spent >= PAUSE_AT and not _stopped():
                STOP["stop"] = True
                STOP["why"] = f"spend ${spent:.2f} passed ${PAUSE_AT:.0f}"
            projected = spent + (eta or 0) * 8 * H100_RATE
            names = sorted(p.name for p in OUT.glob("step-*-base") if (p / "model.safetensors").is_file())
            rate = (step or 0) / max(elapsed, 1)
            next_save = None
            if step is not None and steps_per_pass:
                marks = [4000]
                quarter = max(1, steps_per_pass // 4)
                for k in range(1, 13):
                    marks.append(k * quarter)
                for ep in range(1, 4):
                    marks.append(ep * steps_per_pass)
                future = [mark for mark in marks if mark > step]
                next_save = min(future) if future else None
            eta_ckpt = None
            if next_save and rate > 0:
                eta_ckpt = (next_save - step) / rate
            snapshot("track5", {
                "state": "running", "approved": APPROVED, "spent": round(spent, 2),
                "projected": round(projected, 2), "step": step, "loss": loss, "sub_loss": sub,
                "lr": lr, "grad": grad,
                "pass": None if step is None else round(step / steps_per_pass, 3),
                "pass_pct": None if step is None else round((step % steps_per_pass) / steps_per_pass * 100, 1),
                "samples_per_s": None if not step else round(rate * 32, 2),
                "eta_ckpt": eta_ckpt, "eta_end": None if eta is None else eta * 3600,
                "checkpoints": ", ".join(names), "gpu": _gpu_line(),
                "early_stop": bool(flags.get("early_stop")),
                "early_stop_why": flags.get("early_stop_why") or "",
                "guard_fail": bool(flags.get("guard_fail")),
                "val_stop": bool(flags.get("val_stop")),
                "nan": bool(flags.get("nan")),
                "note": "No region pin. lr 5e-6, cosine, 3% warmup, from Base.",
            })
            write_json(spend_path, {"base_s": base, "total_s": train_s})
            if step and not posted:
                add_event(f"Training is logging at step {step}. TensorBoard: {TB_URL}")
                posted = True
            vol.commit()
        except Exception as exc:
            print("beat", exc, flush=True)


def _run_train(save_every, save_at, steps_per_pass):
    import torch

    _reload()
    if not os.path.isdir(BASE):
        raise SystemExit(f"missing base model {BASE}")
    for path in (TRAIN, DEV, A2, UAE):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
    if torch.cuda.device_count() != 8:
        raise SystemExit(f"expected 8 GPUs, found {torch.cuda.device_count()}")
    if not re.fullmatch(r"[0-9,]+", save_at or ""):
        raise SystemExit(f"bad save list {save_at}")
    OUT.mkdir(parents=True, exist_ok=True)
    spend_path = OUT / "spend_train.json"
    prev = _read(spend_path)
    write_json(spend_path, {"base_s": float(prev.get("total_s") or prev.get("base_s") or 0), "total_s": float(prev.get("total_s") or 0)})
    log_path = str(OUT / "train.log")
    extra = f"--extra_val_jsonl {A2} " if os.path.exists(A2) else ""
    extra2 = f"--extra_val_jsonl2 {UAE} " if os.path.exists(UAE) else ""
    inner = (
        f"cd {RECIPE} && "
        f"accelerate launch --multi_gpu --num_processes 8 --num_machines 1 "
        f"--mixed_precision bf16 --dynamo_backend no train_dialects.py "
        f"--init_model_path {BASE} --dialects {DIALECTS} "
        f"--train_jsonl {TRAIN} --val_jsonl {DEV} {extra}{extra2}"
        f"--require_split train --output_dir {OUT} --lr 5e-6 --epochs 3 "
        f"--batch_size 4 --grad_accum 1 --eval_every 500 --save_every {int(save_every)} "
        f"--save_at {save_at} --resume 1 --export_speakers 0 "
        f"--id_dropout 0.1 --warmup_frac 0.03 --sub_talker_weight 0.3 "
        f"--diac_prob 0.5 --lang_lr_mult 20 --fp32_master 1 "
        f"2>&1 | tee -a {log_path}"
    )
    print("+", inner, flush=True)
    proc = subprocess.Popen(["bash", "-o", "pipefail", "-c", inner], cwd=RECIPE, start_new_session=True)
    stop_beat = threading.Event()
    threading.Thread(
        target=_beat, args=(log_path, time.time(), steps_per_pass, stop_beat), daemon=True,
    ).start()
    reason = _watch(log_path, proc)
    rc = proc.wait()
    stop_beat.set()
    _commit()
    detail = reason.get("detail") or ""
    if reason.get("why") == "paused":
        return "paused"
    if reason.get("why") == "stopped":
        return "stopped:" + detail
    if rc not in (0, -15, 143):
        raise RuntimeError(f"train failed ({rc})")
    return "done"


@app.function(
    image=train_image, gpu="H100:8", cpu=64, memory=393216, timeout=18 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def train_remote(save_every: int, save_at: str, steps_per_pass: int):
    return _run_train(save_every, save_at, steps_per_pass)


@app.function(
    image=train_image, gpu="L40S", cpu=8, memory=65536, timeout=3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def guard4000():
    """Ten short sentences. Speaking rate 5–12 and the duration cap. No ASR."""
    import zlib

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    _reload()
    ckpt = str(OUT / "step-004000-base")
    dest = OUT / "guard-step004000"
    dest.mkdir(parents=True, exist_ok=True)
    want = ["ar_uae", "ar_gulf", "ar_sau", "ar_egy", "ar_msa"]
    picked = {d: [] for d in want}
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
    _commit()
    bad = [it["file"] for it in manifest if it["empty"] or it["near_cap"] or it["rate_out"]]
    return "fail" if bad else "pass"


@app.function(
    image=gen_image, gpu="L40S", cpu=8, memory=65536, timeout=4 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def generate(ckpt: str, tag: str, shard: int):
    from plan13_eval import generate_body

    _reload()
    meta = generate_body(ckpt, tag, shard)
    _commit()
    return meta


@app.function(
    image=score_image, gpu="L40S", cpu=8, memory=65536, timeout=4 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol}, secrets=[secret],
)
def score(tag: str, shard: int, do_refs: int):
    from plan13_eval import score_body

    _reload()
    meta = score_body(tag, shard, do_refs=do_refs)
    _commit()
    return meta


@app.function(
    image=score_image, cpu=2, memory=16384, timeout=3600,
    retries=0, scaledown_window=2, volumes={"/data": vol}, secrets=[secret],
)
def publish(tag: str, step: int, ckpt: str):
    from plan13_eval import publish_body

    _reload()
    summary = publish_body(tag, step, ckpt)
    _commit()
    return {"tag": tag, "priority_wer": summary.get("priority_wer"), "dollars": summary.get("dollars")}


@app.function(
    image=score_image, cpu=1, memory=4096, timeout=3600, retries=0, scaledown_window=2,
)
def warm_score():
    import transformers
    return transformers.__version__


def _poll(call):
    try:
        return "done", call.get(timeout=0)
    except TimeoutError:
        return "wait", None
    except Exception as exc:
        if exc.__class__.__name__ == "TimeoutError":
            return "wait", None
        return "error", exc


def _fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(summaries):
    lines = [
        "Cohere word error, equal weight on Egyptian, MSA, Saudi and Emirati.",
        "Gulf is not in the human-checked set. Jais is on Emirati only.",
        "Real-recording WER is Cohere on the reference audio, once.",
        "",
    ]
    for tag in sorted(summaries, key=lambda name: (not name.startswith("a3"), name)):
        doc = summaries[tag]
        cells = doc.get("by_dialect") or {}
        bits = [f"{tag} step {doc.get('step')} mean {_fmt(doc.get('priority_wer'))} ${doc.get('dollars')}"]
        for dialect in list(PRIORITY) + ["ar_irq", "ar_dza", "ar_mar"]:
            cell = cells.get(dialect) or {}
            if not cell:
                continue
            extra = ""
            if cell.get("jais_wer") is not None:
                extra += f" jais {_fmt(cell.get('jais_wer'))}"
            if cell.get("ref_wer") is not None:
                extra += f" real {_fmt(cell.get('ref_wer'))}"
            bits.append(
                f"  {dialect} n {cell.get('n')} wer {_fmt(cell.get('wer'))} ler {_fmt(cell.get('ler'))} "
                f"sim {_fmt(cell.get('sim'))} cps {_fmt(cell.get('cps'))} fail {_fmt(cell.get('fail'))}{extra}"
            )
        lines.append("\n".join(bits))
    return "\n\n".join(lines)


def _val_at(path, step):
    if not os.path.exists(path):
        return None
    last = 0
    found = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            matched = re.search(r"\bstep (\d+)/", line)
            if matched:
                last = int(matched.group(1))
            if "val extra" in line:
                continue
            matched = re.search(r"val main loss ([0-9.]+)", line)
            if matched and last <= step:
                found = (last, float(matched.group(1)))
    return found


def _early(plan, summaries):
    pairs = []
    for pair in plan.get("early_pairs") or []:
        new = summaries.get(f"b1-{int(pair['step']):06d}")
        base = summaries.get(f"a3-{int(pair['a3_step']):06d}")
        if not new or not base:
            return None
        new_wer = new.get("priority_wer")
        base_wer = base.get("priority_wer")
        if new_wer is None or base_wer is None or base_wer <= 0:
            return {"stop": False, "pairs": [], "detail": f"missing a score near step {pair['step']}"}
        pairs.append({
            "step": pair["step"], "a3_step": pair["a3_step"], "gap": pair["gap"],
            "new": new_wer, "base": base_wer, "rel": (new_wer - base_wer) / base_wer,
        })
    if len(pairs) < 2:
        return None
    return {"stop": all(item["rel"] > 0.10 for item in pairs), "pairs": pairs}


def _listen(summaries):
    ranked = [
        doc for doc in summaries.values()
        if str(doc.get("tag", "")).startswith("b1-") and doc.get("priority_wer") is not None
    ]
    if len(ranked) < 2 or "a3-020408" not in summaries:
        return ""
    if (ROOT / "listen" / "page.html").is_file():
        return PAGE_URL + "/listen"
    ranked.sort(key=lambda doc: doc["priority_wer"])
    best = ranked[:2]
    subset = []
    with open(ROOT / "testset_score.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                subset.append(json.loads(line))
    chosen = []
    for dialect in PRIORITY:
        chosen.extend([row for row in subset if row.get("dialect") == dialect][:3])
    systems = [
        ("recording", None),
        ("step 20408", "a3-020408"),
        (f"step {best[0]['step']}", best[0]["tag"]),
        (f"step {best[1]['step']}", best[1]["tag"]),
    ]
    items = []
    for row in chosen:
        for label, tag in systems:
            if tag is None:
                audio = row.get("ref_audio") or row.get("audio")
            else:
                audio = str(ROOT / "eval" / tag / f"{int(row['i']):05d}.wav")
            if not audio or not os.path.isfile(audio):
                continue
            items.append({
                "audio": audio, "text": row.get("text") or "", "dialect": row.get("dialect"),
                "label": label,
            })
    rng = random.Random(13)
    rng.shuffle(items)
    for i, item in enumerate(items):
        item["i"] = i
    listen = ROOT / "listen"
    listen.mkdir(parents=True, exist_ok=True)
    write_json(listen / "items.json", items)
    blocks = []
    for item in items:
        blocks.append(
            "<section class='card'>"
            f"<p><b>{html.escape(item['dialect'])}</b></p>"
            f"<p>{html.escape(item['text'])}</p>"
            f"<audio controls src='/listen-audio/{item['i']}'></audio>"
            f"<form method='post' action='/listen/answer'>"
            f"<input type='hidden' name='i' value='{item['i']}'>"
            "<p>Naturalness "
            + " ".join(f"<label><input type='radio' name='naturalness' value='{n}'> {n}</label>" for n in range(1, 6))
            + "</p><p>Does this sound like that dialect? "
            "<label><input type='radio' name='dialect' value='yes'> Yes</label> "
            "<label><input type='radio' name='dialect' value='no'> No</label></p>"
            "<p>Words "
            "<label><input type='radio' name='words' value='all'> All right</label> "
            "<label><input type='radio' name='words' value='few'> 1–2 wrong</label> "
            "<label><input type='radio' name='words' value='many'> Many wrong</label> "
            "<label><input type='radio' name='words' value='garbled'> Garbled</label></p>"
            "<button type='submit'>Save</button></form></section>"
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Listening</title>"
        "<style>body{font-family:Georgia,serif;margin:2rem;background:#f6f4ef;color:#1c1a16}"
        ".card{background:white;padding:1rem;margin:0.8rem 0}</style></head><body>"
        "<h1>Listening</h1><p>The systems are hidden. This page is for Tausif.</p>"
        + "".join(blocks) + "</body></html>"
    )
    (listen / "page.html").write_text(page, encoding="utf-8")
    return PAGE_URL + "/listen"


def write_report(why):
    _reload()
    plan = _read(PLAN, {})
    summaries = {}
    eval_root = ROOT / "eval"
    if eval_root.is_dir():
        for path in eval_root.glob("*/summary.json"):
            doc = _read(path, {})
            if doc.get("tag"):
                summaries[doc["tag"]] = doc
    flags = _flags()
    spend = _read(OUT / "spend_train.json", {})
    train_s = float(spend.get("total_s") or 0)
    train_spent = train_s / 3600 * 8 * H100_RATE
    eval_spent = float(flags.get("eval_spent") or 0)
    lines = [
        "# Plan 13",
        "",
        why,
        "",
        "Trained from Qwen/Qwen3-TTS-12Hz-1.7B-Base. Learning rate 5e-6, cosine, 3% warmup.",
        "No region pin. The volume does not require one. Eight H100 at $3.95 per GPU-hour.",
        "unknown keeps its own language id. Gulf is not in the human-checked set.",
        "ElevenLabs was not called. Stage 2 does not start.",
        "",
        f"Train rows {plan.get('train_rows')}. Dev rows {plan.get('dev_rows')}.",
        f"Steps per pass {plan.get('steps_per_pass')}. Save every {plan.get('save_every')}.",
        f"Extra saves {plan.get('save_at')}.",
        "",
        "## Early stop",
        "",
    ]
    decision = _early(plan, summaries) if plan else None
    if decision and decision.get("pairs"):
        for pair in decision["pairs"]:
            ours = _val_at(str(OUT / "train.log"), pair["step"])
            a3 = _val_at(f"{_DATA}/runs/stageA3/train.log", pair["a3_step"])
            lines.append(
                f"Step {pair['step']} word error {pair['new']:.3f} against A3 step {pair['a3_step']} "
                f"({pair['base']:.3f}, gap {pair['gap']} steps, relative {pair['rel']:+.1%}). "
                f"This run's dev loss {ours}. A3's own val loss {a3}."
            )
        lines.append("Stop fired." if decision.get("stop") else "Stop did not fire.")
        lines.append(
            "Dev loss is this run's new dev split. The A3 number is that run's validation loss on a different set. "
            "The stop uses the Cohere word error rate only."
        )
    else:
        lines.append("The two early checkpoints were not both scored.")
    lines += ["", "## Scores", "", _table(summaries), "", "## Spend", ""]
    lines.append(
        f"H100 wall {train_s / 3600:.2f} h, GPU-hours {train_s / 3600 * 8:.2f}, ${train_spent:.2f}."
    )
    lines.append(f"Track 7 ${eval_spent:.2f}. Combined ${train_spent + eval_spent:.2f}. Approved ${APPROVED:.0f}. Pause ${PAUSE_AT:.0f}.")
    lines.append(f"TensorBoard: {TB_URL}")
    listen = ROOT / "listen" / "page.html"
    if listen.is_file():
        lines.append(f"Listening page, for Tausif only: {PAGE_URL}/listen")
        lines.append(f"Results: {PAGE_URL}/listen/results")
    track4 = ROOT / "track4_report.md"
    if track4.is_file():
        lines += ["", "## Track 4", ""]
        lines.extend(track4.read_text(encoding="utf-8").splitlines()[:40])
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote report", why, flush=True)


def _start_gen(job):
    job["phase"] = "gen"
    job["calls"] = [generate.spawn(job["ckpt"], job["tag"], shard) for shard in (0, 1)]
    job["errors"] = []
    print("gen", job["tag"], flush=True)


def _advance(job):
    state, value = "wait", None
    still = []
    for call in job["calls"]:
        state, value = _poll(call)
        if state == "wait":
            still.append(call)
        elif state == "error":
            job["errors"].append(str(value)[:500])
    job["calls"] = still
    if still:
        return "running"
    if job["errors"]:
        return "error"
    if job["phase"] == "gen":
        job["phase"] = "score"
        job["calls"] = [
            score.spawn(job["tag"], shard, int(job.get("do_refs") or 0)) for shard in (0, 1)
        ]
        print("score", job["tag"], flush=True)
        return "running"
    if job["phase"] == "score":
        job["phase"] = "publish"
        job["calls"] = [publish.spawn(job["tag"], int(job["step"]), job["ckpt"])]
        print("publish", job["tag"], flush=True)
        return "running"
    return "done"


@app.function(
    image=cpu_image, cpu=4, memory=32768, timeout=24 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def run():
    why = "finished"
    try:
        why = _orchestrate() or "finished"
    except Exception as exc:
        why = "error: " + str(exc)[:800]
        print(why, flush=True)
    try:
        write_report(why)
        snapshot("track5", {
            "state": "done", "approved": APPROVED,
            "spent": round(float(_flags().get("combined") or 0), 2),
            "note": why, "early_stop": bool(_flags().get("early_stop")),
            "early_stop_why": _flags().get("early_stop_why") or "",
            "guard_fail": bool(_flags().get("guard_fail")),
        })
        add_event("Plan 13 stopped. " + why)
        _commit()
    except Exception as exc:
        print("report failed", exc, flush=True)
    return why


def _orchestrate():
    _reload()
    STOP["stop"] = False
    STOP["why"] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.json").write_text(json.dumps({
        "job": "track5",
        "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "base_path": BASE,
        "tokenizer": TOKENIZER,
        "cohere": "c3e911b42149bf7a1e53d5cef9878aee87515a23",
        "jais": "1323995fa189c479369a3929be0e95039eba5cba",
        "wavlm": "microsoft/wavlm-base-plus-sv",
        "darbuka": "a7370bb3569d92aa76d511cca2d95b02b888cc7a",
        "train": TRAIN,
        "lr": 5e-6,
        "warmup_frac": 0.03,
        "approved": APPROVED,
        "pause_at": PAUSE_AT,
        "region": None,
    }, indent=2), encoding="utf-8")
    notes, n_subset = build_subset()
    _write_flags({
        "approved": APPROVED, "eval_spent": 0, "eval_per": None,
        "early_stop": False, "early_stop_why": "", "guard_fail": False,
        "hold_eval": False, "val_stop": False, "nan": False, "combined": 0,
    })
    snapshot("track5", {
        "state": "running", "approved": APPROVED, "spent": 0,
        "note": "Preparing. No region pin. The volume does not require one. lr 5e-6, 3% warmup.",
    })
    snapshot("track7", {"state": "running", "spent": 0, "approved": 0, "table": "\n".join(notes)})
    add_event(
        f"Track 5 preparing on 8 H100 from Base. lr 5e-6, 3% warmup. Cap ${APPROVED:.0f}, pause ${PAUSE_AT:.0f}. "
        f"No region pin. Subset {n_subset} clips, seed 13. TensorBoard {TB_URL}"
    )
    _commit()
    warm = warm_score.spawn()
    queue = [{
        "tag": "a3-020408", "ckpt": a3_path(20408), "step": 20408, "do_refs": 1,
    }]
    seen = {"a3-020408"}
    why = "finished"
    current = queue.pop(0)
    _start_gen(current)
    measured = None
    hold = False
    summaries = {}
    early_announced = False
    listen_announced = False
    train_call = None
    guard_done = False
    train_done = False
    plan = {}
    dev_ready = False

    def enqueue(tag, ckpt, step, do_refs=0, front=False):
        if tag in seen or not os.path.isdir(ckpt):
            if tag not in seen and not os.path.isdir(ckpt):
                print("missing checkpoint", ckpt, flush=True)
            return
        seen.add(tag)
        item = {"tag": tag, "ckpt": ckpt, "step": step, "do_refs": do_refs}
        if front:
            queue.insert(0, item)
        else:
            queue.append(item)

    def spend_now():
        train_s = float(_read(OUT / "spend_train.json").get("total_s") or 0)
        return train_s / 3600 * 8 * H100_RATE + float(_flags().get("eval_spent") or 0)

    while True:
        if not dev_ready:
            n_train, n_dev = build_dev()
            steps = steps_for(n_train)
            if steps < 1000:
                raise RuntimeError(f"too few train rows {n_train}, steps {steps}")
            quarter = max(1, steps // 4)
            early_steps = [quarter, quarter * 2]
            pairs = []
            for step in early_steps:
                a3_step = nearest_a3(step)
                pairs.append({
                    "step": step, "a3_step": a3_step, "gap": abs(step - a3_step),
                    "a3_ckpt": a3_path(a3_step),
                })
            save_at = [4000] + [steps * ep for ep in (1, 2, 3)]
            plan = {
                "train_rows": n_train, "dev_rows": n_dev, "steps_per_pass": steps,
                "save_every": quarter, "save_at": save_at, "early_pairs": pairs,
                "lr": 5e-6, "warmup_frac": 0.03, "subset": n_subset,
            }
            write_json(PLAN, plan)
            add_event(
                f"Train rows {n_train:,}. Steps per pass {steps}. Quarter every {quarter}. "
                f"Early checkpoints {early_steps[0]} (A3 {pairs[0]['a3_step']}, gap {pairs[0]['gap']}) "
                f"and {early_steps[1]} (A3 {pairs[1]['a3_step']}, gap {pairs[1]['gap']})."
            )
            _commit()
            train_call = train_remote.spawn(quarter, ",".join(str(x) for x in save_at), steps)
            print("TRAIN", train_call.object_id, flush=True)
            dev_ready = True

        warm_state, warm_value = _poll(warm)
        if warm_state == "done":
            print("score image", warm_value, flush=True)
            warm = None
        elif warm_state == "error":
            hold = True
            add_event("Track 7 score image failed to start. " + str(warm_value)[:300])
            warm = None

        if current is not None:
            status = _advance(current)
            if status == "error":
                hold = True
                _reload()
                extra = 0.0
                for shard in (0, 1):
                    for kind in ("gen", "score"):
                        meta = _read(ROOT / "eval" / current["tag"] / f"{kind}_{shard}.json", {})
                        extra += float(meta.get("dollars") or 0)
                flags = _flags()
                flags["eval_spent"] = round(float(flags.get("eval_spent") or 0) + extra, 4)
                flags["hold_eval"] = True
                flags["combined"] = round(spend_now(), 2)
                _write_flags(flags)
                add_event(
                    f"Track 7 {current['tag']} failed. Not starting another scoring job. "
                    + current["errors"][0][:240]
                )
                current = None
            elif status == "done":
                _reload()
                summary = _read(ROOT / "eval" / current["tag"] / "summary.json", {})
                if summary:
                    summaries[current["tag"]] = summary
                    flags = _flags()
                    flags["eval_spent"] = round(float(flags.get("eval_spent") or 0) + float(summary.get("dollars") or 0), 4)
                    if measured is None:
                        measured = float(summary.get("dollars") or 0)
                        flags["eval_per"] = measured
                        add_event(
                            f"Track 7 first checkpoint {current['tag']}: {summary.get('wall_s')} GPU-seconds "
                            f"summed across the workers, ${measured:.2f}."
                        )
                        if measured > 10:
                            hold = True
                            flags["hold_eval"] = True
                            add_event(
                                f"Track 7 would cost ${measured:.2f} per checkpoint, over $10. "
                                "No further scoring until you say so. Training continues."
                            )
                        else:
                            for pair in plan.get("early_pairs") or []:
                                enqueue(
                                    f"a3-{int(pair['a3_step']):06d}", pair["a3_ckpt"],
                                    int(pair["a3_step"]), front=True,
                                )
                    flags["combined"] = round(spend_now(), 2)
                    _write_flags(flags)
                    decision = _early(plan, summaries)
                    if decision and decision.get("stop") and not flags.get("early_stop"):
                        text = "; ".join(
                            f"step {p['step']} is {p['rel']:+.1%} vs A3 step {p['a3_step']} (gap {p['gap']})"
                            for p in decision["pairs"]
                        )
                        flags["early_stop"] = True
                        flags["early_stop_why"] = text
                        _write_flags(flags)
                        STOP["stop"] = True
                        STOP["why"] = "early stop " + text
                        add_event("Early stop. Both of the first two checkpoints are more than 10% worse. " + text)
                        hold = True
                    elif decision and decision.get("pairs") and not decision.get("stop") and not early_announced:
                        early_announced = True
                        add_event("Early-stop check passed. " + "; ".join(
                            f"step {p['step']} {p['rel']:+.1%} vs A3 {p['a3_step']}" for p in decision["pairs"]
                        ))
                    link = _listen(summaries)
                    if link and not listen_announced:
                        listen_announced = True
                        add_event("Listening page, for Tausif only: " + link)
                    snapshot("track7", {
                        "state": "held" if hold else "running",
                        "spent": flags["eval_spent"], "approved": 0,
                        "table": _table(summaries),
                        "note": f"Measured ${measured:.2f} per checkpoint." if measured is not None else "",
                    })
                    _commit()
                else:
                    hold = True
                    flags = _flags()
                    flags["hold_eval"] = True
                    _write_flags(flags)
                    add_event(f"Track 7 {current['tag']} finished without a summary. Holding further scoring.")
                current = None

        if train_call is not None and not train_done:
            state, value = _poll(train_call)
            if state == "done":
                text = str(value)
                print("train returned", text, flush=True)
                if text == "paused" and not guard_done:
                    add_event("Step 4,000 saved. Running the speaking-rate guard.")
                    _commit()
                    result = guard4000.remote()
                    guard_done = True
                    if result != "pass":
                        flags = _flags()
                        flags["guard_fail"] = True
                        _write_flags(flags)
                        STOP["stop"] = True
                        STOP["why"] = "step 4000 guard failed"
                        add_event("Step 4,000 guard failed. Training stays stopped.")
                        train_done = True
                        why = "guard failed"
                        train_call = None
                    else:
                        add_event("Step 4,000 guard passed. Training resumes.")
                        train_call = train_remote.spawn(
                            plan["save_every"], ",".join(str(x) for x in plan["save_at"]), plan["steps_per_pass"],
                        )
                        print("TRAIN resume", train_call.object_id, flush=True)
                else:
                    train_done = True
                    train_call = None
                    if text.startswith("stopped:"):
                        flags = _flags()
                        low = text.lower()
                        if "nan" in low or "inf" in low:
                            flags["nan"] = True
                        if "val " in low:
                            flags["val_stop"] = True
                        flags["combined"] = round(spend_now(), 2)
                        _write_flags(flags)
                        why = text
                    elif text == "done":
                        why = "training finished all planned passes"
                    else:
                        why = text
            elif state == "error":
                train_done = True
                train_call = None
                why = "train error: " + str(value)[:500]
                add_event(why)

        if not hold and not _stopped() and spend_now() < PAUSE_AT:
            _reload()
            for path in sorted(OUT.glob("step-*-base")):
                if not (path / "model.safetensors").is_file():
                    continue
                step = int(path.name.split("-")[1])
                enqueue(f"b1-{step:06d}", str(path), step)

        if current is None and queue and not hold and not _stopped() and spend_now() < PAUSE_AT:
            current = queue.pop(0)
            _start_gen(current)

        if spend_now() >= PAUSE_AT and not _stopped():
            STOP["stop"] = True
            STOP["why"] = f"spend ${spend_now():.2f} passed ${PAUSE_AT:.0f}"
            hold = True
            add_event(STOP["why"] + ". Workers pause. Partial report follows.")

        if train_done and current is None and (hold or not queue or _stopped()):
            return why
        time.sleep(20)


@app.function(
    image=score_image, gpu="L40S", cpu=4, memory=65536, timeout=3600,
    retries=0, scaledown_window=2, volumes={"/data": vol}, secrets=[secret],
)
def debug10():
    from plan13_eval import debug_body

    _reload()
    result = debug_body()
    _commit()
    return result


@app.local_entrypoint()
def go():
    call = run.spawn()
    print(call.object_id, flush=True)


@app.function(
    image=cpu_image, cpu=2, memory=8192, timeout=3600, retries=0, scaledown_window=2,
    volumes={"/data": vol},
)
def hash_pair():
    import hashlib

    _reload()
    paths = {
        "45184": f"{_DATA}/runs/stageB1/step-045184-base/model.safetensors",
        "45190": f"{_DATA}/runs/stageB1/step-045190-base/model.safetensors",
    }
    out = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        out[name] = {"bytes": size, "sha256": digest.hexdigest()}
        print(name, size, out[name]["sha256"], flush=True)
    out["same_weights"] = out["45184"]["sha256"] == out["45190"]["sha256"]
    return out


@app.function(
    image=gen_image, gpu="L40S", cpu=4, memory=65536, timeout=3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def noise_probe():
    from plan13_eval import noise_time

    _reload()
    return noise_time(f"{_DATA}/runs/stageA3/step-020408-base", n=32)


@app.local_entrypoint()
def probe_go():
    hashed = hash_pair.spawn()
    print("probe " + json.dumps(noise_probe.remote()), flush=True)
    print("hash " + json.dumps(hashed.get()), flush=True)


@app.local_entrypoint()
def debug_go():
    print(json.dumps(debug10.remote(), ensure_ascii=False), flush=True)


@app.function(
    image=cpu_image, cpu=1, memory=4096, timeout=3600, retries=0, scaledown_window=2,
    volumes={"/data": vol},
)
def note_scores():
    """Refresh the Track 7 table from the summaries on the volume."""
    _reload()
    summaries = {}
    root = Path(f"{_DATA}/plan13/eval")
    if root.is_dir():
        for path in sorted(root.glob("*/summary.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("tag"):
                summaries[doc["tag"]] = doc
    spent = 0.0
    for doc in summaries.values():
        spent += float(doc.get("dollars") or 0)
    flags = _read(FLAGS)
    prior = float(flags.get("eval_spent") or 0)
    # The first broken pass is already in eval_spent. Add only the new summary total if larger.
    flags["eval_spent"] = round(max(prior, spent), 4)
    write_json(FLAGS, flags)
    snapshot("track7", {
        "state": "running",
        "approved": 0,
        "spent": round(float(flags["eval_spent"]), 2),
        "projected": 0,
        "table": _table(summaries),
        "note": "Rescored with 16 kHz audio. Similarity is generated against the reference.",
    })
    add_event(
        "Track 7 rescored step 20,408 after the 16 kHz fix. "
        + " ".join(
            f"{doc.get('tag')} mean {_fmt(doc.get('priority_wer'))} ${doc.get('dollars')}"
            for doc in summaries.values()
        )
    )
    _commit()
    return {"tags": sorted(summaries), "spent": flags["eval_spent"]}


@app.local_entrypoint()
def rescore20408():
    calls = [score.spawn("a3-020408", shard, 1) for shard in (0, 1)]
    for call in calls:
        print(json.dumps(call.get(), ensure_ascii=False), flush=True)
    print(json.dumps(
        publish.remote("a3-020408", 20408, f"{_DATA}/runs/stageA3/step-020408-base"),
        ensure_ascii=False,
    ), flush=True)
    print(json.dumps(note_scores.remote(), ensure_ascii=False), flush=True)


@app.function(
    image=train_image, gpu="L40S", cpu=8, memory=65536, timeout=6 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def devloss():
    from plan13_devloss import run

    _reload()
    result = run(_commit)
    _commit()
    return result


@app.local_entrypoint()
def devloss_go():
    print(json.dumps(devloss.remote(), ensure_ascii=False), flush=True)


REST = (
    ("b1-022595", 22595, f"{_DATA}/runs/stageB1/step-022595-base"),
    ("b1-045184", 45184, f"{_DATA}/runs/stageB1/step-045184-base"),
    ("b1-045190", 45190, f"{_DATA}/runs/stageB1/step-045190-base"),
)


@app.function(
    image=cpu_image, cpu=1, memory=4096, timeout=3600, retries=0, scaledown_window=2,
    volumes={"/data": vol},
)
def build_listen():
    _reload()
    summaries = {}
    root = Path(f"{_DATA}/plan13/eval")
    for path in sorted(root.glob("*/summary.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("tag"):
            summaries[doc["tag"]] = doc
    url = _listen(summaries)
    snapshot("track7", {
        "state": "done",
        "approved": 0,
        "spent": round(sum(float(doc.get("dollars") or 0) for doc in summaries.values()), 2),
        "projected": 0,
        "table": _table(summaries),
        "note": "Scored step 20,408, end of pass 1, the checkpoint nearest the best dev loss, and step 45,190.",
    })
    if url:
        add_event("Listening page, for Tausif only: " + url)
    _commit()
    return url or "listening page was not built"


@app.local_entrypoint()
def score_rest():
    pending = []
    for tag, step, ckpt in REST:
        calls = [generate.spawn(ckpt, tag, shard) for shard in (0, 1)]
        pending.append((tag, step, ckpt, calls))
    over = False
    for tag, step, ckpt, calls in pending:
        if over:
            for call in calls:
                call.cancel()
            print("cancelled", tag, flush=True)
            continue
        metas = [call.get() for call in calls]
        gen_dollars = sum(float(meta.get("dollars") or 0) for meta in metas)
        print(f"gen {tag} ${gen_dollars:.2f}", flush=True)
        if gen_dollars > 10:
            over = True
            print(f"{tag} generation is already over $10. Not scoring it or starting the rest.", flush=True)
            continue
        scored = [call.get() for call in (score.spawn(tag, shard, 0) for shard in (0, 1))]
        total = gen_dollars + sum(float(meta.get("dollars") or 0) for meta in scored)
        print(f"total {tag} ${total:.2f}", flush=True)
        print(json.dumps(publish.remote(tag, step, ckpt), ensure_ascii=False), flush=True)
        if total > 10:
            over = True
            print(f"{tag} is over $10. Later checkpoints will not be scored.", flush=True)
    print(build_listen.remote(), flush=True)


@app.local_entrypoint()
def continue_go():
    loss_call = devloss.spawn()
    score_calls = [score.spawn("a3-020408", shard, 1) for shard in (0, 1)]
    for call in score_calls:
        print("score " + json.dumps(call.get(), ensure_ascii=False), flush=True)
    print("publish " + json.dumps(
        publish.remote("a3-020408", 20408, f"{_DATA}/runs/stageA3/step-020408-base"),
        ensure_ascii=False,
    ), flush=True)
    print("note " + json.dumps(note_scores.remote(), ensure_ascii=False), flush=True)
    print("devloss " + json.dumps(loss_call.get(), ensure_ascii=False), flush=True)


NOISE_JOBS = (
    ("a3-020408", f"{_DATA}/runs/stageA3/step-020408-base"),
    ("b1-022595", f"{_DATA}/runs/stageB1/step-022595-base"),
    ("b1-045190", f"{_DATA}/runs/stageB1/step-045190-base"),
)


@app.function(
    image=gen_image, gpu="L40S", cpu=8, memory=65536, timeout=4 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def noise_gen(ckpt: str, tag: str, shard: int):
    from plan13_eval import noise_generate

    started = time.time()
    _reload()
    meta = noise_generate(ckpt, tag, shard)
    _commit()
    wall = time.time() - started
    meta["wall_s"] = round(wall, 1)
    meta["dollars"] = round(wall / 3600 * 1.95, 4)
    return meta


@app.function(
    image=score_image, gpu="L40S", cpu=8, memory=65536, timeout=5400,
    retries=0, scaledown_window=2, volumes={"/data": vol}, secrets=[secret],
)
def noise_score_fn(tag: str, shard: int):
    from plan13_eval import noise_score

    started = time.time()
    _reload()
    meta = noise_score(tag, shard)
    _commit()
    wall = time.time() - started
    meta["wall_s"] = round(wall, 1)
    meta["dollars"] = round(wall / 3600 * 1.95, 4)
    return meta


@app.function(
    image=score_image, gpu="L40S", cpu=4, memory=65536, timeout=3600,
    retries=0, scaledown_window=2, volumes={"/data": vol}, secrets=[secret],
)
def jais_refs():
    from plan13_eval import jais_ref_body

    started = time.time()
    _reload()
    meta = jais_ref_body()
    _commit()
    wall = time.time() - started
    meta["wall_s"] = round(wall, 1)
    meta["dollars"] = round(wall / 3600 * 1.95, 4)
    return meta


@app.local_entrypoint()
def noise_go():
    """Three seeds on three checkpoints. Does not train."""
    calls = []
    for tag, ckpt in NOISE_JOBS:
        for shard in (0, 1):
            call = noise_gen.spawn(ckpt, tag, shard)
            calls.append((tag, shard, call))
            print(f"spawn {tag} {shard} {call.object_id}", flush=True)
    jais_call = jais_refs.spawn()
    print(f"spawn jais {jais_call.object_id}", flush=True)
    spent = 0.0
    metas = []
    for tag, shard, call in calls:
        meta = call.get()
        metas.append(meta)
        spent += float(meta.get("dollars") or 0)
        print(f"gen {tag} shard {shard} ${meta.get('dollars')} done={meta.get('done')} seeds={meta.get('seeds_done')}", flush=True)
    print(f"generation ${spent:.2f}", flush=True)
    if spent > 20:
        print("generation passed $20. Not scoring.", flush=True)
        print("jais " + json.dumps(jais_call.get(), ensure_ascii=False), flush=True)
        return
    score_calls = []
    for meta in metas:
        if not meta.get("seeds_done"):
            print(f"no audio for {meta.get('tag')} shard {meta.get('shard')}", flush=True)
            continue
        call = noise_score_fn.spawn(meta["tag"], int(meta["shard"]))
        score_calls.append(call)
        print(f"score {meta['tag']} {meta['shard']} {call.object_id}", flush=True)
    score_spent = 0.0
    for call in score_calls:
        meta = call.get()
        score_spent += float(meta.get("dollars") or 0)
        print(f"scored {meta.get('tag')} shard {meta.get('shard')} ${meta.get('dollars')} n={meta.get('n')}", flush=True)
    jais_meta = jais_call.get()
    print("jais " + json.dumps(jais_meta, ensure_ascii=False), flush=True)
    print(f"total ${spent + score_spent + float(jais_meta.get('dollars') or 0):.2f}", flush=True)
