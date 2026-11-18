"""Plan 14 encode. Reads the ElevenLabs mix. Writes only /data/exp_11labs/.

Does not train. Design rows stay in: the English prefix is removed and the clip
is its own voice prompt. No region pin. L40S at $1.95 per GPU-hour.
"""
import json
import math
import os
import re
import subprocess
import time
import zlib
from collections import Counter
from pathlib import Path

import modal

from modal_app import RECIPE, TOKENIZER, image as tts_image
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

app = modal.App("qwen3-arabic-plan14-encode")
vol = modal.Volume.from_name("qwen3-arabic-tts-v2")
secret = modal.Secret.from_name("qwen3-b2")
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rclone")
    .add_local_python_source("modal_app")
)
gpu_image = tts_image.add_local_python_source("modal_app", "plan8_prepare_data")

ROOT = Path(f"{_DATA}/exp_11labs")
AUDIO = ROOT / "audio"
MANIFEST = ROOT / "manifest"
CODES = ROOT / "codes"
REMOTE = "b2ds:companyNames-dataset/11labs_gen_audios/arabic_dialects_voxcpm"
MIX = f"{REMOTE}/A_ref35_design20"
RATE = 1.95
WORKERS = 4
PAUSE_AT = 6.0
DESC = re.compile(r"^\([^)]*\)\s*")
DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
ARABIC = re.compile(r"[\u0600-\u06FF]")
DIALECT = {
    "msa": "ar_msa",
    "egyptian": "ar_egy",
    "saudi": "ar_sau",
    "iraqi": "ar_irq",
    "tunisian": "ar_tun",
    "moroccan": "ar_mar",
    "omani": "ar_omn",
    "jordanian": "ar_jor",
    "syrian": "ar_syr",
    "sudanese": "ar_sdn",
}


def _env():
    env = os.environ.copy()
    env["RCLONE_CONFIG_B2DS_TYPE"] = "b2"
    env["RCLONE_CONFIG_B2DS_ACCOUNT"] = os.environ["B2_ACCOUNT_ID"]
    env["RCLONE_CONFIG_B2DS_KEY"] = os.environ["B2_APPLICATION_KEY"]
    return env


def _run(args):
    proc = subprocess.run(args, env=_env(), capture_output=True, text=True)
    if proc.returncode:
        err = proc.stderr.replace(os.environ.get("B2_APPLICATION_KEY", "___"), "***")
        err = err.replace(os.environ.get("B2_ACCOUNT_ID", "___"), "***")
        raise RuntimeError(err[-800:])
    return proc.stdout


def _local_audio(rel):
    rel = str(rel or "").replace("\\", "/").lstrip("./")
    at = rel.find("audio/")
    if at < 0:
        raise ValueError(f"audio path {rel}")
    return AUDIO / rel[at + len("audio/"):]


def _strip_design(text):
    matched = DESC.match(text or "")
    if not matched:
        return None
    return text[matched.end():].strip()


def _rows(split, raw_name, seen):
    body = _run(["rclone", "cat", f"{MIX}/{raw_name}"])
    out = []
    bad_prefix = 0
    no_arabic = 0
    unknown = Counter()
    for line in body.splitlines():
        if not line.strip():
            continue
        src = json.loads(line)
        mode = src.get("mode") or ""
        dialect = DIALECT.get(src.get("dialect") or src.get("speaker") or "")
        if not dialect:
            unknown[src.get("dialect") or src.get("speaker") or ""] += 1
            continue
        text = src.get("text") or ""
        if mode == "design":
            stripped = _strip_design(text)
            if not stripped:
                bad_prefix += 1
                continue
            text = stripped
        if not ARABIC.search(text):
            no_arabic += 1
            continue
        audio = _local_audio(src.get("audio"))
        if mode == "ref" and src.get("ref_audio"):
            ref = _local_audio(src["ref_audio"])
        else:
            ref = audio
        clip_id = audio.stem
        if clip_id in seen:
            clip_id = f"{dialect}_{clip_id}"
        if clip_id in seen:
            raise RuntimeError(f"duplicate clip id {clip_id}")
        seen.add(clip_id)
        out.append({
            "clip_id": clip_id,
            "source": "internal_11labs",
            "tag": "internal_11labs",
            "split": split,
            "original_mode": mode,
            "dialect": dialect,
            "speaker": dialect,
            "text": text,
            "no_diac": DIAC.search(text) is None,
            "duration": float(src.get("duration") or 0),
            "audio": str(audio),
            "ref_audio": str(ref),
            "voice_id": src.get("voice_id") or "",
        })
    return out, bad_prefix, no_arabic, unknown


def _steps(n_rows, batch=4, world=8):
    return math.ceil(n_rows / batch) // world


def _train_estimate(steps):
    # Track 5 spent $73.79 to reach step 46,500 on 8 H100 at $3.95, no region pin.
    per_step = 73.79 / 46500
    dollars = steps * per_step
    wall_s = dollars / (8 * 3.95) * 3600
    return round(dollars, 2), round(wall_s / 60, 1)


def build_manifests():
    vol.reload()
    MANIFEST.mkdir(parents=True, exist_ok=True)
    seen = set()
    train, bad_tr, ar_tr, unk_tr = _rows("train", "train.jsonl", seen)
    val, bad_va, ar_va, unk_va = _rows("val", "val.jsonl", seen)
    if unk_tr or unk_va:
        raise RuntimeError(f"unknown dialect train={dict(unk_tr)} val={dict(unk_va)}")
    if bad_tr or bad_va or ar_tr or ar_va:
        raise RuntimeError(
            f"bad design prefix train={bad_tr} val={bad_va}; "
            f"no Arabic train={ar_tr} val={ar_va}"
        )
    if len(train) != 87717 or len(val) != 1024:
        raise RuntimeError(f"split is {len(train)} / {len(val)}, expected 87717 / 1024")
    modes = {
        "train": dict(Counter(row["original_mode"] for row in train)),
        "val": dict(Counter(row["original_mode"] for row in val)),
    }
    modes["all"] = dict(Counter(
        row["original_mode"] for row in train + val
    ))
    for name, rows in (("train", train), ("val", val)):
        path = MANIFEST / f"{name}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    steps = _steps(len(train))
    half = steps // 2
    marks = []
    for k in range(1, 7):
        mark = half * k
        if mark not in marks and mark > 0:
            marks.append(mark)
    for ep in (1, 2, 3):
        mark = steps * ep
        if mark not in marks:
            marks.append(mark)
    marks.sort()
    dollars, minutes = _train_estimate(steps * 3)
    stats = {
        "train": len(train),
        "val": len(val),
        "total": len(train) + len(val),
        "modes": modes,
        "hours": {
            "train": round(sum(row["duration"] for row in train) / 3600, 2),
            "val": round(sum(row["duration"] for row in val) / 3600, 2),
        },
        "steps_per_pass": steps,
        "steps_3_passes": steps * 3,
        "checkpoints": marks,
        "train_minutes": minutes,
        "train_dollars": dollars,
        "encode_clips": len(train) + len(val),
        "encode_dollars_1gpu": round((len(train) + len(val)) / 884 * RATE / 60, 2),
        "lr": 2e-6,
        "gpus": 8,
        "note": "Training is not launched. Estimate scales Track 5's $73.79 to step 46500.",
    }
    (MANIFEST / "counts.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    vol.commit()
    print(json.dumps(stats), flush=True)
    return stats


def _dollars(seconds):
    return seconds / 3600 * RATE


def _load(path):
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _shard(clip_id):
    return zlib.crc32(clip_id.encode()) % WORKERS


def _spend_path(worker):
    return CODES / f"w{worker}.spend.json"


# The first pass paused before spend files existed. Used once, then the file is written.
_FIRST_PASS = {0: 0.8017, 1: 1.5018, 2: 0.8319, 3: 0.7999}


def _prior_spend(worker):
    path = _spend_path(worker)
    if path.is_file():
        try:
            return float(json.loads(path.read_text(encoding="utf-8")).get("spent") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0
    if (CODES / f"w{worker}.jsonl").is_file() or (CODES / f"w{worker}.jsonl.tmp").is_file():
        return _FIRST_PASS.get(worker, 0.0)
    return 0.0


def _save_spend(worker, spent):
    _spend_path(worker).write_text(
        json.dumps({"spent": round(float(spent), 4)}) + "\n", encoding="utf-8",
    )


@app.function(
    image=cpu_image, secrets=[secret], volumes={"/data": vol},
    cpu=4, memory=8192, timeout=3600, retries=0, scaledown_window=2,
)
def count():
    return build_manifests()


@app.function(
    image=cpu_image, secrets=[secret], volumes={"/data": vol},
    cpu=8, memory=8192, timeout=3 * 3600, retries=0, scaledown_window=2,
)
def fetch_audio():
    vol.reload()
    if not (MANIFEST / "train.jsonl").is_file():
        raise RuntimeError("manifest missing")
    AUDIO.mkdir(parents=True, exist_ok=True)
    started = time.time()
    already = sum(1 for _ in AUDIO.rglob("*.wav"))
    if already >= 88741:
        missing = []
        for name in ("train.jsonl", "val.jsonl"):
            for row in _load(MANIFEST / name):
                if not Path(row["audio"]).is_file() or not Path(row["ref_audio"]).is_file():
                    missing.append(row["clip_id"])
                    if len(missing) >= 5:
                        break
            if len(missing) >= 5:
                break
        if not missing:
            meta = {"wavs": already, "wall_s": 0, "skipped": True}
            print(json.dumps(meta), flush=True)
            return meta
    proc = subprocess.run(
        ["rclone", "copy", f"{REMOTE}/audio", str(AUDIO),
         "--include", "*.wav", "--transfers", "32", "--checkers", "16",
         "--fast-list", "--stats", "30s", "--stats-one-line"],
        env=_env(),
    )
    if proc.returncode:
        err = (proc.stderr or b"").decode("utf-8", "replace")
        err = err.replace(os.environ.get("B2_APPLICATION_KEY", "___"), "***")
        raise RuntimeError(err[-800:])
    missing = []
    for name in ("train.jsonl", "val.jsonl"):
        for row in _load(MANIFEST / name):
            if not Path(row["audio"]).is_file() or not Path(row["ref_audio"]).is_file():
                missing.append(row["clip_id"])
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break
    if missing:
        raise RuntimeError("missing wavs, first " + ", ".join(missing))
    wavs = sum(1 for _ in AUDIO.rglob("*.wav"))
    vol.commit()
    meta = {"wavs": wavs, "wall_s": round(time.time() - started, 1)}
    print(json.dumps(meta), flush=True)
    return meta


@app.function(
    image=gpu_image, gpu="L40S", cpu=8, memory=32768, timeout=4 * 3600,
    retries=0, scaledown_window=2, volumes={"/data": vol},
)
def encode_worker(worker: int):
    import sys

    sys.path.insert(0, RECIPE)
    from plan8_prepare_data import encode_batch
    from qwen_tts import Qwen3TTSTokenizer

    started = time.time()
    cap = PAUSE_AT / WORKERS
    vol.reload()
    prior = _prior_spend(worker)
    rows = []
    for name in ("train.jsonl", "val.jsonl"):
        for row in _load(MANIFEST / name):
            if _shard(row["clip_id"]) == worker:
                rows.append(row)
    CODES.mkdir(parents=True, exist_ok=True)
    final = CODES / f"w{worker}.jsonl"
    tmp = CODES / f"w{worker}.jsonl.tmp"
    done = {row["clip_id"] for row in _load(tmp)}
    done.update(row["clip_id"] for row in _load(final))
    left = [row for row in rows if row["clip_id"] not in done]
    print(f"worker {worker} assigned {len(rows)} done {len(done)} left {len(left)}", flush=True)
    if not left:
        if tmp.is_file() and not final.is_file():
            os.replace(tmp, final)
        _save_spend(worker, prior)
        vol.commit()
        return {"worker": worker, "done": len(done), "fail": 0, "stopped": False, "spent": round(prior, 4)}
    tokenizer = Qwen3TTSTokenizer.from_pretrained(TOKENIZER, device_map="cuda:0")
    handle = tmp.open("a", encoding="utf-8")
    fail = 0
    stopped = False
    written = 0
    try:
        batch_lines, batch_audio = [], []

        def flush():
            nonlocal batch_lines, batch_audio, written
            if not batch_lines:
                return
            coded = []
            encode_batch(tokenizer, batch_lines, batch_audio, coded)
            for row in coded:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            batch_lines, batch_audio = [], []

        for i, row in enumerate(left):
            if _dollars(time.time() - started) > cap:
                stopped = True
                print(f"worker {worker} paused at ${cap:.2f}", flush=True)
                break
            if not Path(row["audio"]).is_file():
                fail += 1
                continue
            batch_lines.append(row)
            batch_audio.append(row["audio"])
            if len(batch_lines) >= 4:
                flush()
            if written and written % 2000 == 0:
                vol.commit()
                print(f"worker {worker} wrote {written} ${_dollars(time.time() - started):.2f}", flush=True)
        if not stopped:
            flush()
    finally:
        handle.close()
    if not stopped and fail == 0:
        os.replace(tmp, final)
    spent = round(prior + _dollars(time.time() - started), 4)
    _save_spend(worker, spent)
    vol.commit()
    have = len(_load(final if final.is_file() else tmp))
    print(json.dumps({"worker": worker, "done": have, "fail": fail, "stopped": stopped, "spent": spent}), flush=True)
    return {"worker": worker, "done": have, "fail": fail, "stopped": stopped, "spent": spent}


@app.function(
    image=cpu_image, volumes={"/data": vol}, cpu=2, memory=32768,
    timeout=5 * 3600, retries=0, scaledown_window=2,
)
def encode_all():
    fetch_audio.remote()
    vol.reload()
    parts = []
    pending = []
    for worker in range(WORKERS):
        if (CODES / f"w{worker}.jsonl").is_file():
            parts.append({
                "worker": worker, "done": None, "fail": 0, "stopped": False,
                "spent": round(_prior_spend(worker), 4), "skipped": True,
            })
        else:
            pending.append(worker)
    print(json.dumps({"resume": pending, "kept": [p["worker"] for p in parts]}), flush=True)
    calls = [encode_worker.spawn(worker) for worker in pending]
    parts.extend(call.get() for call in calls)
    if any(part.get("stopped") or part.get("fail") for part in parts):
        raise RuntimeError("encode paused or failed: " + json.dumps(parts))
    vol.reload()
    by_id = {}
    for worker in range(WORKERS):
        path = CODES / f"w{worker}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing codes for worker {worker}")
        for row in _load(path):
            by_id[row["clip_id"]] = row
    written = {}
    for name in ("train", "val"):
        rows = _load(MANIFEST / f"{name}.jsonl")
        missing = [row["clip_id"] for row in rows if row["clip_id"] not in by_id or not by_id[row["clip_id"]].get("audio_codes")]
        if missing:
            raise RuntimeError(f"{name} missing {len(missing)} codes, first {missing[0]}")
        dest = ROOT / f"{name}_codes.jsonl"
        tmp = dest.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(by_id[row["clip_id"]], ensure_ascii=False) + "\n")
        os.replace(tmp, dest)
        written[name] = len(rows)
    spent = round(sum(part.get("spent") or 0 for part in parts), 4)
    meta = {"parts": parts, "written": written, "gpu_dollars": spent, "tag": "internal_11labs"}
    (ROOT / "encode.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    vol.commit()
    print(json.dumps(meta), flush=True)
    return meta


@app.local_entrypoint()
def main(stage: str = "count"):
    if stage == "count":
        print(json.dumps(count.remote()), flush=True)
    elif stage == "encode":
        print(json.dumps(encode_all.remote()), flush=True)
    else:
        raise SystemExit(f"unknown stage {stage}")
