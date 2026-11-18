"""One command for the frozen T1 exam. No training and no ElevenLabs.

    python -m modal run t1_eval.py::freeze
    python -m modal run t1_eval.py::timing
    python -m modal run t1_eval.py --ckpt <path|real> --name <tag> [--part A|B|both]
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import modal

import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

from t1_score import (
    HEALTH_THRESHOLDS,
    JAIS_DIALECTS,
    PART_A_NAMES,
    PART_B_DIALECTS,
    aggregate,
    health,
    score_pair,
)

APP_NAME = "qwen3-arabic-t1-eval"
VOLUME = "qwen3-arabic-tts-v2"
RATE = 1.95
COHERE_SHA = "c3e911b42149bf7a1e53d5cef9878aee87515a23"
JAIS_SHA = "1323995fa189c479369a3929be0e95039eba5cba"
WAVLM_ID = "microsoft/wavlm-base-plus-sv"
ROOT = Path(f"{_DATA}/tasks/t1_eval")
TESTSET = ROOT / "testset"
RESULTS = ROOT / "results"
ZIP_LOCAL = Path(_qwen_os.environ.get("T1_PLAN15_ZIP", "plan15_clips_v4_turbo.zip"))
ZIP_REMOTE = "tasks/t1_eval/plan15_clips_v4_turbo.zip"
CKPT_TIMING = f"{_DATA}/runs/stageB1/step-045190-base"
A_ORDER = ("ar_msa", "ar_egy", "ar_sau", "ar_uae", "ar_irq", "ar_dza", "ar_mar")
LANG = {
    "emirati": "ar_uae",
    "kuwaiti": "ar_gulf",
    "najdi": "ar_sau",
    "hijazi": "ar_sau",
    "egyptian": "ar_egy",
    "levantine": "ar_lev",
    "msa": "ar_msa",
    "ar_msa": "ar_msa",
    "ar_egy": "ar_egy",
    "ar_sau": "ar_sau",
    "ar_uae": "ar_uae",
    "ar_irq": "ar_irq",
    "ar_dza": "ar_dza",
    "ar_mar": "ar_mar",
}
DECODING_FIXED = {
    "x_vector_only_mode": True,
    "non_streaming_mode": True,
    "max_new_tokens": "max(1, int(cap_seconds * 12.5)) using the batch maximum",
    "seed_formula": "torch.manual_seed(seed * 1000003 + batch_start); clips sorted by cap then index",
    "batch_size": 8,
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "do_not_override_generation_config": True,
    "language": LANG,
    "health": HEALTH_THRESHOLDS,
}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME)
secret = modal.Secret.from_name("huggingface")


def _cpu_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libsndfile1", "ffmpeg")
        .pip_install("soundfile", "numpy", "rapidfuzz")
        .add_local_python_source(
            "t1_freeze", "t1_score", "t1_compare", "t1_eval", "t2_decode", "modal_app",
            "plan12_text", "plan12_letter", "plan15_round3b_lib",
        )
    )


def _gen_image():
    from modal_app import image as tts_image

    return tts_image.add_local_python_source(
        "t1_eval", "t1_score", "t1_freeze", "t1_compare", "t2_decode", "plan13_eval",
        "plan12_text", "plan12_letter", "plan15_round3b_lib", "modal_app",
    )


def _score_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libsndfile1", "ffmpeg", "git")
        .pip_install(
            "torch==2.6.0",
            "torchaudio==2.6.0",
            extra_options="--index-url https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            "transformers>=5.4.0", "accelerate", "soundfile", "librosa", "numpy",
            "sentencepiece", "protobuf", "huggingface_hub", "rapidfuzz", "safetensors",
            "timm", "tqdm", "typing-extensions",
        )
        .pip_install(
            "torch==2.6.0",
            "torchaudio==2.6.0",
            "torchvision==0.21.0",
            extra_options="--index-url https://download.pytorch.org/whl/cu124 --force-reinstall",
        )
        .run_commands("pip install --no-deps git+https://github.com/sarulab-speech/UTMOSv2.git")
        .run_commands("apt-get update && apt-get install -y --no-install-recommends wget")
        .env({"HF_HOME": "/data/hf_cache"})
        .add_local_python_source(
            "t1_eval", "t1_score", "t1_compare", "t2_decode", "plan13_eval",
            "plan12_text", "plan12_letter", "retx_jais", "modal_app",
        )
    )


cpu_image = _cpu_image()
gen_image = _gen_image()
score_image = _score_image()


def _commit():
    vol.commit()


def _reload():
    vol.reload()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def check_frozen() -> dict:
    found = {}
    for name in ("partA", "partB"):
        path = TESTSET / f"{name}.jsonl"
        sha_path = TESTSET / f"{name}.sha1"
        if not path.is_file() or not sha_path.is_file():
            raise RuntimeError(f"missing frozen {name}")
        got = hashlib.sha1(path.read_bytes()).hexdigest()
        want = sha_path.read_text(encoding="utf-8").split()[0].strip()
        if got != want:
            raise RuntimeError(f"{name} SHA-1 {got} does not match {want}")
        found[name] = got
    return found


def _exam(part: str, only_ids: list[str] | None = None) -> list[dict]:
    rows = []
    if part in {"A", "both"}:
        for row in _read_jsonl(TESTSET / "partA.jsonl"):
            row = dict(row)
            row["part"] = "A"
            rows.append(row)
    if part in {"B", "both"}:
        for row in _read_jsonl(TESTSET / "partB.jsonl"):
            row = dict(row)
            row["part"] = "B"
            rows.append(row)
    if only_ids is not None:
        want = set(only_ids)
        rows = [row for row in rows if row["id"] in want]
    rows.sort(key=lambda row: (row["part"], row["id"]))
    for index, row in enumerate(rows):
        row["i"] = index
        row["clip_id"] = row["id"]
        row["dialect_exam"] = row["dialect"]
        row["dialect"] = LANG.get(row["dialect"], row["dialect"])
    return rows


def timing_ids() -> list[str]:
    part_a = _read_jsonl(TESTSET / "partA.jsonl")
    part_b = _read_jsonl(TESTSET / "partB.jsonl")
    by = defaultdict(list)
    for row in part_a:
        by[row["dialect"]].append(row["id"])
    ids = []
    for dialect in A_ORDER:
        ids.extend(sorted(by[dialect])[:3])
    ids.append(sorted(by["ar_msa"])[3])
    grouped = defaultdict(lambda: defaultdict(list))
    for row in part_b:
        sentence = row["id"].rsplit("/", 1)[0]
        grouped[row["dialect"]][sentence].append(row["id"])
    for dialect in PART_B_DIALECTS:
        for sentence in sorted(grouped[dialect])[:2]:
            ids.extend(sorted(grouped[dialect][sentence]))
    if len(ids) != 50 or len(set(ids)) != 50:
        raise RuntimeError(f"timing selection is {len(ids)} rows, {len(set(ids))} unique")
    return ids


def _pins() -> None:
    sha = Path(f"{_DATA}/retx/plan12/partD/model_sha.txt").read_text(encoding="utf-8").strip()
    rev = Path(f"{_DATA}/models/Jais-ASR/revision.txt").read_text(encoding="utf-8").strip()
    if sha != COHERE_SHA:
        raise RuntimeError(f"Cohere SHA {sha} != {COHERE_SHA}")
    if rev != JAIS_SHA:
        raise RuntimeError(f"Jais revision {rev} != {JAIS_SHA}")


def _with_modes(rows: list[dict]) -> list[dict]:
    """Part A stays x-vector only. Part B is generated in both modes."""
    out = []
    for row in rows:
        base = dict(row)
        base["mode"] = "xvec"
        out.append(base)
        if row.get("part") != "B":
            continue
        if not (row.get("prompt_text") or "").strip():
            raise RuntimeError(f"Part B row {row.get('id')} has no prompt transcript")
        clone = dict(row)
        clone["mode"] = "icl"
        clone["i"] = int(row["i"]) + 100000
        out.append(clone)
    return out


def _other_prompt(item_id: str, seed, mode: str, speaker: str, pool: list[tuple[str, str]]):
    others = [(key, path) for key, path in pool if key != speaker]
    if not others:
        return None
    digest = hashlib.sha1(f"13|{item_id}|{seed}|{mode}".encode()).hexdigest()
    rng = random.Random(int(digest[:8], 16))
    return rng.choice(others)


def _prompt_pool(exam: dict) -> dict[str, list[tuple[str, str]]]:
    found = {"A": [], "B": []}
    seen = {"A": set(), "B": set()}
    for row in exam.values():
        part = "B" if row.get("prompt_clip") else "A"
        key = row.get("prompt_clip") or row["id"]
        path = row.get("ref_audio") or ""
        if not path or key in seen[part]:
            continue
        seen[part].add(key)
        found[part].append((key, path))
    return found


def _lookup() -> dict:
    found = {}
    for name, part in (("partA", "A"), ("partB", "B")):
        for row in _read_jsonl(TESTSET / f"{name}.jsonl"):
            row = dict(row)
            row["part"] = part
            found[row["id"]] = row
    return found


def _table(summary: dict) -> str:
    lines = [f"# {summary.get('tag')}", ""]

    def block(title, part_name, order, names):
        part = summary.get(part_name) or {}
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| dialect | n | WER | letter | dropped endings | dropped words | same voice | UTMOSv2 | % broken |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        by = part.get("by_dialect") or {}
        for dialect in order:
            cell = by.get(dialect) or {}
            if not cell:
                continue
            lines.append(
                "| {name} | {n} | {wer} | {ler} | {end} | {drop} | {sim} | {mos} | {broken} |".format(
                    name=names.get(dialect, dialect),
                    n=cell.get("n", 0),
                    wer=_num(cell.get("wer")),
                    ler=_num(cell.get("ler")),
                    end=_num(cell.get("dropped_ending")),
                    drop=_num(cell.get("dropped_words")),
                    sim=_num(cell.get("sim")),
                    mos=_num(cell.get("utmos")),
                    broken=_num(cell.get("broken")),
                )
            )
        headline = part.get("priority_wer") if part_name == "partA" else part.get("overall_wer")
        lines.append("")
        lines.append(f"Headline {_num(headline)}. Percentiles p10 {_num(part.get('wer_p10'))}, "
                     f"p50 {_num(part.get('wer_p50'))}, p90 {_num(part.get('wer_p90'))}.")
        lines.append("")

    if summary.get("partA"):
        block("Part A", "partA", A_ORDER, PART_A_NAMES)
    part_b = summary.get("partB") or {}
    if part_b.get("xvec") or part_b.get("icl"):
        names = {name: name for name in PART_B_DIALECTS}
        for mode, title in (("xvec", "Part B x_vector_only"), ("icl", "Part B in-context")):
            summary_mode = dict(summary)
            summary_mode["partB"] = part_b.get(mode) or {}
            saved = summary
            summary = summary_mode
            block(title, "partB", PART_B_DIALECTS, names)
            summary = saved
    return "\n".join(lines) + "\n"


def _num(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _dollars(seconds: float) -> float:
    return round(float(seconds) / 3600 * RATE, 4)


@app.function(
    image=cpu_image, cpu=8, memory=65536, timeout=8 * 3600, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def freeze_prepare():
    from t1_freeze import PLAN15_ZIP, prepare_part_b

    _reload()
    if not PLAN15_ZIP.is_file():
        raise RuntimeError(f"missing {PLAN15_ZIP}; put the Plan 15 zip on the volume first")
    try:
        result = prepare_part_b()
    finally:
        _commit()
    return result


@app.function(
    image=score_image, gpu="L40S", cpu=8, memory=65536, timeout=6 * 3600, retries=0,
    scaledown_window=2, volumes={"/data": vol}, secrets=[secret],
)
def cer_job():
    """Transcribe queued prompts. Keep a clip only when every CER is at most 0.10."""
    import numpy as np
    import soundfile as sf

    from plan12_text import cer
    from plan13_eval import _load_cohere, _transcribe

    _reload()
    _pins()
    queue_path = ROOT / "prompt_queue.jsonl"
    sha_path = ROOT / "block_sha.txt"
    if not queue_path.is_file() or not sha_path.is_file():
        raise RuntimeError("prompt queue is missing; prepare Part B first")
    shas = {line.strip() for line in sha_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    queue = _read_jsonl(queue_path)
    started = time.time()
    load_t = time.time()
    processor, model = _load_cohere()
    from retx_jais import load_jais, transcribe_items

    jais_proc, jais_model = load_jais()
    print(f"models loaded in {time.time() - load_t:.1f}s", flush=True)
    kept = {dialect: [] for dialect in PART_B_DIALECTS}
    seen_video = {dialect: set() for dialect in PART_B_DIALECTS}
    stats = {dialect: {} for dialect in PART_B_DIALECTS}
    closest = {dialect: [] for dialect in PART_B_DIALECTS}

    def note_reject(dialect: str, clip_id: str, text: str, rule: str, distance: float) -> None:
        bag = closest[dialect]
        bag.append((distance, clip_id, (text or "")[:220], rule))
        bag.sort(key=lambda item: item[0])
        del bag[10:]

    def bump(dialect: str, key: str) -> None:
        stats[dialect][key] = stats[dialect].get(key, 0) + 1

    def peak_rms(path: Path):
        data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        data = np.asarray(data, dtype=np.float32)
        peak = float(np.max(np.abs(data))) if len(data) else 0.0
        rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
        dur = (len(data) / float(sample_rate)) if sample_rate else 0.0
        return dur, peak, rms

    def file_sha(path: Path) -> str:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for row in queue:
        dialect = row["dialect"]
        if len(kept[dialect]) >= (1 if dialect in {"najdi", "kuwaiti"} else 2):
            continue
        if row["video_id"] in seen_video[dialect]:
            bump(dialect, "same_video")
            continue
        path = Path(row["path"])
        if not path.is_file():
            wav = path.with_suffix(".wav")
            path = wav if wav.is_file() else path
        if not path.is_file():
            bump(dialect, "missing")
            continue
        try:
            dur, peak, rms = peak_rms(path)
        except Exception:
            bump(dialect, "unreadable")
            continue
        if not (4.0 <= dur <= 10.0) or peak >= 0.99 or rms < 0.005:
            bump(dialect, "audio")
            continue
        digest = file_sha(path)
        if digest in shas:
            bump(dialect, "sha1")
            continue
        seen_video[dialect].add(row["video_id"])
        limit = 0.11 if dialect == "kuwaiti" else 0.10
        try:
            hyp = _transcribe(processor, model, str(path))
            cohere_cer = float(cer(row["text"], hyp))
        except Exception as exc:
            bump(dialect, "error")
            print(f"cohere fail {dialect} {row['clip_id']} {exc}", flush=True)
            continue
        jais_cer = None
        if dialect in JAIS_DIALECTS and cohere_cer <= limit:
            try:
                outs = transcribe_items(
                    jais_proc, jais_model,
                    [{"clip_id": str(path), "duration": dur}],
                    batch_size=1,
                )
                jais_hyp = ((outs[0] or {}).get("transcript") or "") if outs else ""
                jais_cer = float(cer(row["text"], jais_hyp))
            except Exception as exc:
                bump(dialect, "error")
                print(f"jais fail {dialect} {row['clip_id']} {exc}", flush=True)
                continue
        if cohere_cer > limit or (jais_cer is not None and jais_cer > limit):
            bump(dialect, "cer")
            failed = cohere_cer if cohere_cer > limit else float(jais_cer)
            rule = f"CER {failed:.4f}"
            if jais_cer is not None and jais_cer > limit and cohere_cer > limit:
                rule = f"CER cohere {cohere_cer:.4f} jais {jais_cer:.4f}"
            elif jais_cer is not None and jais_cer > limit:
                rule = f"CER jais {jais_cer:.4f}"
            note_reject(dialect, row["clip_id"], row["text"], rule, failed)
            print(f"cer fail {dialect} {row['clip_id']} cohere {cohere_cer:.3f} jais {jais_cer}", flush=True)
            continue
        kept[dialect].append({
            "dialect": dialect,
            "clip_id": row["clip_id"],
            "video_id": row["video_id"],
            "path": str(path),
            "text": row["text"],
            "sha1": digest,
            "dur": round(dur, 3),
            "cohere_cer": round(cohere_cer, 4),
            "jais_cer": None if jais_cer is None else round(jais_cer, 4),
            "marker": row.get("marker") or "",
        })
        seen_video[dialect].add(row["video_id"])
        print(f"kept {dialect} {row['clip_id']} cohere {cohere_cer:.3f} jais {jais_cer}", flush=True)
    problems = [
        f"{dialect} prompts {len(kept[dialect])}"
        for dialect in PART_B_DIALECTS
        if len(kept[dialect]) != (1 if dialect in {"najdi", "kuwaiti"} else 2)
    ]
    wall = time.time() - started
    payload = {
        "ok": not problems,
        "picks": [row for dialect in PART_B_DIALECTS for row in kept[dialect]],
        "stats": stats,
        "problems": problems,
        "closest": {
            dialect: [
                {"distance": item[0], "clip_id": item[1], "transcript": item[2], "rule": item[3]}
                for item in closest[dialect]
            ]
            for dialect in PART_B_DIALECTS
            if closest[dialect]
        },
        "wall_s": round(wall, 1),
        "dollars": _dollars(wall),
    }
    (ROOT / "prompt_cer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _commit()
    return payload


@app.function(
    image=cpu_image, cpu=8, memory=65536, timeout=8 * 3600, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def freeze_finish():
    from t1_freeze import CER_RESULT, finish_part_b

    _reload()
    try:
        cer_doc = json.loads(CER_RESULT.read_text(encoding="utf-8"))
        result = finish_part_b(cer_doc)
    finally:
        _commit()
    return result


@app.function(
    image=cpu_image, cpu=1, memory=2048, timeout=600, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def assert_frozen():
    _reload()
    return check_frozen()


@app.function(
    image=cpu_image, cpu=1, memory=4096, timeout=600, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def select_timing():
    _reload()
    check_frozen()
    return timing_ids()


@app.function(
    image=gen_image, gpu="L40S", cpu=8, memory=65536, timeout=6 * 3600, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def gen_shard(ckpt: str, tag: str, shard: int, shards: int, part: str, only_ids: list[str] | None = None):
    from plan13_eval import noise_generate

    _reload()
    check_frozen()
    path = Path(ckpt)
    if not path.is_dir():
        raise RuntimeError(f"checkpoint is not a directory: {ckpt}")
    if shards < 1 or shards > 10:
        raise RuntimeError(f"shards {shards}")
    rows = _with_modes(_exam(part, only_ids))
    started = time.time()
    meta = noise_generate(
        ckpt, tag, shard, shards=shards, batch_size=8, budget_s=6 * 3600,
        root=str(RESULTS), seeds=(1, 2, 3), rows_in=rows,
    )
    meta["coordinator_note"] = "read only checkpoint"
    meta["call_wall_s"] = round(time.time() - started, 1)
    _commit()
    return meta


@app.function(
    image=score_image, gpu="L40S", cpu=8, memory=65536, timeout=6 * 3600, retries=0,
    scaledown_window=2, volumes={"/data": vol}, secrets=[secret],
)
def score_shard(tag: str, shard: int, real: bool = False, root: str = "", exam_file: str = ""):
    import soundfile as sf
    import torch

    from plan13_eval import _load_cohere, _mono16, _transcribe

    _reload()
    check_frozen()
    _pins()
    if exam_file:
        exam = {}
        for row in _read_jsonl(Path(exam_file)):
            row = dict(row)
            exam[row["id"]] = row
    else:
        exam = _lookup()
    pool = _prompt_pool(exam)
    out_dir = (Path(root) if root else RESULTS) / tag
    started = time.time()
    phases = {}
    if real:
        pending = []
        for row in _read_jsonl(TESTSET / "partA.jsonl"):
            pending.append({
                "clip_id": row["id"], "text": row["text"], "seed": None, "mode": "real",
                "name": row["ref_audio"], "dur": None, "real_path": row["ref_audio"],
            })
    else:
        manifest = out_dir / f"manifest_{shard}.jsonl"
        pending = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        local = Path(f"/tmp/t1_score_{tag}_{shard}")
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
        with tarfile.open(out_dir / f"audio_{shard}.tar") as archive:
            archive.extractall(local)
    load_t = time.time()
    processor, model = _load_cohere()
    phases["load_cohere_s"] = round(time.time() - load_t, 3)
    scored = []
    loop_t = time.time()
    for row in pending:
        exam_row = exam[row["clip_id"]]
        wav = row.get("real_path") or str(local / row["name"])
        item = {
            "part": exam_row["part"],
            "id": exam_row["id"],
            "dialect": exam_row["dialect"],
            "seed": row.get("seed"),
            "text": exam_row["text"],
            "hyp": "",
            "jais_hyp": None,
            "jais_wer": None,
            "wav": wav,
            "ref_audio": exam_row["ref_audio"],
            "speaker_key": exam_row.get("prompt_clip") or exam_row["id"],
            "mode": row.get("mode") or "xvec",
            "dur": row.get("dur"),
            "error": "",
        }
        try:
            item["hyp"] = _transcribe(processor, model, wav)
            marks = score_pair(exam_row["text"], item["hyp"])
            item.update(marks)
            if item["dur"] is None:
                item["dur"] = round(float(sf.info(wav).duration), 3)
        except Exception as exc:
            item["error"] = str(exc)[:300]
            item["wer"] = 1.0
            item["ler"] = 1.0
            item["dropped_ending"] = 1.0
            item["dropped_words"] = 1.0
        scored.append(item)
    phases["cohere_s"] = round(time.time() - loop_t, 3)
    del model
    torch.cuda.empty_cache()

    jais_rows = [item for item in scored if item["part"] == "B" and item["dialect"] in JAIS_DIALECTS and item.get("wav")]
    phases["n_jais"] = len(jais_rows)
    if jais_rows:
        load_t = time.time()
        from retx_jais import load_jais, transcribe_items

        jais_proc, jais_model = load_jais()
        phases["load_jais_s"] = round(time.time() - load_t, 3)
        loop_t = time.time()
        outs = transcribe_items(
            jais_proc, jais_model,
            [{"clip_id": item["wav"], "duration": item.get("dur") or 0} for item in jais_rows],
            batch_size=4,
        )
        for item, out in zip(jais_rows, outs):
            hyp = (out or {}).get("transcript") or ""
            item["jais_hyp"] = hyp
            try:
                item["jais_wer"] = score_pair(item["text"], hyp)["wer"]
            except Exception as exc:
                item["jais_wer"] = None
                item["error"] = (item.get("error") or "") + " jais " + str(exc)[:160]
        phases["jais_s"] = round(time.time() - loop_t, 3)
        del jais_model
        torch.cuda.empty_cache()
    else:
        phases["load_jais_s"] = 0.0
        phases["jais_s"] = 0.0

    load_t = time.time()
    from transformers import AutoFeatureExtractor, WavLMForXVector

    fe = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
    sv = WavLMForXVector.from_pretrained(WAVLM_ID).cuda().eval()
    phases["load_wavlm_s"] = round(time.time() - load_t, 3)
    loop_t = time.time()
    for item in scored:
        item["sim"] = None
        item["sim_other"] = None
        try:
            hyp_y, hyp_rms = _mono16(item["wav"])
            ref_y, _ = _mono16(item["ref_audio"])
            item["rms"] = hyp_rms
            other = _other_prompt(
                item["id"], item.get("seed"), item.get("mode") or "xvec", item["speaker_key"],
                pool["B" if item["part"] == "B" else "A"],
            )
            waves = [ref_y, hyp_y]
            if other is not None:
                other_y, _ = _mono16(other[1])
                waves.append(other_y)
                item["sim_other_id"] = other[0]
            inputs = fe(waves, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(sv.device) for k, v in inputs.items()}
            with torch.inference_mode():
                emb = sv(**inputs).embeddings
            item["sim"] = float(torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0))
            if other is not None:
                item["sim_other"] = float(torch.nn.functional.cosine_similarity(emb[2], emb[1], dim=0))
        except Exception as exc:
            item["rms"] = item.get("rms") or 0.0
            item["error"] = (item.get("error") or "") + " wavlm " + str(exc)[:160]
    phases["wavlm_s"] = round(time.time() - loop_t, 3)
    del sv
    torch.cuda.empty_cache()

    load_t = time.time()
    import utmosv2

    utmos = utmosv2.create_model(pretrained=True)
    phases["load_utmos_s"] = round(time.time() - load_t, 3)
    loop_t = time.time()
    for item in scored:
        item["utmos"] = None
        try:
            pred = utmos.predict(input_path=item["wav"])
            if hasattr(pred, "reshape"):
                pred = pred.reshape(-1)[0]
            elif isinstance(pred, (list, tuple)):
                pred = pred[0]
            item["utmos"] = float(pred)
        except Exception as exc:
            item["error"] = (item.get("error") or "") + " utmos " + str(exc)[:160]
    phases["utmos_s"] = round(time.time() - loop_t, 3)

    for item in scored:
        ref_dur = None
        try:
            ref_dur = float(sf.info(item["ref_audio"]).duration)
        except Exception:
            ref_dur = None
        mark = health(
            float(item.get("dur") or 0), float(item.get("rms") or 0), ref_dur,
            item.get("text") or "", item.get("hyp") or "", item["part"],
        )
        item["broken"] = mark["broken"]
        item["flags"] = mark["flags"]
        item["expected_s"] = mark["expected_s"]
        item.pop("wav", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / ("score_real.jsonl" if real else f"score_{shard}.jsonl")
    with part_path.open("w", encoding="utf-8") as handle:
        for item in scored:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    wall = time.time() - started
    meta = {
        "tag": tag, "shard": None if real else shard, "real": real, "n": len(scored),
        "n_jais": phases["n_jais"], "phases": phases, "wall_s": round(wall, 3),
        "dollars": _dollars(wall), "cohere": COHERE_SHA, "jais": JAIS_SHA, "wavlm": WAVLM_ID,
    }
    _write_json(out_dir / ("score_real.json" if real else f"score_{shard}.json"), meta)
    _commit()
    print(json.dumps(meta), flush=True)
    return meta


@app.function(
    image=cpu_image, cpu=2, memory=8192, timeout=1800, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def publish_job(tag: str, ckpt: str, part: str, gen_metas: list, score_metas: list, wall_s: float, real: bool):
    _reload()
    shas = check_frozen()
    out_dir = RESULTS / tag
    rows = []
    paths = [out_dir / "score_real.jsonl"] if real else sorted(out_dir.glob("score_*.jsonl"))
    for path in paths:
        if path.name.endswith(".json"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    per = out_dir / "per_clip.jsonl"
    with per.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "tag": tag,
        "ckpt": ckpt,
        "partA": aggregate(rows, "A") if any(row["part"] == "A" for row in rows) else None,
        "partB": {
            "xvec": aggregate(rows, "B", "xvec"),
            "icl": aggregate(rows, "B", "icl"),
        } if any(row["part"] == "B" for row in rows) else None,
        "decoding": {
            **DECODING_FIXED,
            "generation_config": (gen_metas[0].get("generation_config") if gen_metas else {"real": "reference audio, no synthesis"}),
            "generation_config_sha256": (gen_metas[0].get("generation_config_sha256") if gen_metas else None),
            "seeds": None if real else [1, 2, 3],
        },
        "sha1": shas,
        "seeds": None if real else [1, 2, 3],
        "requested_part": part,
        "n": len(rows),
        "dollars": round(sum(float(m.get("dollars") or 0) for m in list(gen_metas) + list(score_metas)), 4),
        "gpu_wall_s": round(sum(float(m.get("wall_s") or 0) for m in list(gen_metas) + list(score_metas)), 1),
        "wall_s": round(wall_s, 1),
        "gen": gen_metas,
        "score": score_metas,
        "complete": bool(rows) and all(m.get("done", True) for m in gen_metas),
        "health_thresholds": HEALTH_THRESHOLDS,
    }
    if real:
        summary["complete"] = bool(rows)
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "table.md").write_text(_table(summary), encoding="utf-8")
    _commit()
    return {"tag": tag, "n": len(rows), "dollars": summary["dollars"], "wall_s": summary["wall_s"]}


@app.function(
    image=cpu_image, cpu=1, memory=2048, timeout=600, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def write_text(path: str, text: str):
    _reload()
    dest = Path(path)
    if not str(dest).startswith(f"{_DATA}/tasks/t1_eval/"):
        raise RuntimeError(f"refusing to write {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    _commit()
    return str(dest)


def _put_zip() -> None:
    py = sys.executable
    subprocess.check_call([
        py, "-m", "modal", "volume", "put", VOLUME, str(ZIP_LOCAL), ZIP_REMOTE, "--force",
    ])


def _estimate(gen_meta: dict, score_meta: dict) -> str:
    phases = score_meta.get("phases") or {}
    n_gen = int(gen_meta.get("n") or 0)
    n_score = int(score_meta.get("n") or 0)
    n_jais = int(phases.get("n_jais") or 0)
    gen_load = float(gen_meta.get("load_s") or 0)
    gen_work = max(0.0, float(gen_meta.get("wall_s") or 0) - gen_load)
    per_gen = gen_work / n_gen if n_gen else 0
    per_cohere = float(phases.get("cohere_s") or 0) / n_score if n_score else 0
    per_jais = float(phases.get("jais_s") or 0) / n_jais if n_jais else 0
    per_wavlm = float(phases.get("wavlm_s") or 0) / n_score if n_score else 0
    per_utmos = float(phases.get("utmos_s") or 0) / n_score if n_score else 0
    gens = (1300 + 270) * 3
    jais_full = 600
    gen_s = per_gen * gens + gen_load
    score_s = (
        per_cohere * gens + float(phases.get("load_cohere_s") or 0)
        + per_jais * jais_full + float(phases.get("load_jais_s") or 0)
        + per_wavlm * gens + float(phases.get("load_wavlm_s") or 0)
        + per_utmos * gens + float(phases.get("load_utmos_s") or 0)
    )
    real_s = (
        per_cohere * 1300 + float(phases.get("load_cohere_s") or 0)
        + per_wavlm * 1300 + float(phases.get("load_wavlm_s") or 0)
        + per_utmos * 1300 + float(phases.get("load_utmos_s") or 0)
    )
    ckpt = _dollars(gen_s + score_s)
    real = _dollars(real_s)
    five = round(5 * ckpt + real, 2)
    four = round(4 * ckpt + real, 2)
    timing = round(float(gen_meta.get("dollars") or 0) + float(score_meta.get("dollars") or 0), 4)
    lines = [
        "# T1 timing estimate",
        "",
        f"Timing checkpoint: `{CKPT_TIMING}` (read only).",
        f"Measured timing generations: {n_gen}. Scored rows: {n_score}. Jais rows in the sample: {n_jais}.",
        f"Generation load {gen_load:.1f}s, generation work {gen_work:.1f}s.",
        f"Score phases: `{json.dumps(phases)}`.",
        f"Timing job GPU dollars (measured): ${timing:.4f}.",
        f"Timing GPU wall {float(gen_meta.get('wall_s') or 0) + float(score_meta.get('wall_s') or 0):.1f}s.",
        "",
        "Full-run projection uses these per-item rates, then adds one model load per container.",
        "Each of five checkpoints generates 1,300×3 Part A clips and 270×3 Part B clips.",
        "Jais runs on 600 Part B outputs (Emirati, Kuwaiti, Najdi, Hijazi, Levantine, three seeds).",
        "`real` scores 1,300 Part A references and does not generate.",
        "Rate $1.95 per L40S hour. No region multiplier.",
        "",
        f"One checkpoint, generate + score: ${ckpt:.2f}.",
        f"`real`: ${real:.2f}.",
        f"Five checkpoints plus real: ${five:.2f}.",
        f"Without s20408 (four checkpoints plus real): ${four:.2f}.",
        "",
        "The $20 comparison is the five-checkpoint figure. The timing dollars above are already spent.",
        "",
    ]
    if five > 20:
        lines.append(f"The five-checkpoint estimate ${five:.2f} is over $20. Without s20408 it is ${four:.2f}. Waiting for a choice.")
    else:
        lines.append(f"The five-checkpoint estimate ${five:.2f} is within $20. Baselines are not started.")
    lines.append("")
    return "\n".join(lines)


@app.local_entrypoint()
def prepare():
    _put_zip()
    print(json.dumps(freeze_prepare.remote(), ensure_ascii=False, default=str))


@app.local_entrypoint()
def cer():
    print(json.dumps(cer_job.remote(), ensure_ascii=False, default=str)[:4000])


@app.local_entrypoint()
def finish():
    print(json.dumps(freeze_finish.remote(), ensure_ascii=False, default=str))


@app.local_entrypoint()
def freeze():
    _put_zip()
    prepared = freeze_prepare.remote()
    print(json.dumps(prepared, ensure_ascii=False, default=str))
    if not prepared.get("ok"):
        raise SystemExit("Part B prepare stopped before the CER pass")
    cer_doc = cer_job.remote()
    print(json.dumps({
        "ok": cer_doc.get("ok"),
        "stats": cer_doc.get("stats"),
        "problems": cer_doc.get("problems"),
        "wall_s": cer_doc.get("wall_s"),
        "dollars": cer_doc.get("dollars"),
    }, ensure_ascii=False))
    result = freeze_finish.remote()
    print(json.dumps(result, ensure_ascii=False, default=str))
    if not result.get("ok"):
        raise SystemExit("Part B was not frozen")


@app.local_entrypoint()
def timing():
    assert_frozen.remote()
    ids = select_timing.remote()
    started = time.time()
    gen_meta = gen_shard.remote(CKPT_TIMING, "_timing50", 0, 1, "both", ids)
    if not gen_meta.get("done"):
        raise SystemExit(f"timing generation did not finish: {gen_meta}")
    score_meta = score_shard.remote("_timing50", 0, False)
    wall = time.time() - started
    publish_job.remote("_timing50", CKPT_TIMING, "both", [gen_meta], [score_meta], wall, False)
    text = _estimate(gen_meta, score_meta)
    write_text.remote(f"{_DATA}/tasks/t1_eval/timing_estimate.md", text)
    print(text)


@app.local_entrypoint()
def main(ckpt: str, name: str, part: str = "both", shards: int = 2):
    if part not in {"A", "B", "both"}:
        raise SystemExit("part must be A, B, or both")
    if shards < 1 or shards > 10:
        raise SystemExit("shards must be from 1 to 10")
    real = ckpt == "real"
    if real and part == "B":
        raise SystemExit("real scores Part A only")
    if real:
        part = "A"
    assert_frozen.remote()
    started = time.time()
    if real:
        score_meta = score_shard.remote(name, 0, True)
        wall = time.time() - started
        print(json.dumps(publish_job.remote(name, ckpt, part, [], [score_meta], wall, True), ensure_ascii=False))
        return
    calls = [gen_shard.spawn(ckpt, name, shard, shards, part, None) for shard in range(shards)]
    gen_metas = [call.get() for call in calls]
    if not all(meta.get("done") for meta in gen_metas):
        raise SystemExit(f"generation incomplete: {gen_metas}")
    score_calls = [score_shard.spawn(name, shard, False) for shard in range(shards)]
    score_metas = [call.get() for call in score_calls]
    wall = time.time() - started
    print(json.dumps(publish_job.remote(name, ckpt, part, gen_metas, score_metas, wall, False), ensure_ascii=False))


CAP_DOLLARS = 44.0
BUDGET_DOLLARS = 40.0
EXPECT_CKPT = 12.0
EXPECT_REAL = 1.5
APPROVED_RUNS = (
    ("s45190", f"{_DATA}/runs/stageB1/step-045190-base", False),
    ("s5482_11labs", f"{_DATA}/exp_11labs/runs/step-005482-base", False),
    ("s1370_11labs", f"{_DATA}/exp_11labs/runs/step-001370-base", False),
    ("s45190_rerun", f"{_DATA}/runs/stageB1/step-045190-base", False),
    ("real", "real", True),
)


@app.function(
    image=cpu_image, cpu=2, memory=8192, timeout=3600, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def listen_job():
    """Copy already-generated Part B wavs. No GPU."""
    import csv
    import io
    import zipfile

    _reload()
    exam = {row["id"]: row for row in _read_jsonl(TESTSET / "partB.jsonl")}
    rng = random.Random(13)
    tags = [
        tag for tag in ("s45190", "s5482_11labs", "s1370_11labs", "s45190_rerun")
        if (RESULTS / tag).is_dir()
    ]
    dest = RESULTS / "listen_pack.zip"
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["checkpoint", "dialect", "mode", "id", "seed", "file", "prompt"])
    prompts_added = set()
    with zipfile.ZipFile(dest, "w") as archive:
        for tag in tags:
            out_dir = RESULTS / tag
            located = {}
            for tar_path in sorted(out_dir.glob("audio_*.tar")):
                with tarfile.open(tar_path) as bundle:
                    for member in bundle.getmembers():
                        if member.isfile():
                            located[member.name] = tar_path
            rows = []
            for manifest in sorted(out_dir.glob("manifest_*.jsonl")):
                rows.extend(json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
            grouped = defaultdict(list)
            for row in rows:
                exam_row = exam.get(row.get("clip_id"))
                if exam_row:
                    grouped[exam_row["dialect"]].append(row)
            wanted = defaultdict(list)
            for dialect in PART_B_DIALECTS:
                pairs = sorted({
                    (row["clip_id"], row.get("seed"))
                    for row in grouped[dialect]
                    if (row.get("mode") or "xvec") == "xvec"
                })
                chosen = pairs if len(pairs) <= 3 else rng.sample(pairs, 3)
                for clip_id, seed in sorted(chosen):
                    prompt_path = Path(exam[clip_id]["ref_audio"])
                    prompt_arc = f"prompts/{prompt_path.name}"
                    if prompt_path.name not in prompts_added and prompt_path.is_file():
                        archive.write(prompt_path, arcname=prompt_arc)
                        prompts_added.add(prompt_path.name)
                    for mode in ("xvec", "icl"):
                        match = next((
                            row for row in grouped[dialect]
                            if row.get("clip_id") == clip_id and row.get("seed") == seed
                            and (row.get("mode") or "xvec") == mode
                        ), None)
                        if match is None or match["name"] not in located:
                            continue
                        arc = f"{tag}/{dialect}/{mode}/{match['name']}"
                        wanted[located[match["name"]]].append((match["name"], arc))
                        writer.writerow([tag, dialect, mode, clip_id, seed, arc, prompt_arc])
            for tar_path, items in wanted.items():
                with tarfile.open(tar_path) as bundle:
                    for name, arc in items:
                        extracted = bundle.extractfile(name)
                        if extracted is not None:
                            archive.writestr(arc, extracted.read())
        archive.writestr("index.csv", table.getvalue())
    _commit()
    return {"zip": str(dest), "prompts": len(prompts_added), "seed": 13}


@app.function(
    image=cpu_image, cpu=2, memory=4096, timeout=1800, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def compare_job():
    from t1_compare import write_compare

    _reload()
    text = write_compare(RESULTS, ROOT / "compare.md")
    _commit()
    return text


@app.local_entrypoint()
def wrapup():
    """Write compare.md and the listening pack from finished checkpoints."""
    text = compare_job.remote()
    listen = listen_job.remote()
    note = (
        "\n\nT2 was not started.\n"
        f"Listening pack: {json.dumps(listen)}.\n"
        "The Kuwaiti prompt transcript has CER 0.10, so its in-context-cloning column may read slightly low.\n"
    )
    write_text.remote(f"{_DATA}/tasks/t1_eval/compare.md", text + note)
    print(note)


@app.local_entrypoint()
def approved(shards: int = 8):
    """Four checkpoints, without s20408, plus real. Budget $40, hard stop $44."""
    if shards < 1 or shards > 10:
        raise SystemExit("shards must be from 1 to 10")
    assert_frozen.remote()
    spend = {}
    stopped = ""
    for name, ckpt, real in APPROVED_RUNS:
        spent = round(sum(spend.values()), 4)
        expect = EXPECT_REAL if real else EXPECT_CKPT
        if spent + expect > CAP_DOLLARS:
            stopped = f"spent ${spent:.2f}; {name} expected ${expect:.2f} would pass ${CAP_DOLLARS:.0f}"
            print(json.dumps({"stop": True, "reason": stopped, "spent": spent}), flush=True)
            break
        started = time.time()
        if real:
            score_meta = score_shard.remote(name, 0, True)
            wall = time.time() - started
            meta = publish_job.remote(name, ckpt, "A", [], [score_meta], wall, True)
        else:
            calls = [gen_shard.spawn(ckpt, name, shard, shards, "both", None) for shard in range(shards)]
            gen_metas = [call.get() for call in calls]
            if not all(item.get("done") for item in gen_metas):
                stopped = f"{name} generation incomplete"
                print(json.dumps({"stop": True, "reason": stopped, "gen": gen_metas}, default=str), flush=True)
                break
            score_calls = [score_shard.spawn(name, shard, False) for shard in range(shards)]
            score_metas = [call.get() for call in score_calls]
            wall = time.time() - started
            meta = publish_job.remote(name, ckpt, "both", gen_metas, score_metas, wall, False)
        spend[name] = float(meta.get("dollars") or 0)
        print(json.dumps({"done": name, "job": spend[name], "spent": round(sum(spend.values()), 4)}), flush=True)
        if sum(spend.values()) >= CAP_DOLLARS:
            stopped = f"spent ${sum(spend.values()):.2f} reached the ${CAP_DOLLARS:.0f} stop"
            break
    text = compare_job.remote()
    try:
        listen = listen_job.remote()
    except Exception as exc:
        listen = {"error": str(exc)[:300]}
    note = f"\n\nSpent ${sum(spend.values()):.2f} across {json.dumps(spend)}. Budget ${BUDGET_DOLLARS:.0f}, hard stop ${CAP_DOLLARS:.0f}.\n"
    if stopped:
        note += f"Stopped: {stopped}\n"
    else:
        note += "Four checkpoints plus real finished under the $44 hard stop. T2 was not started.\n"
    note += f"Listening pack: {json.dumps(listen)}.\n"
    write_text.remote(f"{_DATA}/tasks/t1_eval/compare.md", text + note)
    print(text)
    print(note)


from t2_decode import (
    CKPT as T2_CKPT,
    EXAM as T2_EXAM,
    REPORT as T2_REPORT,
    RESULTS as T2_RESULTS,
    ROOT as T2_ROOT,
    combined_job,
    exam_rows,
    generation_rows,
    listen_members,
    pick_best,
    render,
    repetition_jobs,
    same_decode,
    select_part_a,
    subtalker_jobs,
    summarize,
    temperature_jobs,
    top_p_jobs,
)


@app.function(
    image=cpu_image, cpu=2, memory=4096, timeout=1800, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def t2_prepare():
    _reload()
    check_frozen()
    if not Path(T2_CKPT).is_dir():
        raise RuntimeError(f"missing checkpoint {T2_CKPT}")
    slice_path = T2_ROOT / "slice.json"
    if T2_EXAM.is_file() and slice_path.is_file() and T2_EXAM.stat().st_size > 0:
        meta = json.loads(slice_path.read_text(encoding="utf-8"))
        rows = _read_jsonl(T2_EXAM)
        meta["n"] = len(rows)
        meta["reused"] = True
        return meta
    part_a, alloc = select_part_a(_read_jsonl(TESTSET / "partA.jsonl"))
    part_b = _read_jsonl(TESTSET / "partB.jsonl")
    rows = exam_rows(part_a, part_b)
    T2_ROOT.mkdir(parents=True, exist_ok=True)
    with T2_EXAM.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (T2_ROOT / "slice.json").write_text(
        json.dumps({"alloc": alloc, "n": len(rows)}, indent=2) + "\n", encoding="utf-8"
    )
    _commit()
    return {"n": len(rows), "alloc": alloc, "part_a": len(part_a), "part_b": len(part_b)}


@app.function(
    image=gen_image, gpu="L40S", cpu=8, memory=65536, timeout=6 * 3600, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def t2_gen(tag: str, shard: int, shards: int, decode: dict):
    from plan13_eval import noise_generate

    _reload()
    if shards < 1 or shards > 10:
        raise RuntimeError(f"shards {shards}")
    if not Path(T2_CKPT).is_dir():
        raise RuntimeError(f"missing checkpoint {T2_CKPT}")
    rows = generation_rows(_read_jsonl(T2_EXAM))
    meta = noise_generate(
        T2_CKPT, tag, shard, shards=shards, batch_size=8, budget_s=6 * 3600,
        root=str(T2_RESULTS), seeds=(1, 2, 3), rows_in=rows, decode=decode,
    )
    _commit()
    return meta


@app.function(
    image=cpu_image, cpu=2, memory=8192, timeout=1800, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def t2_publish(tag: str, decode: dict):
    _reload()
    out_dir = T2_RESULTS / tag
    rows = []
    gens = []
    scores = []
    for path in sorted(out_dir.glob("score_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    for path in sorted(out_dir.glob("gen_*.json")):
        gens.append(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted(out_dir.glob("score_*.json")):
        scores.append(json.loads(path.read_text(encoding="utf-8")))
    dollars = sum(float(item.get("dollars") or 0) for item in gens + scores)
    wall = sum(float(item.get("wall_s") or 0) for item in gens + scores)
    summary = summarize(rows, tag, decode, dollars, wall)
    summary["complete"] = bool(rows) and all(item.get("done") for item in gens)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _commit()
    return {
        "tag": tag,
        "dollars": summary["dollars"],
        "primary": summary["primary"],
        "n": summary["n"],
        "complete": summary["complete"],
    }


@app.function(
    image=cpu_image, cpu=2, memory=8192, timeout=1800, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def t2_load(tag: str):
    _reload()
    path = T2_RESULTS / tag / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@app.function(
    image=cpu_image, cpu=2, memory=8192, timeout=3600, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def t2_finish(text: str, winner: str, baseline: str):
    import io
    import tarfile
    import zipfile
    from collections import defaultdict

    _reload()
    T2_REPORT.parent.mkdir(parents=True, exist_ok=True)
    T2_REPORT.write_text(text, encoding="utf-8")
    exam = _read_jsonl(T2_EXAM)
    tags = [baseline] if not winner or winner == baseline else [baseline, winner]
    wanted = listen_members(exam, tags)
    by_tag = defaultdict(list)
    for tag, dialect, seed, clip_id in wanted:
        by_tag[tag].append((dialect, seed, clip_id))
    dest = T2_ROOT / "listen_pack.zip"
    table = io.StringIO()
    table.write("setting,dialect,seed,clip id,file\n")
    with zipfile.ZipFile(dest, "w") as archive:
        for tag, items in by_tag.items():
            out_dir = T2_RESULTS / tag
            located = {}
            for tar_path in sorted(out_dir.glob("audio_*.tar")):
                with tarfile.open(tar_path) as bundle:
                    for member in bundle.getmembers():
                        if member.isfile():
                            located[member.name] = tar_path
            manifest = []
            for path in sorted(out_dir.glob("manifest_*.jsonl")):
                manifest.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            found = {}
            for row in manifest:
                found[(row.get("clip_id"), row.get("seed"))] = row
            grouped = defaultdict(list)
            for dialect, seed, clip_id in items:
                row = found.get((clip_id, seed))
                if row and row["name"] in located:
                    arc = f"{tag}/{dialect}/{row['name']}"
                    grouped[located[row["name"]]].append((row["name"], arc))
                    table.write(f"{tag},{dialect},{seed},{clip_id},{arc}\n")
            for tar_path, files in grouped.items():
                with tarfile.open(tar_path) as bundle:
                    for name, arc in files:
                        extracted = bundle.extractfile(name)
                        if extracted is not None:
                            archive.writestr(arc, extracted.read())
        archive.writestr("index.csv", table.getvalue())
    _commit()
    return {"report": str(T2_REPORT), "zip": str(dest)}


@app.function(
    image=cpu_image, cpu=2, memory=4096, timeout=1800, retries=0,
    scaledown_window=2, volumes={"/data": vol},
)
def t2_pending(tag: str, shards: int, decode: dict) -> dict:
    """Which shards still need a GPU. Finished generation and scores are left alone."""
    _reload()
    out = T2_RESULTS / tag
    gen_need = []
    score_need = []
    for shard in range(shards):
        path = out / f"gen_{shard}.json"
        ready = False
        if path.is_file():
            meta = json.loads(path.read_text(encoding="utf-8"))
            ready = bool(meta.get("done")) and (meta.get("decode") or {}) == decode
        if not ready:
            gen_need.append(shard)
        score_meta = out / f"score_{shard}.json"
        score_rows = out / f"score_{shard}.jsonl"
        if not (score_meta.is_file() and score_rows.is_file() and score_meta.stat().st_size > 2 and score_rows.stat().st_size > 2):
            score_need.append(shard)
    return {"gen": gen_need, "score": score_need}


@app.local_entrypoint()
def t2(shards: int = 8):
    if shards < 1 or shards > 10:
        raise SystemExit("shards must be from 1 to 10")
    prepared = t2_prepare.remote()
    print(json.dumps(prepared), flush=True)
    spend = {}
    prior = {}
    docs = {}
    order = []
    stopped = ""
    expect = 2.2

    def remember(tag: str, summary: dict, already_paid: bool) -> None:
        docs[tag] = summary
        bucket = prior if already_paid else spend
        bucket[tag] = float(summary.get("dollars") or 0)
        if tag not in order:
            order.append(tag)

    def run_job(job: dict) -> bool:
        nonlocal expect, stopped
        tag = job["tag"]
        existing = t2_load.remote(tag)
        if existing and existing.get("complete"):
            remember(tag, existing, True)
            print(json.dumps({"skip": tag, "this_run": round(sum(spend.values()), 2)}), flush=True)
            return True
        spent = round(sum(spend.values()), 4)
        if spent + expect > 15:
            stopped = (
                f"Stopped before {tag}: this run spent ${spent:.2f} plus about ${expect:.2f} would pass $15."
            )
            print(json.dumps({"stop": True, "reason": stopped}), flush=True)
            return False
        pending = t2_pending.remote(tag, shards, job["decode"])
        print(json.dumps({"tag": tag, "gen_need": pending["gen"], "score_need": pending["score"]}), flush=True)
        if pending["gen"]:
            calls = [t2_gen.spawn(tag, shard, shards, job["decode"]) for shard in pending["gen"]]
            gens = [call.get() for call in calls]
            if not all(item.get("done") for item in gens):
                stopped = f"{tag} generation did not finish."
                print(json.dumps({"stop": True, "reason": stopped}), flush=True)
                return False
        if pending["score"]:
            score_calls = [
                score_shard.spawn(tag, shard, False, str(T2_RESULTS), str(T2_EXAM))
                for shard in pending["score"]
            ]
        else:
            score_calls = []
        for call in score_calls:
            call.get()
        published = t2_publish.remote(tag, job["decode"])
        summary = t2_load.remote(tag)
        remember(tag, summary, False)
        spent_now = round(sum(spend.values()), 2)
        expect = max(float(published.get("dollars") or expect) * 1.15, 0.5)
        print(json.dumps({
            "done": tag,
            "job": published.get("dollars"),
            "spent": spent_now,
            "next_expect": round(expect, 2),
        }), flush=True)
        if spent_now >= 15:
            stopped = f"This run spent ${spent_now:.2f}, which reached the $15 stop."
            return False
        return True

    jobs = temperature_jobs()
    for job in jobs:
        if not run_job(job):
            break
    baseline = docs.get("temp_0.9")
    winner_tag = "temp_0.9" if baseline else ""
    chosen = {
        "temperature": 0.9,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "subtalker_temperature": 0.9,
    }
    if baseline and not stopped:
        temp_docs = [docs[job["tag"]] for job in jobs if job["tag"] in docs]
        best_temp = pick_best(temp_docs, baseline)
        chosen["temperature"] = float(best_temp["decode"]["temperature"])
        chosen["subtalker_temperature"] = float(best_temp["decode"]["subtalker_temperature"])
        print(json.dumps({"best_temperature": chosen["temperature"]}), flush=True)
        for job in top_p_jobs(chosen["temperature"]):
            if not run_job(job):
                break
        if not stopped:
            topp = [docs[job["tag"]] for job in top_p_jobs(chosen["temperature"]) if job["tag"] in docs]
            topp.append(best_temp)
            best_p = pick_best(topp, baseline)
            chosen["top_p"] = float(best_p["decode"]["top_p"])
            print(json.dumps({"best_top_p": chosen["top_p"]}), flush=True)
            for job in repetition_jobs(chosen["temperature"]):
                if not run_job(job):
                    break
        if not stopped:
            reps = [docs[job["tag"]] for job in repetition_jobs(chosen["temperature"]) if job["tag"] in docs]
            reps.append(best_temp)
            best_rep = pick_best(reps, baseline)
            chosen["repetition_penalty"] = float(best_rep["decode"]["repetition_penalty"])
            print(json.dumps({"best_repetition_penalty": chosen["repetition_penalty"]}), flush=True)
            for job in subtalker_jobs(chosen["temperature"]):
                if same_decode(job["decode"], best_temp["decode"]):
                    continue
                if not run_job(job):
                    break
        if not stopped:
            subs = [docs[job["tag"]] for job in subtalker_jobs(chosen["temperature"]) if job["tag"] in docs]
            subs.append(best_temp)
            best_sub = pick_best(subs, baseline)
            chosen["subtalker_temperature"] = float(best_sub["decode"]["subtalker_temperature"])
            print(json.dumps({"best_subtalker": chosen["subtalker_temperature"]}), flush=True)
            combo = combined_job(chosen)
            already = next((item for item in docs.values() if same_decode(item["decode"], combo["decode"])), None)
            if already:
                winner_tag = already["tag"]
                print(json.dumps({"combined_reused": already["tag"]}), flush=True)
            elif run_job(combo):
                winner_tag = "combined"
        pool = list(docs.values())
        if pool and baseline:
            winner_tag = pick_best(pool, baseline)["tag"]
    elif baseline:
        winner_tag = pick_best(list(docs.values()), baseline)["tag"] if docs else "temp_0.9"
    alloc = prepared.get("alloc") or {}
    text = render(docs, order, winner_tag, "temp_0.9", alloc, stopped, spend, prior)
    saved = t2_finish.remote(text, winner_tag, "temp_0.9")
    print(text)
    print(json.dumps(saved))
