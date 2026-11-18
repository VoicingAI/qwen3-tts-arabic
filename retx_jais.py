"""Plan 1: freeze Stage A paths and transcribe them with a pinned Jais-ASR checkout.

The Modal app is retx_app.py. This file holds the freeze, batch, and merge logic.
Plan 2's mark rules are not applied here.
"""
import hashlib
import io
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

FOLDER_HINT = {
    "emirati": "AE",
    "kuwaiti": "KW",
    "qatari": "QA",
    "bahraini": "BH",
    "sa-najdi": "SA",
    "sa-hijazi": "SA",
    "sa-unsure": "SA",
    "eg": "EG",
    "msa": "msa",
    "omani": "OM",
    "levantine": "LEV",
    "iraqi": "IQ",
    "moroccan": "MA",
    "tunisian": "TN",
    "algerian": "DZ",
}

SAMPLE_HINTS = ["AE", "KW", "QA", "BH", "SA", "EG"]
CHUNK = 2000
BATCH = 48
RETX = Path(f"{_DATA}/retx")
MODEL_DIR = Path(f"{_DATA}/models/Jais-ASR")
MODEL_ID = "IFM/Jais-ASR"
TRAIN = Path(f"{_DATA}/tts_stageA/stageA_train.jsonl")
VAL = Path(f"{_DATA}/tts_stageA/stageA_val.jsonl")
EVAL = Path(f"{_DATA}/tts_stageA/eval.jsonl")


def hint_for(folder):
    if folder not in FOLDER_HINT:
        raise SystemExit(f"unknown folder {folder!r}; stopping, no code guessed")
    return FOLDER_HINT[folder]


def norm_dialect(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]") and len(text) >= 2:
        text = text[1:-1].strip()
    if text.lower() == "msa":
        return "msa"
    return text.upper()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def freeze(train_path=TRAIN, val_path=VAL, eval_path=EVAL, out_dir=RETX):
    """Write input.jsonl once. No hashes. Long eval rows are counted and skipped."""
    out_dir = Path(out_dir)
    out = out_dir / "input.jsonl"
    meta_path = out_dir / "freeze.json"
    if out.exists():
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        print(f"input.jsonl already frozen ({meta.get('rows', '?')} rows); not rewriting", flush=True)
        return meta

    import soundfile as sf

    seen = {}
    order = []

    def add(clip_id, folder, dialect, duration, text):
        if clip_id in seen:
            return
        seen[clip_id] = {
            "clip_id": clip_id,
            "folder": folder,
            "folder_hint": hint_for(folder),
            "dialect": dialect,
            "duration": duration,
            "text": text,
        }
        order.append(clip_id)

    for path in (train_path, val_path):
        for row in read_jsonl(path):
            add(row["audio"], row["folder"], row.get("dialect", ""), row.get("duration"), row.get("text", ""))

    skipped_long = 0
    for row in read_jsonl(eval_path):
        if row.get("kind") == "long":
            skipped_long += 1
            continue
        if row.get("kind") != "short":
            raise SystemExit(f"eval row is neither short nor long: {row.keys()}")
        audio = row["gt_audio"]
        folder = Path(audio).parent.name
        info = sf.info(audio)
        add(audio, folder, row.get("dialect", ""), info.duration, row.get("text", ""))

    rows = [seen[k] for k in order]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    meta = {
        "rows": len(rows),
        "skipped_long": skipped_long,
        "written": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)
    return meta


def resolve_sha():
    from huggingface_hub import HfApi
    return HfApi().model_info(MODEL_ID).sha


def pin(sha=None):
    """Download one commit into /data/models/Jais-ASR. A passed sha is not re-resolved."""
    from huggingface_hub import snapshot_download

    marker = MODEL_DIR / "revision.txt"
    weights = list(MODEL_DIR.rglob("*.safetensors"))
    if sha is None and marker.exists() and weights:
        sha = marker.read_text().strip()
        print(f"reusing pinned sha {sha}", flush=True)
        return sha
    if sha is None:
        sha = resolve_sha()
        print(f"resolved sha {sha}", flush=True)
    if marker.exists() and marker.read_text().strip() == sha and weights:
        print(f"already pinned {sha}", flush=True)
        return sha
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, revision=sha, local_dir=str(MODEL_DIR))
    marker.write_text(sha + "\n")
    print(f"pinned {sha} at {MODEL_DIR}", flush=True)
    return sha


def sha1_bytes(raw):
    return hashlib.sha1(raw).hexdigest()


def read_wave(raw):
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return data, int(sr)


def load_jais(model_dir=MODEL_DIR):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    return processor, model


def _field(result, name, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def transcribe_items(processor, model, items, batch_size=BATCH):
    """Transcribe rows. SHA-1 is of the bytes that were decoded. Returns rows in input order."""
    import torch

    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio"}]}],
        tokenize=False, add_generation_prompt=True,
    )
    indexed = list(enumerate(items))
    indexed.sort(key=lambda pair: (pair[1].get("duration") or 0, pair[0]))
    out = [None] * len(items)
    for start in range(0, len(indexed), batch_size):
        group = indexed[start:start + batch_size]
        waves, rates, ready = [], [], []
        for idx, item in group:
            path = item["clip_id"]
            try:
                raw = Path(path).read_bytes()
                wave, sr = read_wave(raw)
            except Exception as exc:
                out[idx] = {
                    "clip_id": path,
                    "sha1": "",
                    "transcript": "",
                    "jais_dialect": "",
                    "folder_hint": item.get("folder_hint", ""),
                    "refused": True,
                    "truncated": False,
                    "language": "",
                    "error": str(exc),
                }
                continue
            ready.append((idx, item, sha1_bytes(raw)))
            waves.append(wave)
            rates.append(sr)
        if not waves:
            continue
        inputs = processor(
            text=[prompt] * len(waves),
            audio=waves,
            sampling_rate=rates,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=256, language="ar"
            )
        results = processor.post_process_transcription(
            generated, input_lengths=inputs["input_ids"].shape[1]
        )
        if len(results) != len(ready):
            raise RuntimeError(f"post_process returned {len(results)} for {len(ready)} clips")
        for (idx, item, digest), result in zip(ready, results):
            language = _field(result, "language", "") or ""
            out[idx] = {
                "clip_id": item["clip_id"],
                "sha1": digest,
                "transcript": _field(result, "text", "") or "",
                "jais_dialect": norm_dialect(language),
                "folder_hint": item.get("folder_hint", ""),
                "refused": bool(_field(result, "refused", False)),
                "truncated": bool(_field(result, "truncated", False)),
                "language": str(language),
            }
        print(f"batch {start + len(group)}/{len(items)}", flush=True)
    if any(row is None for row in out):
        raise RuntimeError("a clip was neither transcribed nor marked refused")
    return out


def pick_smoke(rows, n=20, folders=6):
    chosen = []
    seen = set()
    for row in rows:
        if row["folder"] not in seen:
            chosen.append(row)
            seen.add(row["folder"])
        if len(seen) >= folders and len(chosen) >= n:
            break
    if len(seen) < folders:
        raise SystemExit(f"smoke needs {folders} folders, found {sorted(seen)}")
    for row in rows:
        if len(chosen) >= n:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen[:n]


def run_listed(rows, out_path, processor=None, model=None):
    own = processor is None
    if own:
        processor, model = load_jais()
    t0 = time.time()
    done = transcribe_items(processor, model, rows)
    elapsed = time.time() - t0
    write_jsonl(out_path, done)
    return done, elapsed


def write_chunks(rows, chunk_dir, size=CHUNK):
    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for i in range(0, len(rows), size):
        cid = f"chunk_{i // size:04d}"
        path = chunk_dir / f"{cid}.jsonl"
        if not path.exists():
            write_jsonl(path, rows[i:i + size])
        ids.append(cid)
    return ids


def merge_check(input_rows, shard_dir):
    got = []
    for path in sorted(Path(shard_dir).glob("chunk_*.jsonl")):
        got.extend(read_jsonl(path))
    ids = [r["clip_id"] for r in got]
    dupes = len(ids) - len(set(ids))
    want = {r["clip_id"] for r in input_rows}
    have = set(ids)
    return {
        "ok": dupes == 0 and want == have,
        "input": len(want),
        "output": len(ids),
        "unique_output": len(have),
        "duplicates": dupes,
        "missing": len(want - have),
        "extra": len(have - want),
        "rows": got,
    }


def report(input_rows, merged_rows, meta, started, finished):
    by_id = {r["clip_id"]: r for r in input_rows}
    refused = {}
    truncated = {}
    for row in merged_rows:
        hint = row.get("folder_hint") or "?"
        if row.get("refused") or not (row.get("transcript") or "").strip():
            refused[hint] = refused.get(hint, 0) + 1
        if row.get("truncated"):
            truncated[hint] = truncated.get(hint, 0) + 1
    rng = random.Random(0)
    lines = [
        f"merge {'pass' if meta['ok'] else 'FAIL'}",
        f"input {meta['input']} output {meta['output']} unique {meta['unique_output']} "
        f"duplicates {meta['duplicates']} missing {meta['missing']} extra {meta['extra']}",
        f"wall {finished} end, start {started}",
        f"skipped long eval rows: {meta.get('skipped_long')}",
        f"refused or empty by folder_hint: {json.dumps(refused, sort_keys=True)}",
        f"truncated by folder_hint: {json.dumps(truncated, sort_keys=True)}",
        "",
    ]
    grouped = {}
    for row in merged_rows:
        grouped.setdefault(row.get("folder_hint"), []).append(row)
    for hint in SAMPLE_HINTS:
        pool = grouped.get(hint, [])
        lines.append(f"## {hint} ({len(pool)} rows, seed 0, five samples)")
        for row in rng.sample(pool, min(5, len(pool))):
            old = by_id.get(row["clip_id"], {}).get("text", "")
            lines.append(
                json.dumps({
                    "clip_id": row["clip_id"],
                    "whisper": old,
                    "jais": row.get("transcript", ""),
                    "jais_dialect": row.get("jais_dialect", ""),
                    "folder_hint": row.get("folder_hint", ""),
                }, ensure_ascii=False)
            )
        lines.append("")
    return "\n".join(lines)


def finish(started, skipped_long):
    """Merge jais-1 shards, check clip ids, and write run.json plus the report."""
    work = RETX / "jais-1"
    rows = read_jsonl(RETX / "input.jsonl")
    checked = merge_check(rows, work / "shards")
    merged_path = work / "merged.jsonl"
    write_jsonl(merged_path, checked["rows"])
    run_path = work / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    finished = datetime.now(timezone.utc).isoformat()
    run.update({
        "language": "ar",
        "started": run.get("started", started),
        "finished": finished,
        "input": checked["input"],
        "output": checked["output"],
        "duplicates": checked["duplicates"],
        "missing": checked["missing"],
        "extra": checked["extra"],
        "merge_ok": checked["ok"],
        "skipped_long": skipped_long,
    })
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    text = report(rows, checked["rows"], {**checked, "skipped_long": skipped_long}, run["started"], finished)
    (work / "report.md").write_text(text + "\n")
    print(text, flush=True)
    return text


def selftest():
    assert hint_for("emirati") == "AE"
    assert hint_for("sa-hijazi") == "SA"
    assert hint_for("msa") == "msa"
    assert norm_dialect("[EG]") == "EG"
    assert norm_dialect("eg") == "EG"
    assert norm_dialect("msa") == "msa"
    assert norm_dialect("MSA") == "msa"
    rows = [{"clip_id": f"/a/{i}.flac", "folder": name, "folder_hint": hint_for(name), "duration": 1, "text": "x"}
            for i, name in enumerate(["emirati", "kuwaiti", "qatari", "bahraini", "eg", "msa", "emirati"])]
    smoke = pick_smoke(rows, n=6, folders=6)
    assert len({r["folder"] for r in smoke}) == 6
    try:
        hint_for("nope")
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown folder should stop")
    print("selftest ok")


if __name__ == "__main__":
    selftest()
