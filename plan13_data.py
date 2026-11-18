"""Plan 13 Tracks 1, 3 and 4. CPU only. Writes under /data/plan13/."""
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import modal

from plan12_cohere import dialect_label
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

from plan12_e import (
    _apply_vote,
    _bad_channels,
    _skeleton,
    _tiers,
    spell_text,
)
from plan12_text import normalize_compare
from plan13_live import add_event, snapshot

app = modal.App("qwen3-arabic-plan13-data")
vol = modal.Volume.from_name("qwen3-arabic-tts-v2")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("rapidfuzz", "huggingface_hub", "datasets", "soundfile")
    .add_local_python_source(
        "plan12_text", "plan12_letter", "plan12_cohere", "plan12_e", "plan13_live",
    )
)
PART = Path(f"{_DATA}/retx/plan12/partD")
OUT = Path(f"{_DATA}/plan13")
TRAIN = Path(f"{_DATA}/tts_stageA/stageA3/train_codes.jsonl")
EXPECT_T1 = 1504.54
SECRET = modal.Secret.from_name("huggingface")


def _canon(dialect):
    if dialect == "ar-egy":
        return "ar_egy"
    if dialect == "ar-saudi":
        return "ar_sau"
    return dialect


def _commit():
    vol.commit()


def _dev_groups(pairs):
    """pairs: (group, seconds). Hold out about 2% of the seconds, whole groups only."""
    totals = defaultdict(float)
    for group, seconds in pairs:
        totals[group] += seconds
    target = sum(totals.values()) * 0.02
    ordered = sorted(totals, key=lambda group: hashlib.sha1(str(group).encode()).hexdigest())
    chosen, got = set(), 0.0
    for group in ordered:
        if got >= target:
            break
        size = totals[group]
        if got > 0 and got + size > target * 1.5 and got >= target * 0.5:
            continue
        chosen.add(group)
        got += size
    if not chosen and ordered:
        chosen.add(min(ordered, key=lambda group: totals[group]))
        got = totals[next(iter(chosen))]
    return chosen, got, sum(totals.values())


def _table(rows):
    lines = ["dialect  split  clips  hours"]
    bucket = defaultdict(lambda: [0, 0.0])
    for row in rows:
        key = (row["dialect"], row["split"])
        bucket[key][0] += 1
        bucket[key][1] += row["duration"]
    for dialect, split in sorted(bucket):
        n, sec = bucket[(dialect, split)]
        lines.append(f"{dialect}  {split}  {n:,}  {sec / 3600:.2f}")
    return "\n".join(lines)


def _load_spell():
    path = Path(f"{_DATA}/retx/plan12/train_text.jsonl")
    out = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("clip_id") or ""
            if cid:
                out[cid] = row
    return out


def _load_cohere():
    out = {}
    for path in sorted((PART / "gpu").glob("w*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("sha_ok") is False:
                    continue
                out[row["clip_id"]] = row.get("cohere") or ""
    return out


def _load_audio():
    sha, audio = {}, {}
    root = PART / "assign"
    if not root.is_dir():
        return sha, audio
    for path in sorted(root.glob("w*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                cid = row.get("clip_id") or ""
                if not cid:
                    continue
                if row.get("sha1"):
                    sha[cid] = row["sha1"]
                if row.get("audio"):
                    audio[cid] = row["audio"]
    return sha, audio


def _texts(dialect, train_row, spell_row):
    if spell_row and not spell_row.get("vote"):
        text = spell_row.get("text") or ""
        original = spell_row.get("text_original") or text
        diac = spell_row.get("text_diac") or ""
        diac2 = ""
        no_diac = bool(spell_row.get("no_diac"))
    else:
        text = (train_row or {}).get("text_plain") or (train_row or {}).get("text") or ""
        original = text
        diac = (train_row or {}).get("text_diac") or ""
        diac2 = (train_row or {}).get("text_diac2") or ""
        no_diac = False
    if no_diac:
        return text, original, "", "", True
    if dialect == "ar_msa":
        if not diac and diac2:
            diac = diac2
        diac2 = ""
    elif not diac2 and diac:
        diac2 = diac
    return text, original, diac, diac2, False


@app.function(
    image=image, volumes={"/data": vol}, cpu=4, memory=32768, timeout=6 * 3600, retries=0,
    scaledown_window=2,
)
def track1():
    vol.reload()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot("track1", {"state": "running", "note": "Scoring clips", "table": ""})
    add_event("Track 1 started. Scoring every clip with the Part E rules.")
    _commit()
    t0 = time.time()
    cohere = _load_cohere()
    print(f"cohere {len(cohere)}", flush=True)
    spell = _load_spell()
    sha, audio_of = _load_audio()
    channel_s, channel_t3 = defaultdict(float), defaultdict(float)
    pending = []
    with (PART / "our_rows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            dur = float(row.get("duration") or 0)
            dialect = row["dialect"]
            jais = row.get("jais") or ""
            audar = row.get("audar") or ""
            text_c = cohere.get(row["clip_id"], "")
            _old, new = _tiers(jais, audar, text_c)
            source = row.get("source") or "(no channel)"
            channel_s[source] += dur
            if new == "Tier 3":
                channel_t3[source] += dur
            pending.append((row["clip_id"], dialect, dur, source, int(row.get("words") or 0), new, jais, audar, text_c))
    bad = _bad_channels(channel_s, channel_t3)
    ours = []
    tier1_ids = set()
    for cid, dialect, dur, source, words, new, jais, audar, text_c in pending:
        drop = dur < 2 or dur > 20 or dur <= 0 or words / dur < 1 or source in bad
        if drop:
            tier = "dropped"
        else:
            spelled = {"text": jais, "text_original": jais, "text_diac": "", "text_diac_original": "",
                       "no_diac": False, "rules": []}
            tier, _sp, _off = _apply_vote(dialect, new, jais, audar, text_c, spelled)
        ours.append({"clip_id": cid, "source": "ours", "dialect": dialect, "duration": dur,
                      "group": source, "tier": tier})
        if tier == "Tier 1":
            tier1_ids.add(cid)
        if len(ours) % 100000 == 0:
            snapshot("track1", {"state": "running", "note": f"Our clips {len(ours):,}", "table": ""})
            _commit()
    print(f"ours {len(ours)} tier1 {len(tier1_ids)} {time.time() - t0:.0f}s", flush=True)
    del pending

    labels = {}
    for path in sorted((PART / "darbuka_dialect").glob("[0-9][0-9][0-9][0-9].jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("clip_id"):
                        labels[row["clip_id"]] = dialect_label(row.get("dialect_value"))
    darbuka = []
    darbuka_t1 = []
    seen = 0
    for path in sorted(Path(f"{_DATA}/retx/plan12/darbuka_parts").glob("[0-9][0-9][0-9][0-9].jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("_empty"):
                    continue
                jais = row.get("jais") or ""
                audar = row.get("audar") or ""
                text_c = row.get("cohere") or ""
                if not jais and not audar and not text_c:
                    continue
                dialect = labels.get(row.get("clip_id") or "", "unknown")
                dur = float(row.get("duration") or 0)
                _old, new = _tiers(jais, audar, text_c)
                words = len(normalize_compare(jais, equate=False).split())
                drop = dur < 2 or dur > 20 or dur <= 0 or words / dur < 1
                if drop:
                    tier = "dropped"
                else:
                    spelled = {"text": jais, "text_original": jais, "text_diac": "", "text_diac_original": "",
                               "no_diac": False, "rules": []}
                    tier, _sp, _off = _apply_vote(dialect, new, jais, audar, text_c, spelled)
                cid = row.get("clip_id") or ""
                group = f"tar:{path.stem}"
                darbuka.append({"clip_id": cid, "source": "darbuka", "dialect": _canon(dialect),
                                 "duration": dur, "group": group, "tier": tier, "raw_dialect": dialect})
                if tier == "Tier 1":
                    text, _fired, _lost = spell_text(jais, _canon(dialect), with_marks=False)
                    darbuka_t1.append({
                        "clip_id": cid, "source": "darbuka", "dialect": _canon(dialect),
                        "duration": dur, "group": group, "part": path.stem, "text": text,
                        "text_original": jais, "text_diac": "", "text_diac2": "", "no_diac": False,
                        "mos": row.get("mos") or "", "sha1": row.get("sha256") or "",
                    })
                seen += 1
                if seen % 100000 == 0:
                    print(f"darbuka {seen}", flush=True)
                    snapshot("track1", {"state": "running", "note": f"Darbuka {seen:,}", "table": ""})
                    _commit()

    all_rows = ours + [{"clip_id": r["clip_id"], "source": "darbuka", "dialect": r["dialect"],
                        "duration": r["duration"], "group": r["group"], "tier": r["tier"]} for r in darbuka]
    tiers_path = OUT / "tiers_v1.jsonl"
    tmp = tiers_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(tiers_path)

    t1_sec = sum(r["duration"] for r in all_rows if r["tier"] == "Tier 1")
    t1_h = t1_sec / 3600
    print(f"tier1 hours {t1_h:.2f}", flush=True)
    by = Counter()
    for row in all_rows:
        if row["tier"] == "Tier 1":
            by[row["dialect"]] += row["duration"]
    lines = [f"Tier 1 hours {t1_h:.2f}. Part E combined Tier 1 was {EXPECT_T1}."]
    for dialect, sec in sorted(by.items(), key=lambda item: -item[1]):
        lines.append(f"- {dialect}: {sec / 3600:.2f} h")
    if abs(t1_h - EXPECT_T1) > 0.3:
        lines.append("Stopped. The total does not match Part E, so tier1_v1 was not written.")
        report = "\n".join(lines) + "\n"
        (OUT / "track1_report.md").write_text(report, encoding="utf-8")
        snapshot("track1", {"state": "failed", "note": f"Tier 1 {t1_h:.2f} h, expected {EXPECT_T1}", "table": report})
        add_event(f"Track 1 stopped. Tier 1 is {t1_h:.2f} h, Part E was {EXPECT_T1}.")
        _commit()
        return {"ok": False, "hours": round(t1_h, 2)}

    dev = {}
    dev_note = []
    for dialect in by:
        pool = [(r["group"], r["duration"]) for r in all_rows if r["tier"] == "Tier 1" and r["dialect"] == dialect]
        chosen, got, total = _dev_groups(pool)
        dev[dialect] = chosen
        dev_note.append(f"- {dialect}: dev {got / 3600:.2f} h of {total / 3600:.2f} h, {len(chosen)} groups")

    kept_groups = {r["clip_id"]: r["group"] for r in all_rows if r["tier"] == "Tier 1"}
    meta = {r["clip_id"]: r for r in all_rows if r["tier"] == "Tier 1" and r["source"] == "ours"}

    ours_out = OUT / "tier1_v1.jsonl"
    tmp = ours_out.with_suffix(".jsonl.tmp")
    written = 0
    no_diac_h = Counter()
    channels = set()
    with TRAIN.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as handle:
        for line in src:
            row = json.loads(line)
            cid = row.get("clip_id") or ""
            if cid not in meta:
                continue
            info = meta[cid]
            text, original, diac, diac2, no_diac = _texts(info["dialect"], row, spell.get(cid))
            split = "dev" if info["group"] in dev.get(info["dialect"], ()) else "train"
            out_row = {
                "clip_id": cid, "source": "ours", "dialect": info["dialect"],
                "duration": info["duration"], "group": info["group"], "split": split,
                "text": text, "text_original": original, "text_diac": diac, "text_diac2": diac2,
                "no_diac": no_diac, "audio": row.get("audio") or audio_of.get(cid) or "",
                "ref_audio": row.get("audio") or audio_of.get(cid) or "",
                "sha1": sha.get(cid) or "", "audio_codes": row.get("audio_codes"),
            }
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1
            channels.add(info["group"])
            if no_diac:
                no_diac_h[info["dialect"]] += info["duration"]
            if written % 20000 == 0:
                print(f"codes {written}", flush=True)
    if written != len(meta):
        lines.append(f"Warning: training file had {written:,} of {len(meta):,} our Tier 1 clips.")
    tmp.replace(ours_out)

    index = OUT / "tier1_darbuka_index.jsonl"
    with index.open("w", encoding="utf-8") as handle, ours_out.open("a", encoding="utf-8") as dest:
        for row in darbuka_t1:
            split = "dev" if row["group"] in dev.get(row["dialect"], ()) else "train"
            row["split"] = split
            row["audio"] = ""
            row["ref_audio"] = ""
            row["audio_codes"] = None
            dest.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.write(json.dumps({
                "clip_id": row["clip_id"], "part": row["part"], "dialect": row["dialect"],
                "duration": row["duration"], "split": split, "text": row["text"],
            }, ensure_ascii=False) + "\n")

    summary_rows = []
    with ours_out.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            summary_rows.append({"dialect": row["dialect"], "split": row["split"], "duration": row["duration"]})
    table = _table(summary_rows)
    lines += ["", "Dev holdout (whole channel or tar):", *dev_note, "", table,
              "", f"Our channels in Tier 1: {len(channels):,}.",
              f"Darbuka groups: {len({r['group'] for r in darbuka_t1}):,}.",
              "no_diac hours:"]
    if not no_diac_h:
        lines.append("- none")
    for dialect, sec in sorted(no_diac_h.items()):
        lines.append(f"- {dialect}: {sec / 3600:.2f} h")
    lines.append("")
    lines.append("unknown is Darbuka clips with no dialect label. It is in the file and not folded into MSA.")
    report = "\n".join(lines) + "\n"
    (OUT / "track1_report.md").write_text(report, encoding="utf-8")
    snapshot("track1", {"state": "done", "note": f"Tier 1 {t1_h:.2f} h", "table": table})
    add_event(f"Track 1 done. Tier 1 {t1_h:.2f} h. tier1_v1.jsonl written.")
    _commit()
    print(report, flush=True)
    return {"ok": True, "hours": round(t1_h, 2), "ours": written, "darbuka": len(darbuka_t1)}


def _norm_set():
    keys = set()
    groups = set()
    with (PART / "our_rows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = normalize_compare(row.get("jais") or "", equate=True)
            if text:
                keys.add(text)
            if row.get("source"):
                groups.add(str(row["source"]))
    for path in sorted(Path(f"{_DATA}/retx/plan12/darbuka_parts").glob("[0-9][0-9][0-9][0-9].jsonl")):
        groups.add(f"tar:{path.stem}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = normalize_compare(row.get("jais") or "", equate=True)
                if text:
                    keys.add(text)
    return keys, groups


def _save_audio(example, dest: Path, name: str):
    audio = example.get("audio")
    if isinstance(audio, dict) and audio.get("bytes"):
        target = dest / f"{name}.bin"
        target.write_bytes(audio["bytes"])
        return str(target)
    if audio is None:
        return ""
    path = getattr(audio, "path", None) or (audio.get("path") if isinstance(audio, dict) else None)
    if path and Path(path).is_file():
        target = dest / Path(path).name
        target.write_bytes(Path(path).read_bytes())
        return str(target)
    array = getattr(audio, "array", None)
    sr = getattr(audio, "sampling_rate", None)
    if array is None and isinstance(audio, dict):
        array, sr = audio.get("array"), audio.get("sampling_rate")
    if array is None or not sr:
        return ""
    import soundfile as sf
    target = dest / f"{name}.wav"
    sf.write(str(target), array, int(sr))
    return str(target)


@app.function(
    image=image, volumes={"/data": vol}, secrets=[SECRET], cpu=4, memory=32768,
    timeout=6 * 3600, retries=0, scaledown_window=2,
)
def track3():
    from datasets import Audio, get_dataset_config_names, load_dataset

    def _load(repo, config, split):
        ds = load_dataset(repo, config, split=split)
        if "audio" in getattr(ds, "column_names", []):
            ds = ds.cast_column("audio", Audio(decode=False))
        return ds

    vol.reload()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot("track3", {"state": "running", "note": "Loading benchmarks", "table": ""})
    add_event("Track 3 started. Habibi and Casablanca.")
    _commit()
    keys, groups = _norm_set()
    print(f"overlap keys {len(keys)}", flush=True)
    audio_dir = OUT / "testset_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    kept, removed = [], Counter()
    notes = []

    def take(name, row, dialect, speaker):
        text = row.get("text") or row.get("transcription") or row.get("transcript") or row.get("sentence") or ""
        norm = normalize_compare(text, equate=True)
        speaker = str(speaker or "")
        if not text:
            removed["empty"] += 1
            return
        if norm in keys:
            removed[f"{name} text"] += 1
            return
        if speaker and speaker in groups:
            removed[f"{name} speaker"] += 1
            return
        path = _save_audio(row, audio_dir, f"{name}-{len(kept)}")
        kept.append({
            "clip_id": f"{name}:{len(kept)}", "source": name, "dialect": dialect,
            "text": text, "audio": path, "ref_audio": path, "speaker": speaker,
            "never_train": True,
        })

    habibi_map = {
        "UAE": "ar_uae", "EGY": "ar_egy", "Egypt": "ar_egy", "KSA": "ar_sau", "SAU": "ar_sau",
        "MSA": "ar_msa", "GLF": "ar_gulf", "GULF": "ar_gulf",
    }
    try:
        configs = get_dataset_config_names("SWivid/Habibi")
        for config in configs:
            try:
                ds = _load("SWivid/Habibi", config, "test")
            except Exception as exc:
                notes.append(f"Habibi {config}: {exc}")
                continue
            dialect = habibi_map.get(config, config)
            notes.append(f"Habibi {config} columns {getattr(ds, 'column_names', [])}")
            before = len(kept)
            for row in ds:
                take("habibi", row, dialect, row.get("speaker_id"))
            notes.append(f"Habibi {config}: kept {len(kept) - before}")
    except Exception as exc:
        notes.append(f"Habibi unavailable: {exc}")

    try:
        ds = _load("UBC-NLP/Casablanca", "UAE", "test")
        before = len(kept)
        for row in ds:
            take("casablanca", row, "ar_uae", row.get("speaker_id") or row.get("speaker"))
        notes.append(f"Casablanca UAE test: kept {len(kept) - before}")
    except Exception as exc:
        notes.append(f"Casablanca unavailable: {exc}")

    path = OUT / "testset_v1.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps({k: v for k, v in row.items() if k != "audio" or True}, ensure_ascii=False) + "\n")
    tmp.replace(path)
    counts = Counter(row["dialect"] for row in kept)
    table = "\n".join(f"{dialect}  {n}" for dialect, n in sorted(counts.items()))
    report = ["# Test set", "", f"Kept {len(kept)}.", "", table, "", "Removed:"]
    for key, n in sorted(removed.items()):
        report.append(f"- {key}: {n}")
    report += ["", *notes, "", "This set is never used for training.", ""]
    text = "\n".join(report) + "\n"
    (OUT / "track3_report.md").write_text(text, encoding="utf-8")
    snapshot("track3", {"state": "done" if kept else "failed", "note": f"{len(kept)} clips", "table": table})
    add_event(f"Track 3 done. {len(kept)} test clips. Removed {sum(removed.values())}.")
    _commit()
    print(text, flush=True)
    return {"kept": len(kept), "removed": sum(removed.values())}


def _cps(text, duration):
    if duration <= 0:
        return 0
    return len(text or "") / duration


def _audio_ok(path):
    """No clipping, audible, and not a flat noise bed. Named checks, not a speaker model."""
    import numpy as np
    import soundfile as sf

    if not path or not Path(path).is_file():
        return False
    audio, _sr = sf.read(path, always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if len(audio) < 1000:
        return False
    peak = float(np.max(np.abs(audio)))
    if peak >= 0.99 or peak < 0.01:
        return False
    window = max(1, len(audio) // 8)
    rms = [float(np.sqrt(np.mean(audio[i:i + window] ** 2))) for i in range(0, len(audio), window)]
    rms = [v for v in rms if v > 0]
    if not rms or max(rms) / min(rms) > 8:
        return False
    spec = np.abs(np.fft.rfft(audio[: min(len(audio), 24000)])) + 1e-9
    flat = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    return flat < 0.4


@app.function(
    image=image, volumes={"/data": vol}, cpu=4, memory=16384, timeout=8 * 3600, retries=0,
    scaledown_window=2,
)
def track4():
    import random

    vol.reload()
    snapshot("track4", {"state": "running", "note": "Reading Tier 1", "table": ""})
    add_event("Track 4 started in the background.")
    _commit()
    rows = []
    with (OUT / "tier1_v1.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            dur = float(row.get("duration") or 0)
            if not (3 <= dur <= 15):
                continue
            rate = _cps(row.get("text") or "", dur)
            if not (5 <= rate <= 12):
                continue
            if row.get("source") == "darbuka":
                try:
                    mos = float(row.get("mos") or 0)
                except (TypeError, ValueError):
                    mos = 0
                if mos < 3.8:
                    continue
            elif not _audio_ok(row.get("audio") or ""):
                continue
            rows.append(row)
            if len(rows) % 500 == 0:
                snapshot("track4", {"state": "running", "note": f"{len(rows):,} clips passed", "table": ""})
                _commit()
    want = {"ar_msa", "ar_egy", "ar_sau", "ar_gulf", "ar_uae"}
    by = defaultdict(list)
    for row in rows:
        if row["dialect"] in want:
            by[row["dialect"]].append(row)
    target = 120 * 3600
    picked = []
    # Balance by taking the same number of seconds from each dialect, as far as the data allows.
    caps = {dialect: sum(r["duration"] for r in items) for dialect, items in by.items()}
    share = min([target / 5] + [caps[d] for d in caps] or [0])
    for dialect, items in by.items():
        got = 0.0
        for row in items:
            if got >= share:
                break
            picked.append(row)
            got += row["duration"]
    path = OUT / "polish_v1.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in picked:
            slim = {k: row.get(k) for k in (
                "clip_id", "source", "dialect", "duration", "text", "text_original", "audio", "group", "mos",
            )}
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
    tmp.replace(path)
    rng = random.Random(12)
    listen = []
    for dialect in sorted(want):
        pool = [r for r in picked if r["dialect"] == dialect]
        listen.extend(rng.sample(pool, min(10, len(pool))))
    native_pool = []
    for dialect in sorted(want):
        pool = [r for r in rows if r["dialect"] == dialect]
        native_pool.extend(rng.sample(pool, min(20, len(pool))))
    native = OUT / "native"
    native.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, row in enumerate(native_pool):
        cards.append(
            f"<section><p><b>{html_escape(row['dialect'])}</b></p>"
            f"<p>{html_escape(row.get('text') or '')}</p>"
            f"<audio controls src='/native-audio/{i}'></audio>"
            f"<form method='post' action='/native/answer'>"
            f"<input type='hidden' name='i' value='{i}'>"
            f"<button name='answer' value='yes'>Yes</button> "
            f"<button name='answer' value='few'>1–2 words wrong</button> "
            f"<button name='answer' value='more'>More wrong</button></form></section>"
        )
    # Audio is referenced by index. The answers are collected by Tausif, not posted to testers.
    (native / "items.json").write_text(json.dumps([
        {"i": i, "clip_id": r["clip_id"], "dialect": r["dialect"], "text": r.get("text") or "",
         "audio": r.get("audio") or ""}
        for i, r in enumerate(native_pool)
    ], ensure_ascii=False), encoding="utf-8")
    (native / "page.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Native check</title></head><body>"
        "<h1>Is the text exactly what is said?</h1>"
        "<p>Yes, 1–2 words wrong, or more wrong. For Tausif only.</p>"
        + "".join(cards) + "</body></html>",
        encoding="utf-8",
    )
    hours = defaultdict(float)
    speakers = defaultdict(set)
    for row in picked:
        hours[row["dialect"]] += row["duration"]
        speakers[row["dialect"]].add(row.get("group") or "")
    table = "\n".join(
        f"{dialect}  {hours[dialect] / 3600:.2f} h  {len(speakers[dialect])} groups"
        for dialect in sorted(hours)
    )
    (OUT / "track4_report.md").write_text(
        "Checks: 3–15 s, 5–12 characters/second. Darbuka DistillMOS ≥ 3.8. "
        "Our clips: peak under 0.99, RMS window ratio under 8, spectral flatness under 0.4. "
        "No speaker-ID model was run.\n\n" + table + "\n",
        encoding="utf-8",
    )
    snapshot("track4", {"state": "done", "note": f"{sum(hours.values()) / 3600:.1f} h", "table": table})
    add_event(f"Track 4 done. Polish set {sum(hours.values()) / 3600:.1f} h. Native page has {len(native_pool)} clips.")
    _commit()
    return {"hours": round(sum(hours.values()) / 3600, 2), "listen": len(listen)}


def _skel_words(text):
    return [_skeleton(word) for word in (text or "").split() if word]


@app.function(
    image=image, volumes={"/data": vol}, cpu=4, memory=32768, timeout=6 * 3600, retries=0,
    scaledown_window=2,
)
def flag_spelling():
    """Tier 1 rows whose spelling changed. Flag no_diac only when the vowel text does not match."""
    vol.reload()
    path = OUT / "tier1_v1.jsonl"
    changed_n, changed_s = Counter(), Counter()
    vowel_n, vowel_s = Counter(), Counter()
    mismatch_n, mismatch_s = Counter(), Counter()
    match_n = Counter()
    flag_ids = set()
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seen += 1
            if seen % 100000 == 0:
                print(f"scanned {seen}", flush=True)
            text = row.get("text") or ""
            original = row.get("text_original") or ""
            if text == original:
                continue
            dialect = row.get("dialect") or ""
            dur = float(row.get("duration") or 0)
            changed_n[dialect] += 1
            changed_s[dialect] += dur
            vowel = row.get("text_diac2") or row.get("text_diac") or ""
            if not vowel:
                continue
            vowel_n[dialect] += 1
            vowel_s[dialect] += dur
            if _skel_words(vowel) == _skel_words(text):
                match_n[dialect] += 1
                continue
            mismatch_n[dialect] += 1
            mismatch_s[dialect] += dur
            flag_ids.add(row.get("clip_id") or "")
    lines = [
        "# Tier 1 spelling",
        "",
        "A row counts as changed when its training text differs from text_original.",
        "Vowel text matches when its letters, after removing vowel marks, equal the new text.",
        "Only a mismatch is flagged no_diac. A match was already rewritten with the new spelling.",
        "",
        "Changed text:",
    ]
    if not changed_n:
        lines.append("- none")
    for dialect in sorted(changed_n):
        lines.append(
            f"- {dialect}: {changed_n[dialect]:,} clips, {changed_s[dialect] / 3600:.2f} h "
            f"(vowel text {vowel_n[dialect]:,}, matches new spelling {match_n[dialect]:,}, "
            f"mismatch {mismatch_n[dialect]:,} / {mismatch_s[dialect] / 3600:.2f} h)"
        )
    lines.append("")
    text = "\n".join(lines) + "\n"
    if flag_ids:
        tmp = path.with_suffix(".jsonl.tmp")
        with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dest:
            for line in src:
                row = json.loads(line)
                if row.get("clip_id") in flag_ids:
                    row["no_diac"] = True
                    row["text_diac"] = ""
                    row["text_diac2"] = ""
                dest.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
    (OUT / "spelling_tier1.md").write_text(text, encoding="utf-8")
    (OUT / "spelling_tier1.done").write_text("ok\n", encoding="utf-8")
    add_event(
        "Tier 1 spelling check done. "
        + ("No row changed." if not changed_n else f"Mismatch flagged: {len(flag_ids):,}.")
    )
    _commit()
    print(text, flush=True)
    return {"changed": sum(changed_n.values()), "flagged": len(flag_ids)}


def html_escape(text):
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@app.local_entrypoint()
def go(which: str = "track1"):
    fn = {"track1": track1, "track3": track3, "track4": track4, "flag": flag_spelling}[which]
    print(fn.spawn().object_id)
