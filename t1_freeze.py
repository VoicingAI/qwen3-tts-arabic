"""Freeze the T1 exam. CPU only. Writes under /data/tasks/t1_eval/."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from plan12_text import _edit, normalize_compare
from plan15_round3b_lib import FOLDER, PICKS, RETIRED
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

ROOT = Path(f"{_DATA}/tasks/t1_eval")
TESTSET = ROOT / "testset"
PROMPTS = TESTSET / "prompts"
REPORT = ROOT / "freeze_report.md"
PART_A_SRC = Path(f"{_DATA}/plan13/testset_score.jsonl")
PART_A_POOL = Path(f"{_DATA}/plan13/testset_v1.jsonl")
TRACK3 = Path(f"{_DATA}/plan13/track3_report.md")
TIER1 = Path(f"{_DATA}/plan13/tier1_v1.jsonl")
OUR_ROWS = Path(f"{_DATA}/retx/plan12/partD/our_rows.jsonl")
DARBUKA = Path(f"{_DATA}/retx/plan12/darbuka_parts")
TRAIN_CODES = Path(f"{_DATA}/tts_stageA/stageA3/train_codes.jsonl")
PLAN14 = (
    Path(f"{_DATA}/exp_11labs/manifest/train.jsonl"),
    Path(f"{_DATA}/exp_11labs/manifest/val.jsonl"),
)
PLAN15_ZIP = Path(f"{_DATA}/tasks/t1_eval/plan15_clips_v4_turbo.zip")
YT = Path(f"{_DATA}/yt/yt_clean_24k")
YT_RAW = Path(f"{_DATA}/yt/raw")
PART_A_SHA = "b139e7c1214c562ac93407c537dab48146d0b3d7"
QUEUE = ROOT / "prompt_queue.jsonl"
SHA_BLOCK = ROOT / "block_sha.txt"
CER_RESULT = ROOT / "prompt_cer.json"
META = ROOT / "partb_meta.json"
JAIS_PROMPT = {"emirati", "kuwaiti", "najdi", "hijazi", "levantine"}

EXPECTED_A = {
    "ar_msa": 250,
    "ar_egy": 250,
    "ar_sau": 250,
    "ar_uae": 250,
    "ar_irq": 100,
    "ar_dza": 100,
    "ar_mar": 100,
}
A_NAMES = {
    "ar_msa": "MSA",
    "ar_egy": "Egyptian",
    "ar_sau": "Saudi",
    "ar_uae": "Emirati",
    "ar_irq": "Iraqi",
    "ar_dza": "Algerian",
    "ar_mar": "Moroccan",
}
B_ORDER = ("emirati", "kuwaiti", "najdi", "hijazi", "egyptian", "levantine", "msa")
MANIFEST_DIALECT = {
    "emirati": "emirati",
    "kuwaiti": "kuwaiti",
    "saudi_najdi": "najdi",
    "saudi_hijazi": "hijazi",
    "egyptian": "egyptian",
    "levantine": "levantine",
    "msa": "msa",
}
QUIZ_IDS = {clip_id for clip_id, _dialect in PICKS}
RETIRED_IDS = {clip_id for clip_id, _dialect in RETIRED}
HEARD_IDS = QUIZ_IDS | RETIRED_IDS


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha(path: Path) -> str:
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    (path.parent / f"{path.stem}.sha1").write_text(digest + "\n", encoding="utf-8")
    return digest


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _keys_of(clip_id: str) -> set[str]:
    text = str(clip_id or "")
    keys = {text}
    name = Path(text).name
    if name:
        keys.add(name)
        keys.add(Path(name).stem)
    return {k for k in keys if k}


def video_id_of(clip_id: str) -> str:
    stem = Path(Path(str(clip_id or "")).name).stem
    if "_" in stem:
        head, tail = stem.rsplit("_", 1)
        if head and tail.isdigit():
            return head
    return stem


def median_f0(samples: np.ndarray, sample_rate: int, fmin: float = 70.0, fmax: float = 400.0):
    """Copied from TTSDataFilter pitch.median_f0. The filter package is not imported."""
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio - np.mean(audio)
    if audio.size < sample_rate * 0.1 or np.max(np.abs(audio)) < 1e-4:
        return None
    frame = int(0.04 * sample_rate)
    hop = int(0.01 * sample_rate)
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(frame - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return None
    window = np.hamming(frame)
    found = []
    for start in range(0, audio.size - frame, hop):
        segment = audio[start:start + frame] * window
        corr = np.correlate(segment, segment, mode="full")[frame - 1:]
        if corr[0] <= 0:
            continue
        lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        if corr[lag] / corr[0] < 0.3:
            continue
        found.append(sample_rate / lag)
    if not found:
        return None
    return float(np.median(found))


def gender_from_f0(f0, line_hz: float = 175.0) -> str:
    if f0 is None:
        return "unknown"
    return "male" if f0 < line_hz else "female"


def audio_stats(path: Path) -> dict:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
    dur = (len(data) / float(sr)) if sr else 0.0
    return {
        "dur": dur,
        "peak": peak,
        "rms": rms,
        "f0": None if (pitch := median_f0(data, int(sr))) is None else round(float(pitch), 2),
        "gender": gender_from_f0(pitch),
    }


def _hit(needle: set, value: str) -> bool:
    return bool(value) and value in needle


def _stream_hits(path: Path, needles: set, field: str, label: str) -> int:
    if not path.is_file() or not needles:
        return 0
    hits = 0
    seen = 0
    for row in iter_jsonl(path):
        seen += 1
        value = row.get(field) or ""
        if field == "text":
            value = normalize_compare(value)
        elif field == "jais":
            value = normalize_compare(value)
        if _hit(needles, value):
            hits += 1
        if seen % 200000 == 0:
            print(f"{label} {seen} hits {hits}", flush=True)
    print(f"{label} done {seen} hits {hits}", flush=True)
    return hits


def _id_hits(path: Path, needles: set, label: str) -> int:
    hits = 0
    seen = 0
    if not path.is_file():
        return 0
    for row in iter_jsonl(path):
        seen += 1
        if _keys_of(row.get("clip_id") or "") & needles:
            hits += 1
        if seen % 200000 == 0:
            print(f"{label} {seen} id hits {hits}", flush=True)
    print(f"{label} id done {seen} hits {hits}", flush=True)
    return hits


def _load_blocked(paths, split_train_only: bool) -> tuple[set, set, str]:
    ids, shas = set(), set()
    hash_field = ""
    for path in paths:
        if not Path(path).is_file():
            continue
        seen = 0
        for row in iter_jsonl(path):
            seen += 1
            if split_train_only and row.get("split") != "train":
                continue
            ids.update(_keys_of(row.get("clip_id") or ""))
            if not hash_field:
                if row.get("sha1"):
                    hash_field = "sha1"
                elif row.get("audio_sha1"):
                    hash_field = "audio_sha1"
            if hash_field and row.get(hash_field):
                shas.add(str(row[hash_field]))
            if seen % 200000 == 0:
                print(f"blocked {path.name} {seen}", flush=True)
        print(f"blocked {path.name} rows {seen} ids {len(ids)} sha {len(shas)} field {hash_field or '(none)'}", flush=True)
    return ids, shas, hash_field


def _stage_sets() -> tuple[set, set, str]:
    ours, train, hash_field = set(), set(), ""
    for row in iter_jsonl(OUR_ROWS):
        cid = row.get("clip_id") or ""
        if cid:
            ours.add(cid)
    print(f"our_rows ids {len(ours)}", flush=True)
    seen = 0
    for row in iter_jsonl(TRAIN_CODES):
        seen += 1
        cid = row.get("clip_id") or ""
        if cid:
            train.add(cid)
        if not hash_field:
            if "sha1" in row:
                hash_field = "sha1"
            elif "audio_sha1" in row:
                hash_field = "audio_sha1"
        if seen % 200000 == 0:
            print(f"train_codes {seen}", flush=True)
    print(f"train_codes ids {len(train)} hash_field {hash_field or '(none)'}", flush=True)
    return ours, train, hash_field


def freeze_part_a(lines: list[str]) -> dict:
    if not PART_A_SRC.is_file():
        raise RuntimeError(f"missing {PART_A_SRC}")
    rows = []
    for row in iter_jsonl(PART_A_SRC):
        rows.append({
            "id": row.get("clip_id") or "",
            "dialect": row.get("dialect") or "",
            "text": row.get("text") or "",
            "ref_audio": row.get("ref_audio") or row.get("audio") or "",
            "source": row.get("source") or "",
        })
    speakers = {}
    if PART_A_POOL.is_file():
        for row in iter_jsonl(PART_A_POOL):
            speakers[row.get("clip_id") or ""] = str(row.get("speaker") or "")
    counts = Counter(row["dialect"] for row in rows)
    import soundfile as sf

    seconds = Counter()
    missing_audio = []
    audio_sha = {}
    for row in rows:
        path = Path(row["ref_audio"])
        if not path.is_file():
            missing_audio.append(row["ref_audio"])
            continue
        try:
            seconds[row["dialect"]] += float(sf.info(path).duration)
        except Exception as exc:
            missing_audio.append(f"{row['ref_audio']} ({exc})")
            continue
        audio_sha[row["id"]] = file_sha1(path)
    norms = {normalize_compare(row["text"]) for row in rows if row["text"]}
    norms.discard("")
    ids = set()
    for row in rows:
        ids.update(_keys_of(row["id"]))
    speaker_values = {speakers.get(row["id"], "") for row in rows}
    speaker_values.discard("")

    text_hits = {
        "tier1 text": _stream_hits(TIER1, norms, "text", "tier1 text"),
        "plan14 train text": _stream_hits(PLAN14[0], norms, "text", "plan14 train text"),
        "plan14 val text": _stream_hits(PLAN14[1], norms, "text", "plan14 val text"),
    }
    darbuka_text = 0
    if DARBUKA.is_dir():
        for path in sorted(DARBUKA.glob("[0-9][0-9][0-9][0-9].jsonl")):
            darbuka_text += _stream_hits(path, norms, "jais", f"darbuka {path.name}")
    text_hits["darbuka jais"] = darbuka_text

    id_hits = {
        "tier1": _id_hits(TIER1, ids, "tier1 id"),
        "plan14 train": _id_hits(PLAN14[0], ids, "plan14 train id"),
        "plan14 val": _id_hits(PLAN14[1], ids, "plan14 val id"),
    }
    darbuka_ids = 0
    darbuka_groups = set()
    if DARBUKA.is_dir():
        for path in sorted(DARBUKA.glob("[0-9][0-9][0-9][0-9].jsonl")):
            darbuka_groups.add(f"tar:{path.stem}")
            darbuka_ids += _id_hits(path, ids, f"darbuka id {path.name}")
    id_hits["darbuka"] = darbuka_ids

    tier_groups = set()
    tier_sha = set()
    plan14_sha = set()
    plan14_hash_field = ""
    for row in iter_jsonl(TIER1):
        if row.get("group"):
            tier_groups.add(str(row["group"]))
        if row.get("sha1"):
            tier_sha.add(str(row["sha1"]))
    for path in PLAN14:
        if not path.is_file():
            continue
        for row in iter_jsonl(path):
            if not plan14_hash_field:
                if row.get("sha1"):
                    plan14_hash_field = "sha1"
                elif row.get("audio_sha1"):
                    plan14_hash_field = "audio_sha1"
            if plan14_hash_field and row.get(plan14_hash_field):
                plan14_sha.add(str(row[plan14_hash_field]))
    sha_hits = sum(1 for digest in audio_sha.values() if digest in tier_sha or digest in plan14_sha)
    speaker_hits = len(speaker_values & tier_groups) + len(speaker_values & darbuka_groups)

    ours, train_ids, train_hash_field = _stage_sets()
    only_train = len(train_ids - ours)
    only_ours = len(ours - train_ids)

    count_bad = []
    for dialect, want in EXPECTED_A.items():
        got = counts.get(dialect, 0)
        if abs(got - want) > 10:
            count_bad.append(f"{dialect} {got} expected {want}")
    extra = sorted(set(counts) - set(EXPECTED_A))
    problems = []
    if len(rows) != 1300:
        problems.append(f"row count {len(rows)} not 1300")
    problems.extend(count_bad)
    if extra:
        problems.append(f"unexpected dialects {extra}")
    if missing_audio:
        problems.append(f"unreadable audio {len(missing_audio)} e.g. {missing_audio[:3]}")
    for name, n in text_hits.items():
        if n:
            problems.append(f"sentence overlap {name}: {n}")
    for name, n in id_hits.items():
        if n:
            problems.append(f"clip id overlap {name}: {n}")
    if sha_hits:
        problems.append(f"audio sha1 overlap {sha_hits}")
    if speaker_hits:
        problems.append(f"speaker overlap {speaker_hits}")

    lines.append("# T1 freeze")
    lines.append("")
    lines.append("## Part A")
    lines.append("")
    lines.append(f"Source: `{_DATA}/plan13/testset_score.jsonl`, the seed-13 subset of Track 3 (`testset_v1.jsonl`). Not rebuilt.")
    lines.append("")
    if TRACK3.is_file():
        lines.append("Track 3 report, as stored:")
        lines.append("")
        lines.append("```")
        lines.append(TRACK3.read_text(encoding="utf-8").rstrip())
        lines.append("```")
        lines.append("")
    lines.append("Track 3 removed 1 Habibi sentence and 3 Casablanca sentences. That check compared normalised Jais text from `our_rows.jsonl` and Darbuka parts. Speaker ids were compared to YouTube channel ids and `tar:NNNN` names. This re-check found "
                 + ("no remaining intersection." if not any(text_hits.values()) and not any(id_hits.values()) and not sha_hits and not speaker_hits else "an intersection, so Part A was not frozen."))
    lines.append("")
    lines.append("Speaker namespace: Habibi `speaker_id` does not share a namespace with YouTube `source` or Darbuka tar stems. The stored Track 3 report lists no speaker removals. This re-check compared those speaker strings to Tier 1 `group` and Darbuka `tar:` names.")
    lines.append("")
    lines.append("| dialect | items | seconds |")
    lines.append("| --- | ---: | ---: |")
    for dialect in list(EXPECTED_A) + extra:
        lines.append(f"| {A_NAMES.get(dialect, dialect)} | {counts.get(dialect, 0)} | {seconds.get(dialect, 0):.1f} |")
    lines.append(f"| all | {len(rows)} | {sum(seconds.values()):.1f} |")
    lines.append("")
    lines.append("Overlap re-check (a hit stops the freeze):")
    lines.append("")
    for name, n in {**text_hits, **{f"id {k}": v for k, v in id_hits.items()}}.items():
        lines.append(f"- {name}: {n}")
    lines.append(f"- audio SHA-1 against Tier 1 `sha1` and Plan 14 `{plan14_hash_field or 'no hash field'}`: {sha_hits}")
    lines.append(f"- speaker against Tier 1 groups ({len(tier_groups)}) and Darbuka tar names ({len(darbuka_groups)}): {speaker_hits}")
    lines.append("")
    lines.append("## Stage A3 coverage")
    lines.append("")
    lines.append(f"`plan13_data._norm_set` reads `{_DATA}/retx/plan12/partD/our_rows.jsonl` (the file next to the Part D weights). It never opens `{_DATA}/tts_stageA/stageA3/train_codes.jsonl`. Track 3 did not compare Stage A3 clip ids or audio SHA-1s.")
    lines.append("")
    lines.append(f"- `our_rows.jsonl` ids: {len(ours)}")
    lines.append(f"- `train_codes.jsonl` ids: {len(train_ids)}")
    lines.append(f"- Stage A3 ids missing from `our_rows`: {only_train}")
    lines.append(f"- `our_rows` ids missing from Stage A3: {only_ours}")
    lines.append(f"- hash field on `train_codes.jsonl`: {train_hash_field or '(none)'}")
    lines.append("")
    lines.append("A Stage A3 row was in the Track 3 sentence check only when its clip id was in `our_rows.jsonl`. This note does not change the Part A stop rule.")
    lines.append("")
    if problems:
        lines.append("Part A was not frozen:")
        lines.append("")
        for item in problems:
            lines.append(f"- {item}")
        lines.append("")
        return {"ok": False, "problems": problems, "counts": dict(counts), "train_hash_field": train_hash_field}

    TESTSET.mkdir(parents=True, exist_ok=True)
    dest = TESTSET / "partA.jsonl"
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "id": row["id"], "dialect": row["dialect"], "text": row["text"],
                "ref_audio": row["ref_audio"], "source": row["source"],
            }, ensure_ascii=False) + "\n")
    digest = write_sha(dest)
    lines.append(f"Frozen `{dest}` SHA-1 `{digest}`.")
    lines.append("")
    return {
        "ok": True,
        "sha1": digest,
        "n": len(rows),
        "counts": dict(counts),
        "seconds": {k: round(v, 3) for k, v in seconds.items()},
        "train_hash_field": train_hash_field,
    }


def _correct(dialect: str, sentence_id: str, text: str) -> tuple[str, str]:
    original = text
    if dialect == "emirati":
        text = text.replace("ياهزة", "جاهزة").replace("يديدة", "جديدة")
    if dialect == "egyptian" and sentence_id == "core_15":
        text = text.replace("نهارك سعيد", "يومك جميل")
    return original, text


def load_sentences(zip_path: Path) -> dict:
    found = {}
    with zipfile.ZipFile(zip_path) as archive:
        body = archive.read("manifest.jsonl").decode("utf-8")
    for line in body.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dialect = MANIFEST_DIALECT.get(row.get("dialect") or "")
        if not dialect:
            continue
        kind = row.get("set") or ""
        if dialect == "msa":
            if kind != "ab_plain":
                continue
        elif kind not in {"core", "shib"}:
            continue
        key = (dialect, row.get("sentence_id") or "")
        text = row.get("text") or ""
        if key in found and found[key] != text:
            raise RuntimeError(f"conflicting text for {key}")
        found[key] = text
    return found


NEEDED_FIELDS = ("n_speakers", "overlap_ratio", "firered_music", "music_filter", "text", "duration")
HIT_LABELS = (
    "a3_id", "a3_sha", "a3_video",
    "train_id", "train_sha", "train_video",
    "dev_id", "dev_sha", "dev_video",
    "plan_id", "plan_sha", "plan_video",
    "heard_id", "heard_video",
)
# Whole words after normalize_compare(equate=False), which strips diacritics and shadda.
MARKERS = {
    "emirati": (
        "مب", "هيه", "ياي", "يايه", "يالس", "يالسة", "ريّال", "شحالك", "شحالج", "شحالكم",
        "تبا", "تبين", "عيل", "باكر", "عقب", "إنزين", "السموحة", "وايد", "الحين", "اليديد", "يديد",
    ),
    "kuwaiti": ("شلونك", "شلون", "چ", "شنو", "ماكو", "أكو", "باچر", "چذي", "شلونچ"),
    "hijazi": ("ايش", "دحين", "إيوه", "فين", "كدا", "تبغى", "تبغي", "هادا", "مرة"),
    "egyptian": ("دلوقتي", "دلوقت", "كده", "ازاي"),
    "levantine": ("هلق", "هلأ", "هيك", "بدي", "بدك", "كتير", "منيح", "شو"),
}
NAJDI_CLIP = "rL1ESsDonEg_0048"
KUWAITI_CLIP = "TORLLuHa4fI_0050"
SINGLE_CLIP = {"najdi": NAJDI_CLIP, "kuwaiti": KUWAITI_CLIP}
_ARABIC_LANG = {
    "", "ar", "ara", "arabic", "msa",
    "ae", "bh", "dz", "eg", "iq", "jo", "kw", "lb", "ly", "ma", "mr",
    "om", "ps", "qa", "sa", "sd", "sy", "tn", "ye",
}
_DIAC = str.maketrans("", "", "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0640")


def _says_arabic(value) -> bool:
    text = str(value or "").strip().lower()
    if text in {"none", "null"}:
        text = ""
    if text in _ARABIC_LANG or text.startswith("ar-") or text.startswith("ar_"):
        return True
    return False


def _arabic_letter_ratio(text: str) -> float:
    arabic = 0
    other = 0
    for ch in text:
        if ch.isdigit() or "\u0660" <= ch <= "\u0669" or "\u06f0" <= ch <= "\u06f9":
            continue
        if "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" or "\u08a0" <= ch <= "\u08ff":
            if ch in "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0640":
                continue
            arabic += 1
        elif ch.isalpha():
            other += 1
    total = arabic + other
    if total == 0:
        return 0.0
    return arabic / total


def _norm_tokens(text: str) -> set[str]:
    return set(normalize_compare(text, equate=False).split())


def _marker_hit(dialect: str, text: str) -> str:
    words = MARKERS.get(dialect)
    if not words:
        return ""
    tokens = _norm_tokens(text)
    for word in sorted(words, key=len, reverse=True):
        token = normalize_compare(word, equate=False)
        if token and token in tokens:
            return word
    return ""


def _marker_distance(dialect: str, text: str) -> int:
    words = MARKERS.get(dialect) or ()
    tokens = _norm_tokens(text)
    markers = [normalize_compare(word, equate=False) for word in words]
    markers = [item for item in markers if item]
    if not tokens or not markers:
        return 999
    best = 999
    for token in tokens:
        for marker in markers:
            if abs(len(token) - len(marker)) >= best:
                continue
            best = min(best, _edit(token, marker))
            if best == 0:
                return 0
    return best


def _keep_close(found: list, distance: int, clip_id: str, text: str, rule: str) -> None:
    found.append((distance, clip_id, text[:220], rule))
    found.sort()
    del found[10:]


def _retire_v1() -> None:
    """Move the previous Part B freeze aside so nothing can load it."""
    for name in ("partB.jsonl", "partB.sha1"):
        src = TESTSET / name
        dest = TESTSET / name.replace("partB", "partB_v1_rejected")
        if src.is_file():
            if dest.exists():
                dest.unlink()
            src.rename(dest)
    prompts = TESTSET / "prompts"
    rejected = TESTSET / "prompts_v1_rejected"
    if prompts.is_dir():
        if rejected.exists():
            shutil.rmtree(rejected)
        prompts.rename(rejected)


def _empty_block() -> dict:
    return {"ids": set(), "videos": set(), "shas": set(), "hash_field": ""}


def _add_block(block: dict, row: dict) -> None:
    cid = row.get("clip_id") or ""
    block["ids"].update(_keys_of(cid))
    vid = video_id_of(cid)
    if vid:
        block["videos"].add(vid)
    if row.get("video_id"):
        block["videos"].add(str(row["video_id"]))
    if not block["hash_field"]:
        if row.get("sha1"):
            block["hash_field"] = "sha1"
        elif row.get("audio_sha1"):
            block["hash_field"] = "audio_sha1"
    field = block["hash_field"]
    if field and row.get(field):
        block["shas"].add(str(row[field]))


def _manifest_reason(row: dict) -> str:
    try:
        speakers = int(row.get("n_speakers") or 0)
    except (TypeError, ValueError):
        return "fields"
    if speakers != 1:
        return "speakers"
    try:
        overlap = float(row.get("overlap_ratio") or 0)
        music = float(row.get("firered_music") or 0)
        dur = float(row.get("duration"))
    except (TypeError, ValueError):
        return "fields"
    if overlap != 0:
        return "overlap"
    if music != 0 or (row.get("music_filter") or "") != "pass":
        return "music"
    if not str(row.get("text") or "").strip():
        return "text"
    if not (4.0 <= dur <= 10.0):
        return "duration"
    return ""


def _blocked_reason(blocks: dict, clip_id: str, video_id: str) -> str:
    keys = _keys_of(clip_id)
    derived = video_id_of(clip_id)
    for name in ("heard", "a3", "train", "dev", "plan"):
        block = blocks[name]
        if keys & block["ids"]:
            return f"{name}_id"
        if video_id and video_id in block["videos"]:
            return f"{name}_video"
        if derived and derived != video_id and derived in block["videos"]:
            return f"{name}_video"
    return ""


def _part_a_head() -> str:
    if not REPORT.is_file():
        raise RuntimeError("freeze_report.md is missing, and Part A is not being rebuilt")
    text = REPORT.read_text(encoding="utf-8")
    if "## Part A" not in text:
        raise RuntimeError("freeze report has no Part A section")
    marker = "\n## Part B\n"
    if marker in text:
        text = text.split(marker, 1)[0]
    return text.rstrip() + "\n"


def _write_report(part_b: list[str]) -> None:
    REPORT.write_text(_part_a_head() + "\n" + "\n".join(part_b).rstrip() + "\n", encoding="utf-8")


def assert_part_a() -> str:
    path = TESTSET / "partA.jsonl"
    sha_path = TESTSET / "partA.sha1"
    if not path.is_file() or not sha_path.is_file():
        raise RuntimeError("Part A is not frozen")
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    recorded = sha_path.read_text(encoding="utf-8").split()[0].strip()
    if digest != PART_A_SHA or recorded != PART_A_SHA:
        raise RuntimeError(f"Part A SHA changed: file {digest} recorded {recorded}")
    return digest


def _sentence_rows() -> tuple[dict, list[str]]:
    sentences = load_sentences(PLAN15_ZIP)
    by_dialect = defaultdict(list)
    for (dialect, sentence_id), text in sentences.items():
        original, corrected = _correct(dialect, sentence_id, text)
        by_dialect[dialect].append([sentence_id, original, corrected])
    for dialect in B_ORDER:
        by_dialect[dialect].sort(key=lambda item: item[0])
    problems = []
    for dialect in B_ORDER:
        want = 15 if dialect == "msa" else 20
        got = len(by_dialect.get(dialect, ()))
        if got != want:
            problems.append(f"{dialect} sentences {got} expected {want}")
    lookup = {(d, s): (o, c) for d, rows in by_dialect.items() for s, o, c in rows}
    checks = {
        ("emirati", "core_09"): ("ياهزة", "جاهزة"),
        ("emirati", "shib_03"): ("ياهزة", "جاهزة"),
        ("emirati", "shib_04"): ("يديدة", "جديدة"),
        ("egyptian", "core_15"): ("نهارك سعيد", "يومك جميل"),
    }
    for key, (old, new) in checks.items():
        original, corrected = lookup.get(key, ("", ""))
        if old not in original or new not in corrected:
            problems.append(f"correction missing for {key[0]} {key[1]}")
    if "ياي" not in lookup.get(("emirati", "shib_02"), ("",))[1]:
        problems.append("emirati shib_02 lost ياي")
    if "ريّال" not in lookup.get(("emirati", "shib_02"), ("",))[1]:
        problems.append("emirati shib_02 lost ريّال")
    if "اليديد" not in lookup.get(("emirati", "shib_01"), ("",))[1]:
        problems.append("emirati shib_01 changed اليديد")
    if "نهارك سعيد" not in lookup.get(("levantine", "core_15"), ("", ""))[1]:
        problems.append("levantine core_15 was changed")
    return by_dialect, problems


def _plan15_sources() -> dict:
    counts = Counter()
    with zipfile.ZipFile(PLAN15_ZIP) as archive:
        body = archive.read("manifest.jsonl").decode("utf-8")
    for line in body.splitlines():
        if line.strip():
            counts[json.loads(line).get("voice_source") or "(none)"] += 1
    return dict(counts)


def _build_blocks() -> dict:
    blocks = {name: _empty_block() for name in ("a3", "train", "dev", "plan", "heard")}
    for clip_id in HEARD_IDS:
        blocks["heard"]["ids"].update(_keys_of(clip_id))
        blocks["heard"]["videos"].add(video_id_of(clip_id))
    seen = 0
    for row in iter_jsonl(TRAIN_CODES):
        seen += 1
        _add_block(blocks["a3"], row)
        if seen % 200000 == 0:
            print(f"train_codes {seen}", flush=True)
    print(f"train_codes ids {len(blocks['a3']['ids'])} videos {len(blocks['a3']['videos'])} sha {len(blocks['a3']['shas'])}", flush=True)
    seen = 0
    for row in iter_jsonl(TIER1):
        seen += 1
        name = "dev" if row.get("split") == "dev" else "train"
        _add_block(blocks[name], row)
        if seen % 200000 == 0:
            print(f"tier1 {seen}", flush=True)
    print(f"tier1 train {len(blocks['train']['ids'])} dev {len(blocks['dev']['ids'])}", flush=True)
    for path in PLAN14:
        if path.is_file():
            for row in iter_jsonl(path):
                _add_block(blocks["plan"], row)
    print(f"plan14 ids {len(blocks['plan']['ids'])} sha {len(blocks['plan']['shas'])}", flush=True)
    return blocks


def _queue_dialect(dialect: str, blocks: dict) -> tuple[list[dict], Counter, list[str], list]:
    folder = FOLDER[dialect]
    manifest = YT_RAW / folder / "manifest_single.transcribed.jsonl"
    stats = Counter()
    close = []
    if not manifest.is_file():
        stats["no_manifest"] += 1
        return [], stats, [], close
    queued = []
    seen_ids = set()
    keys = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not keys:
                keys = sorted(row)
                missing = [field for field in NEEDED_FIELDS if field not in row]
                if missing:
                    stats["missing_fields"] = len(missing)
                    stats["missing_field_names"] = ",".join(missing)
                    return [], stats, keys, close
            reason = _manifest_reason(row)
            if reason:
                stats[reason] += 1
                continue
            stem = Path(row.get("audio_filepath") or "").stem
            if not stem or stem in seen_ids:
                stats["fields"] += 1
                continue
            video_id = str(row.get("video_id") or "") or video_id_of(stem)
            why = _blocked_reason(blocks, stem, video_id)
            if why:
                stats[why] += 1
                continue
            text = str(row.get("text") or "").strip()
            if not all(_says_arabic(row.get(field)) for field in ("lang", "audio_language", "jais_language")):
                stats["language"] += 1
                continue
            if _arabic_letter_ratio(text) < 0.80:
                stats["script"] += 1
                continue
            marker = _marker_hit(dialect, text)
            if dialect in MARKERS and not marker:
                stats["marker"] += 1
                _keep_close(close, _marker_distance(dialect, text), stem, text, "marker")
                continue
            seen_ids.add(stem)
            queued.append({
                "dialect": dialect,
                "clip_id": stem,
                "video_id": video_id,
                "path": str(YT / folder / f"{stem}.flac"),
                "text": text,
                "marker": marker,
                "manifest_dur": float(row["duration"]),
            })
    if dialect in SINGLE_CLIP:
        stats[f"{dialect}_pool"] = len(queued)
        queued = [item for item in queued if item["clip_id"] == SINGLE_CLIP[dialect]]
        if not queued:
            stats[f"{dialect}_missing"] += 1
    queued.sort(key=lambda item: item["clip_id"])
    stats["queued"] = len(queued)
    return queued, stats, keys, close


def _block_lines(blocks: dict) -> list[str]:
    lines = [
        "Held-out check. A prompt is rejected when its clip id, audio SHA-1, or YouTube video id is in any of:",
        "",
    ]
    labels = {
        "a3": f"`{_DATA}/tts_stageA/stageA3/train_codes.jsonl` (Stage A3)",
        "train": f"`{_DATA}/plan13/tier1_v1.jsonl` rows with `split=train` (Stage B1 training file, train split)",
        "dev": f"`{_DATA}/plan13/tier1_v1.jsonl` rows with `split=dev`",
        "plan": f"`{_DATA}/exp_11labs/manifest/train.jsonl` and `val.jsonl`",
        "heard": "the 21 quiz clips plus retired `OGZQSN4xEQg_0103` and `-9QCqIA0eyg_0494`",
    }
    for name in ("a3", "train", "dev", "plan", "heard"):
        block = blocks[name]
        lines.append(
            f"- {labels[name]}: ids {len(block['ids'])}, videos {len(block['videos'])}, "
            f"hashes {len(block['shas'])}, hash field `{block['hash_field'] or '(none)'}`"
        )
    lines.append("")
    return lines


def _prompt_listen(copied: dict) -> Path:
    import csv
    import io

    dest = TESTSET / "prompt_listen.zip"
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["dialect", "id", "clip id", "video id", "transcript", "matched marker", "CER", "duration", "f0"])
    with zipfile.ZipFile(dest, "w") as archive:
        for dialect in B_ORDER:
            for index in (0, 1):
                path, suffix, row = copied[(dialect, index)]
                rates = [float(row["cohere_cer"])]
                if row.get("jais_cer") is not None:
                    rates.append(float(row["jais_cer"]))
                writer.writerow([
                    dialect,
                    f"{dialect}_{suffix}",
                    row["clip_id"],
                    row["video_id"],
                    row["text"],
                    row.get("marker") or "",
                    f"{max(rates):.4f}",
                    f"{float(row['dur']):.2f}",
                    "" if row.get("f0") is None else row["f0"],
                ])
                archive.write(path, arcname=path.name)
        archive.writestr("prompts.csv", table.getvalue())
    return dest


def _closest_lines(lines: list[str], closes: dict, dialects: list[str]) -> None:
    lines.append("Closest rejects. Each line is clip id, transcript, and the rule it failed. The held-out block was not relaxed.")
    lines.append("")
    for dialect in dialects:
        lines.append(f"### {dialect}")
        lines.append("")
        items = list(closes.get(dialect) or [])
        if not items:
            lines.append("- no near miss was recorded")
            lines.append("")
            continue
        for item in items[:10]:
            if isinstance(item, dict):
                distance, clip_id, text, rule = item.get("distance"), item.get("clip_id"), item.get("transcript"), item.get("rule")
            else:
                distance, clip_id, text, rule = item
            extra = f", distance {distance}" if rule == "marker" else ""
            lines.append(f"- `{clip_id}` — {text} — failed `{rule}`{extra}")
        lines.append("")


def _fail(lines: list[str], problems: list[str]) -> dict:
    lines.append("Part B was not frozen:")
    lines.append("")
    for item in problems:
        lines.append(f"- {item}")
    lines.append("")
    _write_report(lines)
    return {"ok": False, "problems": problems, "report": str(REPORT)}


def prepare_part_b() -> dict:
    assert_part_a()
    _retire_v1()
    if not PLAN15_ZIP.is_file():
        raise RuntimeError(f"missing {PLAN15_ZIP}")
    by_dialect, problems = _sentence_rows()
    lines = [
        "## Part B",
        "",
        f"Prompts come from `{_DATA}/yt/yt_clean_24k`, one row per clip in `{_DATA}/yt/raw/<folder>/manifest_single.transcribed.jsonl`. Plan 13 dev rows are not the pool.",
        "",
        "Text is the Plan 15 manifest. Spelling A/B (`ab`) and mismatch (`mm`) rows are dropped. MSA uses the 15 `ab_plain` lines because that dialect has no core or shibboleth set.",
        "",
        "Corrections: Emirati `ياهزة` → `جاهزة`, `يديدة` → `جديدة`. `ياي`, `ريّال`, and `اليديد` stay. Egyptian core_15 `نهارك سعيد` → `يومك جميل`. Levantine core_15 is unchanged.",
        "",
        "A clip must be 4–10 s, one speaker, overlap ratio 0, `firered_music` 0, and `music_filter` pass, with a non-empty manifest transcript.",
        "",
        "Language, before CER: `lang`, `audio_language`, and `jais_language` must be Arabic (or empty), and at least 80% of the letters must be Arabic script.",
        "",
        "Dialect markers are whole words after `normalize_compare(equate=False)`, so diacritics and shadda are ignored. Emirati uses the replacement list (`مب`, `هيه`, `ياي`, `يايه`, `يالس`, `يالسة`, `ريّال`, `شحالك`, `شحالج`, `شحالكم`, `تبا`, `تبين`, `عيل`, `باكر`, `عقب`, `إنزين`, `السموحة`, `وايد`, `الحين`, `اليديد`, `يديد`). Kuwaiti adds `شنو`, `ماكو`, `أكو`, `باچر`, `چذي`, `شلونچ` to `شلونك`/`شلون`/`چ`. Hijazi adds `إيوه`, `فين`, `كدا`, `تبغى`, `تبغي`, `هادا`, `مرة` to `ايش`/`دحين`. Levantine adds `بدي`, `بدك`, `كتير`, `منيح`, `شو` to `هلق`/`هلأ`/`هيك`. Egyptian stays `دلوقتي`/`دلوقت`/`كده`/`ازاي`. MSA has no marker list.",
        "",
        f"Najdi uses `{NAJDI_CLIP}` for both slots. That clip still has to pass the video-level block. Both rows are `najdi_single_speaker=1`.",
        "",
        f"Kuwaiti uses `{KUWAITI_CLIP}` for both slots. The CER gate for Kuwaiti only is 0.11. Both rows are `kuwaiti_single_speaker=1`. The video-level block is not relaxed.",
        "",
    ]
    sources = _plan15_sources()
    lines.append("Plan 15 Round 3 voices are ElevenLabs ids, not YouTube clips: " + ", ".join(f"{key} {value}" for key, value in sorted(sources.items())) + ". No YouTube prompt is taken from that list.")
    lines.append("")
    if problems:
        return _fail(lines, problems)
    print("loading held-out block lists", flush=True)
    blocks = _build_blocks()
    lines.extend(_block_lines(blocks))
    queue = []
    rejects = {}
    keys_note = {}
    closes = {}
    short = []
    for dialect in B_ORDER:
        rows, stats, keys, close = _queue_dialect(dialect, blocks)
        rejects[dialect] = {key: value for key, value in stats.items() if key != "missing_field_names"}
        keys_note[dialect] = keys[:40]
        closes[dialect] = close
        print(f"queue {dialect} {len(rows)} {dict(stats)}", flush=True)
        need = 1 if dialect in SINGLE_CLIP else 2
        if stats.get("missing_fields"):
            problems.append(f"{dialect} manifest is missing {stats.get('missing_field_names')}; keys {keys}")
        elif stats.get("no_manifest"):
            problems.append(f"{dialect} manifest missing")
        elif len(rows) < need:
            extra = f" clip {SINGLE_CLIP[dialect]} is not in the held-out Arabic pool" if dialect in SINGLE_CLIP else ""
            problems.append(f"{dialect} queued {len(rows)} ({dict(stats)}){extra}")
            short.append(dialect)
        queue.extend(rows)
        lines.append(f"- {FOLDER[dialect]} for {dialect}: queued {len(rows)}; rejects {dict(stats)}")
    lines.append("")
    if problems:
        _closest_lines(lines, closes, short)
        return _fail(lines, problems)
    ROOT.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("w", encoding="utf-8") as handle:
        for row in queue:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    shas = set()
    for block in blocks.values():
        shas.update(block["shas"])
    with SHA_BLOCK.open("w", encoding="utf-8") as handle:
        for digest in shas:
            handle.write(digest + "\n")
    META.write_text(json.dumps({
        "sentences": {dialect: by_dialect[dialect] for dialect in B_ORDER},
        "rejects": rejects,
        "queue": {dialect: sum(1 for row in queue if row["dialect"] == dialect) for dialect in B_ORDER},
        "blocks": {
            name: {
                "ids": len(block["ids"]),
                "videos": len(block["videos"]),
                "shas": len(block["shas"]),
                "hash_field": block["hash_field"],
            }
            for name, block in blocks.items()
        },
        "plan15_sources": sources,
        "manifest_keys": keys_note,
    }, ensure_ascii=False), encoding="utf-8")
    lines.append(f"Queued {len(queue)} clips for the CER pass. `partB.sha1` is not written yet.")
    lines.append("")
    _write_report(lines)
    print(json.dumps({"prepare_ok": True, "queued": len(queue)}), flush=True)
    return {"ok": True, "queued": len(queue), "report": str(REPORT)}


def _recheck(prompts: list[dict]) -> dict:
    want_id = defaultdict(list)
    want_video = defaultdict(list)
    want_sha = defaultdict(list)
    for prompt in prompts:
        for key in _keys_of(prompt["clip_id"]):
            want_id[key].append(prompt["clip_id"])
        if prompt.get("video_id"):
            want_video[prompt["video_id"]].append(prompt["clip_id"])
            derived = video_id_of(prompt["clip_id"])
            if derived and derived != prompt["video_id"]:
                want_video[derived].append(prompt["clip_id"])
        if prompt.get("sha1"):
            want_sha[prompt["sha1"]].append(prompt["clip_id"])
    hits = {prompt["clip_id"]: Counter() for prompt in prompts}

    def mark(label: str, keys: list[str], table: dict) -> None:
        seen = set()
        for key in keys:
            for clip_id in table.get(key, ()):
                if (clip_id, label) in seen:
                    continue
                seen.add((clip_id, label))
                hits[clip_id][label] += 1

    def scan(path: Path, prefix_of) -> None:
        if not path.is_file():
            return
        seen = 0
        for row in iter_jsonl(path):
            seen += 1
            prefix = prefix_of(row)
            if not prefix:
                continue
            cid = row.get("clip_id") or ""
            mark(f"{prefix}_id", list(_keys_of(cid)), want_id)
            vids = [video_id_of(cid)]
            if row.get("video_id"):
                vids.append(str(row["video_id"]))
            mark(f"{prefix}_video", vids, want_video)
            for field in ("sha1", "audio_sha1"):
                if row.get(field):
                    mark(f"{prefix}_sha", [str(row[field])], want_sha)
            if seen % 200000 == 0:
                print(f"recheck {path.name} {seen}", flush=True)
        print(f"recheck {path.name} done {seen}", flush=True)

    scan(TRAIN_CODES, lambda _row: "a3")
    scan(TIER1, lambda row: "dev" if row.get("split") == "dev" else "train")
    for path in PLAN14:
        scan(path, lambda _row: "plan")
    heard_ids = set()
    heard_videos = set()
    for clip_id in HEARD_IDS:
        heard_ids.update(_keys_of(clip_id))
        heard_videos.add(video_id_of(clip_id))
    for prompt in prompts:
        if _keys_of(prompt["clip_id"]) & heard_ids:
            hits[prompt["clip_id"]]["heard_id"] += 1
        if prompt.get("video_id") in heard_videos or video_id_of(prompt["clip_id"]) in heard_videos:
            hits[prompt["clip_id"]]["heard_video"] += 1
    return hits


def finish_part_b(cer: dict) -> dict:
    assert_part_a()
    meta = json.loads(META.read_text(encoding="utf-8"))
    lines = [
        "## Part B",
        "",
        f"Prompts come from `{_DATA}/yt/yt_clean_24k`. Each clip is one speaker, 4–10 s, no music, no overlap, and held out by clip id, audio SHA-1, and YouTube video id.",
        "",
        "Text is the Plan 15 manifest. Spelling A/B (`ab`) and mismatch (`mm`) rows are dropped. MSA uses the 15 `ab_plain` lines.",
        "",
        "Corrections: Emirati `ياهزة` → `جاهزة`, `يديدة` → `جديدة`. `ياي`, `ريّال`, and `اليديد` stay. Egyptian core_15 `نهارك سعيد` → `يومك جميل`. Levantine core_15 is unchanged.",
        "",
        "Plan 15 Round 3 voices are ElevenLabs ids, not YouTube clips: "
        + ", ".join(f"{key} {value}" for key, value in sorted((meta.get("plan15_sources") or {}).items()))
        + ".",
        "",
        "CER is `plan12_text.cer` against the manifest transcript. Cohere runs for every prompt. Jais also runs for Emirati, Kuwaiti, Najdi, Hijazi, and Levantine. A prompt is kept only when every computed CER is at most 0.10, except Kuwaiti, where the gate is 0.11.",
        "",
        "Before CER, `lang`, `audio_language`, and `jais_language` must be Arabic or empty, and at least 80% of the letters must be Arabic script. Emirati, Kuwaiti, Hijazi, Egyptian, and Levantine also need one whole-word marker after `normalize_compare(equate=False)`. MSA does not. Prompt ids are `p1` and `p2`. Measured f0 is a column, not part of the id.",
        "",
        f"Najdi is `{NAJDI_CLIP}` in both slots (`najdi_single_speaker=1`). Kuwaiti is `{KUWAITI_CLIP}` in both slots (`kuwaiti_single_speaker=1`). The video-level block was not relaxed.",
        "",
    ]
    blocks = meta.get("blocks") or {}
    for name, label in (
        ("a3", "Stage A3 `train_codes.jsonl`"),
        ("train", "Plan 13 `split=train` / Stage B1 train file"),
        ("dev", "Plan 13 `split=dev`"),
        ("plan", "Plan 14 manifests"),
        ("heard", "quiz clips and two retired clips"),
    ):
        block = blocks.get(name) or {}
        lines.append(
            f"- {label}: ids {block.get('ids', 0)}, videos {block.get('videos', 0)}, "
            f"hashes {block.get('shas', 0)}, hash field `{block.get('hash_field') or '(none)'}`"
        )
    lines.append("")
    stats = cer.get("stats") or {}
    for dialect in B_ORDER:
        lines.append(f"- {dialect}: queue {(meta.get('queue') or {}).get(dialect, 0)}; CER pass {stats.get(dialect, {})}")
    lines.append("")
    picks = cer.get("picks") or []
    by_dialect = defaultdict(list)
    for row in picks:
        by_dialect[row["dialect"]].append(row)
    problems = list(cer.get("problems") or [])
    for dialect in B_ORDER:
        need = 1 if dialect in SINGLE_CLIP else 2
        if len(by_dialect.get(dialect, ())) != need:
            problems.append(f"{dialect} prompts {len(by_dialect.get(dialect, ()))}")
    if problems or not cer.get("ok"):
        short = [
            dialect for dialect in B_ORDER
            if len(by_dialect.get(dialect, ())) != (1 if dialect in SINGLE_CLIP else 2)
        ]
        _closest_lines(lines, cer.get("closest") or {}, short)
        return _fail(lines, problems or ["CER pass did not keep two prompts per dialect"])
    print("rechecking 14 prompts against training lists", flush=True)
    hits = _recheck(picks)
    bad = []
    for prompt in picks:
        counts = hits.get(prompt["clip_id"]) or Counter()
        if any(counts.get(label, 0) for label in HIT_LABELS):
            bad.append(f"{prompt['dialect']} {prompt['clip_id']} hits {dict(counts)}")
    if bad:
        return _fail(lines, bad)
    sentences = meta["sentences"]
    PROMPTS.mkdir(parents=True, exist_ok=True)
    for stale in PROMPTS.iterdir():
        if stale.is_file():
            stale.unlink()
    copied = {}
    for dialect in B_ORDER:
        chosen = sorted(by_dialect[dialect], key=lambda row: row["clip_id"])
        if dialect in SINGLE_CLIP:
            if len(chosen) != 1 or chosen[0]["clip_id"] != SINGLE_CLIP[dialect]:
                return _fail(lines, [f"{dialect} prompts {[row['clip_id'] for row in chosen]}"])
            chosen = [dict(chosen[0]), dict(chosen[0])]
        enriched = []
        for row in chosen:
            info = audio_stats(Path(row["path"]))
            row = dict(row)
            row["gender"] = info["gender"]
            row["f0"] = info["f0"]
            row["dur"] = info["dur"]
            enriched.append(row)
        for index, row in enumerate(enriched):
            suffix = f"p{index + 1}"
            dest = PROMPTS / f"{dialect}_{suffix}{Path(row['path']).suffix}"
            shutil.copyfile(row["path"], dest)
            copied[(dialect, index)] = (dest, suffix, row)
    out_rows = []
    for dialect in B_ORDER:
        for sentence_id, original, corrected in sentences[dialect]:
            for index in (0, 1):
                dest, suffix, row = copied[(dialect, index)]
                out_rows.append({
                    "id": f"{dialect}/{sentence_id}/{suffix}",
                    "dialect": dialect,
                    "text": corrected,
                    "text_original": original,
                    "ref_audio": str(dest),
                    "prompt_text": row["text"],
                    "prompt_id": suffix,
                    "prompt_clip": row["clip_id"],
                    "video_id": row["video_id"],
                    "marker": row.get("marker") or "",
                    "f0": row.get("f0"),
                    "duration": row["dur"],
                    "cohere_cer": row["cohere_cer"],
                    "jais_cer": row.get("jais_cer"),
                    "voice_gender": "m" if row["gender"] == "male" else "f",
                    "source": f"plan15:{sentence_id} prompt:{row['clip_id']}",
                    "najdi_single_speaker": 1 if dialect == "najdi" else 0,
                    "kuwaiti_single_speaker": 1 if dialect == "kuwaiti" else 0,
                })
    if len(out_rows) != 270:
        return _fail(lines, [f"rows {len(out_rows)}"])
    dest = TESTSET / "partB.jsonl"
    with dest.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = write_sha(dest)
    lines.append(f"Frozen `{dest}` SHA-1 `{digest}`. Rows {len(out_rows)}.")
    lines.append("")
    lines.append(f"CER GPU wall {float(cer.get('wall_s') or 0):.1f}s, ${float(cer.get('dollars') or 0):.4f}.")
    lines.append("")
    lines.append("Prompt clips:")
    lines.append("")
    for dialect in B_ORDER:
        for index in (0, 1):
            dest_path, suffix, row = copied[(dialect, index)]
            jais = row.get("jais_cer")
            jais_text = f", Jais CER {float(jais):.4f}" if jais is not None else ""
            marker = row.get("marker") or "(none)"
            f0 = row.get("f0")
            f0_text = "unknown" if f0 is None else f"{float(f0):.1f} Hz"
            single = f" `{dialect}_single_speaker=1`;" if dialect in SINGLE_CLIP else ""
            lines.append(
                f"- {dialect} {suffix}: clip `{row['clip_id']}` video `{row['video_id']}` "
                f"{float(row['dur']):.2f}s f0 {f0_text} Cohere CER {float(row['cohere_cer']):.4f}{jais_text}; "
                f"marker `{marker}`;{single} "
                + ", ".join(f"{label} 0 hits" for label in HIT_LABELS)
                + f" → `{dest_path}`"
            )
            lines.append(f"  transcript: {row['text']}")
    lines.append("")
    listen = _prompt_listen(copied)
    lines.append(f"Listening pack `{listen}` has the 14 prompt files and `prompts.csv`.")
    lines.append("")
    _write_report(lines)
    again = assert_part_a()
    print(json.dumps({"partB_ok": True, "sha1": digest, "partA": again}), flush=True)
    return {"ok": True, "sha1": digest, "n": len(out_rows), "partA": again, "report": str(REPORT)}


