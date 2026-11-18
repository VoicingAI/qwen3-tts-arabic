"""Track 7 scoring. Generation and Cohere run in separate images. No ElevenLabs."""
import json
import os
import shutil
import time
import zlib
from collections import defaultdict
from pathlib import Path
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

EVAL = Path(f"{_DATA}/plan13/eval")
SUBSET = Path(f"{_DATA}/plan13/testset_score.jsonl")
COHERE_WEIGHTS = Path(f"{_DATA}/retx/plan12/partD/weights")
COHERE_SHA = "c3e911b42149bf7a1e53d5cef9878aee87515a23"
JAIS_DIR = Path(f"{_DATA}/models/Jais-ASR")
JAIS_SHA = "1323995fa189c479369a3929be0e95039eba5cba"
WAVLM_ID = "microsoft/wavlm-base-plus-sv"
RATE = 1.95
PRIORITY = ("ar_egy", "ar_msa", "ar_sau", "ar_uae")
COHERE_LANGUAGE = "ar"
DEBUG_INDEXES = (0, 100, 102, 350, 450, 550, 800, 802, 1050, 1052)


def _rows():
    rows = []
    with SUBSET.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cap_seconds(text):
    chars = len((text or "").replace(" ", ""))
    return min(60.0, 2.5 * chars / 12.0 + 3.0)


def _as_bf16(inputs, model):
    import torch

    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            value = value.to(model.device)
            if getattr(value, "is_floating_point", lambda: False)():
                value = value.to(dtype=torch.bfloat16)
        moved[key] = value
    return moved


def _load_cohere():
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    local = Path("/tmp/cohere_weights")
    if not (local / "config.json").is_file():
        if not (COHERE_WEIGHTS / "config.json").is_file():
            raise RuntimeError(f"missing Cohere weights at {COHERE_WEIGHTS}")
        shutil.copytree(COHERE_WEIGHTS, local, dirs_exist_ok=True)
    processor = AutoProcessor.from_pretrained(local)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        local, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    return processor, model


def _load_16k(path):
    """Cohere's processor accepts 16 kHz only. Qwen writes 24 kHz, and some references are 44.1 kHz."""
    import librosa
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype="float32")
    sr = int(sr)
    if sr != 16000 and len(data):
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    return data, sr


def _transcribe(processor, model, path):
    import torch

    data, sr = _load_16k(path)
    inputs = _as_bf16(
        processor(data, sampling_rate=sr, return_tensors="pt", language=COHERE_LANGUAGE),
        model,
    )
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=False, num_beams=1, repetition_penalty=1.0,
        )
    decoded = processor.batch_decode(out, skip_special_tokens=True)
    return decoded[0] if isinstance(decoded, list) else str(decoded)


def _repeating(text):
    from plan12_text import normalize_compare

    words = normalize_compare(text or "").split()
    if len(words) < 8:
        return False
    grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    best = max((grams.count(gram) for gram in set(grams)), default=0)
    return best >= 3


def _mono16(path):
    import librosa
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype="float32")
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    if len(data) > 16000 * 30:
        data = data[: 16000 * 30]
    rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
    return data, rms


def generate_body(ckpt, tag, shard, shards=2):
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    started = time.time()
    rows = [row for row in _rows() if int(row["i"]) % shards == shard]
    out_dir = EVAL / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / f"gen_{shard}.jsonl"
    if part.exists():
        part.unlink()
    local = Path(f"/tmp/plan13_gen_{tag}_{shard}")
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    tts = Qwen3TTSModel.from_pretrained(
        ckpt, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    errors = 0
    for row in rows:
        text = row.get("text") or ""
        ref = row.get("ref_audio") or row.get("audio")
        dest = out_dir / f"{int(row['i']):05d}.wav"
        item = {
            "i": row["i"], "clip_id": row.get("clip_id"), "dialect": row.get("dialect"),
            "text": text, "source": row.get("source"), "wav": str(dest), "error": "",
        }
        try:
            local_ref = local / f"ref_{int(row['i']):05d}.wav"
            if not local_ref.is_file():
                shutil.copyfile(ref, local_ref)
            ref_dur = float(sf.info(local_ref).duration)
            chars = len(text.replace(" ", ""))
            cap_s = cap_seconds(text)
            torch.manual_seed(zlib.crc32(f"{row.get('clip_id')}|{tag}".encode()))
            wavs, sr = tts.generate_voice_clone(
                text=text, language=row["dialect"], ref_audio=str(local_ref),
                x_vector_only_mode=True, non_streaming_mode=True,
                max_new_tokens=max(1, int(cap_s * 12.5)),
            )
            y = wavs[0]
            local_out = local / dest.name
            sf.write(local_out, y, sr)
            shutil.copyfile(local_out, dest)
            dur = len(y) / sr
            item.update({
                "dur": round(dur, 3), "ref_dur": round(ref_dur, 3),
                "cps": round(chars / max(dur, 1e-3), 3),
                "empty": dur < 0.3, "near_cap": dur >= 0.95 * cap_s,
                "too_long": ref_dur > 0 and dur > 2 * ref_dur,
            })
        except Exception as exc:
            errors += 1
            item["error"] = str(exc)[:300]
        _append(part, item)
    wall = time.time() - started
    meta = {
        "tag": tag, "shard": shard, "n": len(rows), "errors": errors,
        "wall_s": round(wall, 1), "dollars": round(wall / 3600 * RATE, 4),
    }
    _write(out_dir / f"gen_{shard}.json", meta)
    print(json.dumps(meta), flush=True)
    return meta


def score_body(tag, shard, shards=2, do_refs=0):
    import torch

    from plan12_letter import letter_score
    from plan12_text import normalize_compare, wer

    started = time.time()
    subset = {int(row["i"]): row for row in _rows()}
    out_dir = EVAL / tag
    gen_path = out_dir / f"gen_{shard}.jsonl"
    rows = [json.loads(line) for line in gen_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    part = out_dir / f"score_{shard}.jsonl"
    if part.exists():
        part.unlink()
    processor, model = _load_cohere()
    scored = []
    for row in rows:
        item = dict(row)
        hyp = ""
        if row.get("wav") and Path(row["wav"]).is_file() and not row.get("error"):
            try:
                hyp = _transcribe(processor, model, row["wav"])
            except Exception as exc:
                item["error"] = (item.get("error") or "") + " cohere " + str(exc)[:200]
        item["hyp"] = hyp
        item["cohere_language"] = COHERE_LANGUAGE
        ref_text = row.get("text") or ""
        item["ref_norm"] = normalize_compare(ref_text)
        try:
            item["wer"] = wer(ref_text, hyp)
            item["ler"] = letter_score(ref_text, hyp)["letter_rate"]
        except Exception as exc:
            item["wer"] = 1.0
            item["ler"] = 1.0
            item["error"] = (item.get("error") or "") + " wer " + str(exc)[:160]
        item["repeating"] = _repeating(hyp) if hyp else False
        if do_refs:
            src = subset.get(int(row["i"])) or {}
            ref_audio = src.get("ref_audio") or src.get("audio")
            ref_hyp = ""
            if ref_audio and Path(ref_audio).is_file():
                try:
                    ref_hyp = _transcribe(processor, model, ref_audio)
                except Exception as exc:
                    item["ref_error"] = str(exc)[:200]
            item["ref_hyp"] = ref_hyp
            try:
                item["ref_wer"] = wer(ref_text, ref_hyp)
            except Exception:
                item["ref_wer"] = None
        scored.append(item)
    del model
    torch.cuda.empty_cache()

    wavlm_error = ""
    try:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        fe = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
        sv = WavLMForXVector.from_pretrained(WAVLM_ID).cuda().eval()
        for item in scored:
            item["sim"] = None
            item["silent"] = bool(item.get("empty"))
            if not item.get("wav") or not Path(item["wav"]).is_file():
                continue
            try:
                hyp_y, hyp_rms = _mono16(item["wav"])
                item["silent"] = bool(item.get("empty")) or hyp_rms < 0.005
                src = subset.get(int(item["i"])) or {}
                ref_audio = src.get("ref_audio") or src.get("audio")
                if not ref_audio:
                    continue
                ref_y, _ = _mono16(ref_audio)
                inputs = fe([ref_y, hyp_y], sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(sv.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    emb = sv(**inputs).embeddings
                item["sim_gen"] = str(item["wav"])
                item["sim_ref"] = str(ref_audio)
                item["sim"] = float(torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0))
            except Exception as exc:
                item["sim_error"] = str(exc)[:200]
        del sv
        torch.cuda.empty_cache()
    except Exception as exc:
        wavlm_error = str(exc)[:300]
        for item in scored:
            item["sim"] = None

    jais_error = ""
    uae = [item for item in scored if item.get("dialect") == "ar_uae" and item.get("wav")]
    if uae:
        try:
            from retx_jais import load_jais, transcribe_items

            jais_proc, jais_model = load_jais(JAIS_DIR)
            items = [{"clip_id": item["wav"], "duration": item.get("dur") or 0} for item in uae]
            outs = transcribe_items(jais_proc, jais_model, items, batch_size=4)
            for item, out in zip(uae, outs):
                hyp = (out or {}).get("transcript") or ""
                item["jais_hyp"] = hyp
                try:
                    item["jais_wer"] = wer(item.get("text") or "", hyp)
                except Exception:
                    item["jais_wer"] = None
            del jais_model
            torch.cuda.empty_cache()
        except Exception as exc:
            jais_error = str(exc)[:300]

    for item in scored:
        flags = {
            "empty": bool(item.get("empty")),
            "near_cap": bool(item.get("near_cap")),
            "too_long": bool(item.get("too_long")),
            "repeating": bool(item.get("repeating")),
            "silent": bool(item.get("silent")),
            "error": bool(item.get("error")),
        }
        item["fail_why"] = [name for name, fired in flags.items() if fired]
        item["fail"] = bool(item["fail_why"])
        _append(part, item)
    wall = time.time() - started
    meta = {
        "tag": tag, "shard": shard, "n": len(scored), "wall_s": round(wall, 1),
        "dollars": round(wall / 3600 * RATE, 4), "wavlm_error": wavlm_error,
        "jais_error": jais_error, "cohere": COHERE_SHA, "jais": JAIS_SHA, "wavlm": WAVLM_ID,
        "do_refs": int(do_refs),
    }
    _write(out_dir / f"score_{shard}.json", meta)
    print(json.dumps(meta), flush=True)
    return meta


def _mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def publish_body(tag, step, ckpt):
    from torch.utils.tensorboard import SummaryWriter

    out_dir = EVAL / tag
    rows = []
    for shard in (0, 1):
        path = out_dir / f"score_{shard}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    by = defaultdict(list)
    for row in rows:
        by[row.get("dialect") or "?"].append(row)
    summary = {
        "tag": tag, "step": int(step), "ckpt": ckpt, "n": len(rows),
        "gulf": "Gulf is not in the human-checked set.",
        "by_dialect": {},
    }
    dollars = 0.0
    wall = 0.0
    for shard in (0, 1):
        for kind in ("gen", "score"):
            meta_path = out_dir / f"{kind}_{shard}.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                dollars += float(meta.get("dollars") or 0)
                wall += float(meta.get("wall_s") or 0)
                if meta.get("wavlm_error"):
                    summary["wavlm_error"] = meta["wavlm_error"]
                if meta.get("jais_error"):
                    summary["jais_error"] = meta["jais_error"]
    summary["dollars"] = round(dollars, 4)
    summary["wall_s"] = round(wall, 1)
    priority = []
    for dialect, group in sorted(by.items()):
        cell = {
            "n": len(group),
            "wer": _mean([r.get("wer") for r in group]),
            "ler": _mean([r.get("ler") for r in group]),
            "sim": _mean([r.get("sim") for r in group]),
            "cps": _mean([r.get("cps") for r in group]),
            "fail": _mean([1.0 if r.get("fail") else 0.0 for r in group]),
            "jais_wer": _mean([r.get("jais_wer") for r in group]),
            "ref_wer": _mean([r.get("ref_wer") for r in group]),
        }
        summary["by_dialect"][dialect] = cell
        if dialect in PRIORITY and cell["wer"] is not None:
            priority.append(cell["wer"])
    summary["priority_wer"] = _mean(priority)
    _write(out_dir / "summary.json", summary)
    writer = SummaryWriter(log_dir=f"{_DATA}/plan13/tb_eval")
    step_i = int(step)
    for dialect, cell in summary["by_dialect"].items():
        if cell["wer"] is not None:
            writer.add_scalar(f"wer/{dialect}", cell["wer"], step_i)
        if cell["ler"] is not None:
            writer.add_scalar(f"ler/{dialect}", cell["ler"], step_i)
        if cell["sim"] is not None:
            writer.add_scalar(f"similarity/{dialect}", cell["sim"], step_i)
        if cell["cps"] is not None:
            writer.add_scalar(f"speaking_rate/{dialect}", cell["cps"], step_i)
        if cell["fail"] is not None:
            writer.add_scalar(f"fail_share/{dialect}", cell["fail"], step_i)
    if summary["priority_wer"] is not None:
        writer.add_scalar("wer/priority_mean", summary["priority_wer"], step_i)
    ref_path = EVAL / "a3-020408" / "summary.json"
    if ref_path.is_file():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        if ref.get("priority_wer") is not None:
            for mark in (0, 4000, 5648, 11296, 16944, 22595, 45190, 67785, step_i):
                writer.add_scalar("wer/step20408_priority", ref["priority_wer"], mark)
    writer.flush()
    writer.close()
    print(json.dumps({"tag": tag, "priority_wer": summary["priority_wer"], "dollars": summary["dollars"]}), flush=True)
    return summary


def _cosine(left, right):
    import torch

    return float(torch.nn.functional.cosine_similarity(left, right, dim=0))


def _embed_one(fe, sv, wave):
    import torch

    inputs = fe([wave], sampling_rate=16000, return_tensors="pt")
    inputs = {key: value.to(sv.device) for key, value in inputs.items()}
    with torch.inference_mode():
        return sv(**inputs).embeddings[0]


def _embed_batch(fe, sv, left, right):
    import torch

    inputs = fe([left, right], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(sv.device) for key, value in inputs.items()}
    with torch.inference_mode():
        emb = sv(**inputs).embeddings
    return _cosine(emb[0], emb[1])


def _corr(left, right):
    import numpy as np

    width = min(len(left), len(right))
    if width < 16:
        return None
    a = left[:width] - left[:width].mean()
    b = right[:width] - right[:width].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)


def noise_time(ckpt, n=32):
    """Time one-clip generation against a batch, on the first n shard-0 clips. No files kept."""
    import torch
    from qwen_tts import Qwen3TTSModel

    rows = [row for row in _rows() if int(row["i"]) % 2 == 0][:n]
    local = Path("/tmp/plan13_noise_time")
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    started = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        ckpt, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    load_s = time.time() - started
    copy_t = time.time()
    ready = []
    for row in rows:
        ref = row.get("ref_audio") or row.get("audio")
        dest = local / f"{int(row['i']):05d}.wav"
        shutil.copyfile(ref, dest)
        ready.append({
            "i": int(row["i"]), "text": row.get("text") or "", "dialect": row["dialect"],
            "ref": str(dest),
        })
    copy_s = time.time() - copy_t

    def run(batch_size):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        t0 = time.time()
        made = 0
        for start in range(0, len(ready), batch_size):
            group = ready[start:start + batch_size]
            caps = [cap_seconds(item["text"]) for item in group]
            tts.generate_voice_clone(
                text=[item["text"] for item in group],
                language=[item["dialect"] for item in group],
                ref_audio=[item["ref"] for item in group],
                x_vector_only_mode=[True] * len(group),
                non_streaming_mode=True,
                max_new_tokens=max(1, int(max(caps) * 12.5)),
            )
            made += len(group)
        elapsed = time.time() - t0
        return {"batch": batch_size, "n": made, "seconds": round(elapsed, 1),
                "sec_per_clip": round(elapsed / max(made, 1), 3)}

    out = {"load_s": round(load_s, 1), "copy_s": round(copy_s, 1), "runs": []}
    for size in (1, 8):
        try:
            out["runs"].append(run(size))
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            out["runs"].append({"batch": size, "error": "oom " + str(exc)[:160]})
    print(json.dumps(out), flush=True)
    return out


def debug_body():
    """Score 10 already-generated step-20408 clips and record both similarity methods."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    from plan12_letter import letter_score
    from plan12_text import normalize_compare, wer

    started = time.time()
    subset = {int(row["i"]): row for row in _rows()}
    gen_rows = {}
    for shard in (0, 1):
        path = EVAL / "a3-020408" / f"gen_{shard}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                gen_rows[int(row["i"])] = row
    processor, model = _load_cohere()
    clips = []
    waves = {}
    for index in DEBUG_INDEXES:
        row = gen_rows[index]
        src = subset[index]
        ref_audio = src.get("ref_audio") or src.get("audio")
        item = {
            "i": index,
            "dialect": row.get("dialect"),
            "clip_id": row.get("clip_id"),
            "wav": row.get("wav"),
            "dur": row.get("dur"),
            "ref_dur": row.get("ref_dur"),
            "ref_audio": ref_audio,
            "text": row.get("text") or "",
            "ref_norm": normalize_compare(row.get("text") or ""),
            "cohere_language": COHERE_LANGUAGE,
            "empty": row.get("empty"),
            "near_cap": row.get("near_cap"),
            "too_long": row.get("too_long"),
        }
        hyp = ""
        try:
            hyp = _transcribe(processor, model, row["wav"])
        except Exception as exc:
            item["error"] = "cohere " + str(exc)[:240]
        item["hyp"] = hyp
        try:
            item["wer"] = wer(item["text"], hyp)
            item["ler"] = letter_score(item["text"], hyp)["letter_rate"]
        except Exception as exc:
            item["wer"] = 1.0
            item["ler"] = 1.0
            item["error"] = (item.get("error") or "") + " wer " + str(exc)[:160]
        item["repeating"] = _repeating(hyp) if hyp else False
        gen_y, gen_rms = _mono16(row["wav"])
        ref_y, _ = _mono16(ref_audio)
        item["silent"] = bool(item.get("empty")) or gen_rms < 0.005
        item["corr"] = _corr(gen_y, ref_y)
        waves[index] = (gen_y, ref_y)
        clips.append(item)
    del model
    torch.cuda.empty_cache()

    fe = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
    sv = WavLMForXVector.from_pretrained(WAVLM_ID).cuda().eval()
    for pos, item in enumerate(clips):
        gen_y, ref_y = waves[item["i"]]
        item["sim_gen"] = item["wav"]
        item["sim_ref"] = item["ref_audio"]
        item["sim_batch"] = _embed_batch(fe, sv, ref_y, gen_y)
        item["sim"] = _cosine(_embed_one(fe, sv, ref_y), _embed_one(fe, sv, gen_y))
        other = clips[(pos + 1) % len(clips)]
        other_ref = waves[other["i"]][1]
        item["control_ref"] = other["ref_audio"]
        item["sim_other_ref"] = _cosine(_embed_one(fe, sv, other_ref), _embed_one(fe, sv, gen_y))
        flags = {
            "empty": bool(item.get("empty")),
            "near_cap": bool(item.get("near_cap")),
            "too_long": bool(item.get("too_long")),
            "repeating": bool(item.get("repeating")),
            "silent": bool(item.get("silent")),
            "error": bool(item.get("error")),
        }
        item["fail_why"] = [name for name, fired in flags.items() if fired]
        item["fail"] = bool(item["fail_why"])
        print(json.dumps({
            "i": item["i"], "dialect": item["dialect"], "wer": round(item["wer"], 3),
            "sim": round(item["sim"], 3), "sim_batch": round(item["sim_batch"], 3),
            "sim_other": round(item["sim_other_ref"], 3), "corr": item["corr"],
            "fail": item["fail_why"], "hyp": item["hyp"][:80],
        }, ensure_ascii=False), flush=True)
    del sv
    wall = time.time() - started
    report = {
        "tag": "a3-020408", "n": len(clips), "language": COHERE_LANGUAGE,
        "wall_s": round(wall, 1), "dollars": round(wall / 3600 * RATE, 4),
        "clips": clips,
    }
    _write(Path(f"{_DATA}/plan13/debug10/report.json"), report)
    print(json.dumps({"n": len(clips), "wall_s": report["wall_s"], "dollars": report["dollars"]}), flush=True)
    return {
        "n": len(clips), "wall_s": report["wall_s"], "dollars": report["dollars"],
        "wer": [round(item["wer"], 3) for item in clips],
        "sim": [round(item["sim"], 3) for item in clips],
        "sim_batch": [round(item["sim_batch"], 3) for item in clips],
        "sim_other": [round(item["sim_other_ref"], 3) for item in clips],
    }


NOISE = Path(f"{_DATA}/plan13/eval_noise")
NOISE_SEEDS = (1, 2, 3)


def _write_16k(path, wave, sr):
    import librosa
    import numpy as np
    import soundfile as sf

    data = np.asarray(wave, dtype="float32").reshape(-1)
    if int(sr) != 16000 and len(data):
        data = librosa.resample(data, orig_sr=int(sr), target_sr=16000)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, 16000, subtype="PCM_16")
    return len(data) / 16000.0


def noise_generate(ckpt, tag, shard, shards=2, batch_size=8, budget_s=5400, root=None, seeds=None, rows_in=None, decode=None):
    """Fixed seeds, one model load, batched. Wavs stay in a tar for scoring.

    The seed is set once per batch, not once per clip. Clip order is the shard
    sorted by the text cap, then by index, so every checkpoint gets the same batches.
    Sampler settings are whatever generation_config.json says, unless decode is set.
    decode replaces only the sampler keys it contains. The default leaves Plan 13 unchanged.
    root, seeds, and rows_in keep the Plan 13 call unchanged and let Plan 14
    write under its own directory.
    """
    import hashlib
    import tarfile

    import torch
    from qwen_tts import Qwen3TTSModel

    started = time.time()
    decode = {key: value for key, value in dict(decode or {}).items() if value is not None}
    seed_list = tuple(seeds) if seeds is not None else NOISE_SEEDS
    source = list(rows_in) if rows_in is not None else _rows()
    rows = [row for row in source if int(row["i"]) % shards == shard]
    out_dir = (Path(root) if root else NOISE) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"gen_{shard}.json"
    tar_path = out_dir / f"audio_{shard}.tar"
    if meta_path.is_file() and tar_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("done") and int(old.get("n") or 0) >= len(rows) * len(seed_list) and (old.get("decode") or {}) == decode:
            print(json.dumps({"skip": True, "tag": tag, "shard": shard}), flush=True)
            return old

    local = Path(f"/tmp/plan13_noise_{tag}_{shard}")
    if local.exists():
        shutil.rmtree(local)
    ref_dir = local / "ref"
    wav_root = local / "wav"
    ref_dir.mkdir(parents=True)
    wav_root.mkdir(parents=True)

    ready = []
    copy_errors = 0
    for row in rows:
        ref = row.get("ref_audio") or row.get("audio")
        dest = ref_dir / f"{int(row['i']):05d}.wav"
        try:
            shutil.copyfile(ref, dest)
        except Exception as exc:
            copy_errors += 1
            print(json.dumps({"copy_error": int(row["i"]), "error": str(exc)[:160]}), flush=True)
            continue
        text = row.get("text") or ""
        ready.append({
            "i": int(row["i"]),
            "clip_id": row.get("clip_id"),
            "dialect": row.get("dialect"),
            "text": text,
            "cap": cap_seconds(text),
            "ref": str(dest),
            "mode": row.get("mode") or "xvec",
            "prompt_text": row.get("prompt_text") or "",
        })
    ready.sort(key=lambda item: (round(item["cap"], 5), item["i"]))

    cfg_path = Path(ckpt) / "generation_config.json"
    cfg_bytes = cfg_path.read_bytes()
    cfg = json.loads(cfg_bytes)
    load_t = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        ckpt, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    load_s = time.time() - load_t

    prompts = []
    prompt_t = time.time()
    prompt_at = 0
    prompt_batch = batch_size
    while prompt_at < len(ready):
        group = ready[prompt_at:prompt_at + prompt_batch]
        try:
            prompts.extend(tts.create_voice_clone_prompt(
                ref_audio=[item["ref"] for item in group],
                ref_text=[item["prompt_text"] or None for item in group],
                x_vector_only_mode=[item["mode"] != "icl" for item in group],
            ))
            prompt_at += len(group)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(group) == 1:
                raise
            prompt_batch = max(1, len(group) // 2)
            print(json.dumps({"prompt_oom": prompt_batch, "at": prompt_at}), flush=True)
    prompt_s = time.time() - prompt_t
    if len(prompts) != len(ready):
        raise RuntimeError(f"prompt count {len(prompts)} != clips {len(ready)}")
    print(json.dumps({
        "tag": tag, "shard": shard, "clips": len(ready), "load_s": round(load_s, 1),
        "prompt_s": round(prompt_s, 1), "copy_errors": copy_errors,
    }), flush=True)

    manifest = []
    errors = 0
    fallbacks = 0
    stopped = ""
    seeds_done = []
    last_gen_s = 0.0

    def batched(group, group_prompts, seed_value):
        nonlocal fallbacks
        caps = [item["cap"] for item in group]
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        try:
            wavs, sr = tts.generate_voice_clone(
                text=[item["text"] for item in group],
                language=[item["dialect"] for item in group],
                voice_clone_prompt=group_prompts,
                non_streaming_mode=True,
                max_new_tokens=max(1, int(max(caps) * 12.5)),
                **decode,
            )
            return list(wavs), int(sr)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(group) == 1:
                raise
            fallbacks += 1
            mid = len(group) // 2
            left, sr = batched(group[:mid], group_prompts[:mid], seed_value)
            right, sr = batched(group[mid:], group_prompts[mid:], seed_value + mid)
            return left + right, sr

    for seed in seed_list:
        elapsed = time.time() - started
        if seed != seed_list[0] and elapsed + last_gen_s > budget_s:
            stopped = f"not starting seed {seed}"
            print(json.dumps({"tag": tag, "shard": shard, "stopped": stopped, "elapsed": round(elapsed, 1)}), flush=True)
            break
        seed_t = time.time()
        made = 0
        for start in range(0, len(ready), batch_size):
            if time.time() - started > budget_s:
                stopped = f"budget during seed {seed}"
                break
            group = ready[start:start + batch_size]
            group_prompts = prompts[start:start + batch_size]
            seed_value = int(seed) * 1_000_003 + start
            try:
                wavs, sr = batched(group, group_prompts, seed_value)
            except Exception as exc:
                errors += len(group)
                print(json.dumps({
                    "tag": tag, "shard": shard, "seed": seed, "start": start,
                    "error": str(exc)[:240],
                }), flush=True)
                continue
            for item, wave in zip(group, wavs):
                name = f"{seed}_{item['i']:05d}.wav" if item["mode"] == "xvec" else f"icl_{seed}_{item['i']:05d}.wav"
                dest = wav_root / name
                try:
                    dur = _write_16k(dest, wave, sr)
                    own_cap = item["cap"]
                    manifest.append({
                        "i": item["i"], "clip_id": item["clip_id"], "dialect": item["dialect"],
                        "text": item["text"], "seed": seed, "name": name, "mode": item["mode"],
                        "dur": round(dur, 3), "near_cap": dur >= 0.95 * own_cap,
                    })
                    made += 1
                except Exception as exc:
                    errors += 1
                    print(json.dumps({"wav_error": item["i"], "seed": seed, "error": str(exc)[:160]}), flush=True)
            if start % (batch_size * 10) == 0:
                print(json.dumps({
                    "tag": tag, "shard": shard, "seed": seed, "made": made,
                    "elapsed": round(time.time() - started, 1),
                    "dollars": round((time.time() - started) / 3600 * RATE, 3),
                }), flush=True)
        last_gen_s = time.time() - seed_t
        if made == len(ready) and not stopped:
            seeds_done.append(seed)
        print(json.dumps({
            "tag": tag, "shard": shard, "seed": seed, "made": made,
            "seed_s": round(last_gen_s, 1), "stopped": stopped,
        }), flush=True)
        if stopped:
            break

    man_path = out_dir / f"manifest_{shard}.jsonl"
    with man_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_tar = tar_path.with_suffix(".tar.tmp")
    if tmp_tar.exists():
        tmp_tar.unlink()
    with tarfile.open(tmp_tar, "w") as archive:
        for row in manifest:
            archive.add(wav_root / row["name"], arcname=row["name"])
    os.replace(tmp_tar, tar_path)
    shutil.rmtree(local, ignore_errors=True)

    wall = time.time() - started
    done = seeds_done == list(seed_list) and errors == 0 and copy_errors == 0
    meta = {
        "tag": tag, "shard": shard, "ckpt": ckpt, "clips": len(ready),
        "seeds": list(seed_list), "seeds_done": seeds_done,
        "n": len(manifest), "errors": errors, "copy_errors": copy_errors,
        "fallbacks": fallbacks, "stopped": stopped, "done": done,
        "batch_size": batch_size,
        "seed_formula": "torch.manual_seed(seed * 1000003 + batch_start) before each batch; clips sorted by cap then index",
        "load_s": round(load_s, 1), "prompt_s": round(prompt_s, 1),
        "wall_s": round(wall, 1), "dollars": round(wall / 3600 * RATE, 4),
        "decode": decode,
        "generation_config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
        "generation_config": {
            key: cfg.get(key)
            for key in (
                "do_sample", "temperature", "top_p", "top_k", "repetition_penalty",
                "subtalker_dosample", "subtalker_temperature", "subtalker_top_p", "subtalker_top_k",
            )
        },
    }
    _write(meta_path, meta)
    print(json.dumps({k: meta[k] for k in ("tag", "shard", "done", "n", "seeds_done", "dollars", "fallbacks", "stopped")}), flush=True)
    return meta


def noise_score(tag, shard, root=None):
    """Cohere on the tar from noise_generate. One row per clip per seed."""
    from plan12_text import wer

    started = time.time()
    out_dir = (Path(root) if root else NOISE) / tag
    manifest = [
        json.loads(line)
        for line in (out_dir / f"manifest_{shard}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    part = out_dir / f"score_{shard}.jsonl"
    done = set()
    if part.is_file():
        for line in part.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("wer") is not None:
                done.add((int(row["i"]), int(row["seed"])))
    pending = [row for row in manifest if (int(row["i"]), int(row["seed"])) not in done]
    local = Path(f"/tmp/plan13_noise_score_{tag}_{shard}")
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    import tarfile
    with tarfile.open(out_dir / f"audio_{shard}.tar") as archive:
        archive.extractall(local)
    processor, model = _load_cohere()
    wrote = 0
    for row in pending:
        item = {
            "i": row["i"], "clip_id": row.get("clip_id"), "dialect": row.get("dialect"),
            "text": row.get("text") or "", "seed": row["seed"], "dur": row.get("dur"),
            "near_cap": row.get("near_cap"), "hyp": "", "error": "",
        }
        path = local / row["name"]
        try:
            item["hyp"] = _transcribe(processor, model, path)
            item["wer"] = wer(item["text"], item["hyp"])
        except Exception as exc:
            item["error"] = str(exc)[:240]
            item["wer"] = 1.0
        _append(part, item)
        wrote += 1
        if wrote % 50 == 0:
            print(json.dumps({
                "tag": tag, "shard": shard, "scored": len(done) + wrote,
                "elapsed": round(time.time() - started, 1),
            }), flush=True)
    shutil.rmtree(local, ignore_errors=True)
    wall = time.time() - started
    meta = {
        "tag": tag, "shard": shard, "n": len(done) + wrote, "new": wrote,
        "wall_s": round(wall, 1), "dollars": round(wall / 3600 * RATE, 4),
    }
    _write(out_dir / f"score_{shard}.json", meta)
    print(json.dumps(meta), flush=True)
    return meta


def jais_ref_body():
    """Jais on the Emirati reference recordings. Cohere on those files is already stored."""
    import soundfile as sf

    from plan12_text import wer
    from retx_jais import load_jais, transcribe_items

    started = time.time()
    rows = [row for row in _rows() if row.get("dialect") == "ar_uae"]
    items = []
    for row in rows:
        path = row.get("ref_audio") or row.get("audio")
        dur = 0.0
        try:
            dur = float(sf.info(path).duration)
        except Exception:
            dur = 0.0
        items.append({"clip_id": path, "duration": dur, "i": int(row["i"]), "text": row.get("text") or ""})
    processor, model = load_jais(JAIS_DIR)
    outs = transcribe_items(processor, model, items, batch_size=4)
    scored = []
    for row, out in zip(items, outs):
        hyp = (out or {}).get("transcript") or ""
        scored.append({
            "i": row["i"],
            "text": row["text"],
            "hyp": hyp,
            "wer": wer(row["text"], hyp),
            "error": (out or {}).get("error") or "",
            "refused": bool((out or {}).get("refused")),
        })
    wall = time.time() - started
    wers = [row["wer"] for row in scored]
    report = {
        "dialect": "ar_uae", "n": len(scored),
        "wer": round(sum(wers) / len(wers), 4) if wers else None,
        "errors": sum(1 for row in scored if row["error"] or row["refused"]),
        "wall_s": round(wall, 1), "dollars": round(wall / 3600 * RATE, 4),
        "rows": scored,
    }
    _write(NOISE / "jais_ref.json", report)
    print(json.dumps({k: report[k] for k in ("n", "wer", "errors", "dollars")}), flush=True)
    return {k: report[k] for k in ("n", "wer", "errors", "wall_s", "dollars")}
