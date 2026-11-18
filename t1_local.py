"""Score a checkpoint on this machine. No Modal volume and no training.

Reads QWEN_DATA_ROOT (default /data). Paths inside the exam that start with
/data are rewritten onto that root. Generation uses plan13_eval.noise_generate.
Scoring is the T1 order: Cohere, Jais on Gulf and Levantine Part B, WavLM, UTMOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tarfile
import time
from collections import defaultdict
from pathlib import Path

from plan13_eval import _load_cohere, _mono16, _transcribe, noise_generate
from t1_score import JAIS_DIALECTS, health, score_pair

LANG = {
    "emirati": "ar_uae",
    "kuwaiti": "ar_gulf",
    "najdi": "ar_sau",
    "hijazi": "ar_sau",
    "egyptian": "ar_egy",
    "levantine": "ar_lev",
    "msa": "ar_msa",
}


def data_root() -> str:
    return os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"


def remap(path: str) -> str:
    text = str(path)
    root = data_root()
    if text == "/data" or text.startswith("/data/"):
        return root + text[len("/data"):]
    return text


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def other_prompt(item_id, seed, mode, speaker, pool):
    others = [(key, path) for key, path in pool if key != speaker]
    if not others:
        return None
    digest = hashlib.sha1(f"13|{item_id}|{seed}|{mode}".encode()).hexdigest()
    rng = random.Random(int(digest[:8], 16))
    return rng.choice(others)


def prepare_rows(exam_path: Path, limit: int, modes: list[str]) -> list[dict]:
    rows = []
    for row in read_jsonl(exam_path):
        row = dict(row)
        row["part"] = row.get("part") or "B"
        row["ref_audio"] = remap(row.get("ref_audio") or "")
        if row.get("prompt_clip"):
            row["prompt_clip"] = row["prompt_clip"]
        rows.append(row)
    if limit:
        rows = rows[:limit]
    rows.sort(key=lambda row: (row["part"], row["id"]))
    prepared = []
    for index, row in enumerate(rows):
        row["i"] = index
        row["clip_id"] = row["id"]
        row["dialect_exam"] = row["dialect"]
        row["dialect"] = LANG.get(row["dialect"], row["dialect"])
        for mode in modes:
            if mode == "icl" and not (row.get("prompt_text") or "").strip():
                raise SystemExit(f"{row['id']} has no prompt transcript for in-context mode")
            item = dict(row)
            item["mode"] = mode
            if mode == "icl":
                item["i"] = index + 100000
            prepared.append(item)
    return prepared, rows


def score_dir(out_dir: Path, exam: dict, shard: int = 0) -> dict:
    import soundfile as sf
    import torch

    manifest = [
        json.loads(line)
        for line in (out_dir / f"manifest_{shard}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    local = Path(f"/tmp/t1_local_score_{shard}")
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    with tarfile.open(out_dir / f"audio_{shard}.tar") as archive:
        archive.extractall(local)
    pool = []
    seen = set()
    for row in exam.values():
        key = row.get("prompt_clip") or row["id"]
        path = row.get("ref_audio") or ""
        if path and key not in seen:
            seen.add(key)
            pool.append((key, path))
    started = time.time()
    processor, model = _load_cohere()
    scored = []
    for row in manifest:
        exam_row = exam[row["clip_id"]]
        wav = str(local / row["name"])
        item = {
            "part": exam_row["part"],
            "id": exam_row["id"],
            "dialect": exam_row.get("dialect_exam") or exam_row["dialect"],
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
            item.update(score_pair(exam_row["text"], item["hyp"]))
            if item["dur"] is None:
                item["dur"] = round(float(sf.info(wav).duration), 3)
        except Exception as exc:
            item["error"] = str(exc)[:300]
            item["wer"] = 1.0
            item["ler"] = 1.0
            item["dropped_ending"] = 1.0
            item["dropped_words"] = 1.0
        scored.append(item)
    del model
    torch.cuda.empty_cache()

    jais_rows = [
        item for item in scored
        if item["part"] == "B" and item["dialect"] in JAIS_DIALECTS and item.get("wav")
    ]
    if jais_rows:
        from retx_jais import load_jais, transcribe_items

        jais_proc, jais_model = load_jais()
        outs = transcribe_items(
            jais_proc, jais_model,
            [{"clip_id": item["wav"], "duration": item.get("dur") or 0} for item in jais_rows],
            batch_size=1,
        )
        for item, out in zip(jais_rows, outs):
            hyp = (out or {}).get("transcript") or ""
            item["jais_hyp"] = hyp
            try:
                item["jais_wer"] = score_pair(item["text"], hyp)["wer"]
            except Exception as exc:
                item["error"] = (item.get("error") or "") + " jais " + str(exc)[:160]
        del jais_model
        torch.cuda.empty_cache()

    from transformers import AutoFeatureExtractor, WavLMForXVector

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()
    for item in scored:
        item["sim"] = None
        try:
            hyp_y, hyp_rms = _mono16(item["wav"])
            ref_y, _ = _mono16(item["ref_audio"])
            item["rms"] = hyp_rms
            other = other_prompt(item["id"], item.get("seed"), item.get("mode") or "xvec", item["speaker_key"], pool)
            waves = [ref_y, hyp_y]
            if other is not None:
                other_y, _ = _mono16(other[1])
                waves.append(other_y)
            inputs = fe(waves, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {key: value.to(sv.device) for key, value in inputs.items()}
            with torch.inference_mode():
                emb = sv(**inputs).embeddings
            item["sim"] = float(torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0))
        except Exception as exc:
            item["rms"] = item.get("rms") or 0.0
            item["error"] = (item.get("error") or "") + " wavlm " + str(exc)[:160]
    del sv
    torch.cuda.empty_cache()

    import utmosv2

    utmos = utmosv2.create_model(pretrained=True)
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

    for item in scored:
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
        item.pop("wav", None)
    part_path = out_dir / f"score_{shard}.jsonl"
    with part_path.open("w", encoding="utf-8") as handle:
        for item in scored:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    usable = [item for item in scored if not item.get("broken") and not item.get("error")]
    wers = [item.get("wer") for item in usable if item.get("wer") is not None]
    meta = {
        "n": len(scored),
        "n_usable": len(usable),
        "wer_mean": (sum(wers) / len(wers)) if wers else None,
        "wall_s": round(time.time() - started, 1),
        "score_path": str(part_path),
    }
    (out_dir / f"score_{shard}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--exam", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--modes", default="xvec,icl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--phase", choices=("all", "generate", "score"), default="all")
    args = parser.parse_args()
    seeds = tuple(int(part) for part in args.seeds.split(",") if part)
    modes = [part for part in args.modes.split(",") if part]
    prepared, exam_rows = prepare_rows(Path(args.exam), args.limit, modes)
    exam = {}
    for row in exam_rows:
        exam[row["id"]] = row
    gen_rows = []
    for row in prepared:
        gen_rows.append({
            "i": row["i"],
            "clip_id": row["id"],
            "dialect": row["dialect"],
            "text": row["text"],
            "ref_audio": row["ref_audio"],
            "mode": row["mode"],
            "part": row["part"],
            "prompt_text": row.get("prompt_text") or "",
        })
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase in ("all", "generate"):
        meta = noise_generate(
            args.ckpt, "run", 0, shards=1, batch_size=args.batch_size, budget_s=3600,
            root=str(out), seeds=seeds, rows_in=gen_rows,
        )
        print(json.dumps({"generate": {k: meta.get(k) for k in ("done", "n", "errors", "wall_s", "load_s")}}), flush=True)
    if args.phase == "all":
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    if args.phase in ("all", "score"):
        score_dir(out / "run", exam, 0)


if __name__ == "__main__":
    main()
