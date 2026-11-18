"""Dev-loss table for the saved Stage B1 checkpoints. Does not train."""
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

DEV = f"{_DATA}/plan13/tier1_dev.jsonl"
LOG = f"{_DATA}/runs/stageB1/train.log"
OUT = Path(f"{_DATA}/plan13/devloss.json")
DIALECTS = [
    "ar_msa", "ar_egy", "ar_sau", "ar_uae", "ar_gulf", "ar_omn",
    "ar_lev", "ar_irq", "ar_mar", "ar_tun", "ar_dza", "unknown",
]
CKPTS = [
    (4000, f"{_DATA}/runs/stageB1/step-004000-base"),
    (5648, f"{_DATA}/runs/stageB1/step-005648-base"),
    (11296, f"{_DATA}/runs/stageB1/step-011296-base"),
    (16944, f"{_DATA}/runs/stageB1/step-016944-base"),
    (22592, f"{_DATA}/runs/stageB1/step-022592-base"),
    (22595, f"{_DATA}/runs/stageB1/step-022595-base"),
    (28240, f"{_DATA}/runs/stageB1/step-028240-base"),
    (33888, f"{_DATA}/runs/stageB1/step-033888-base"),
    (39536, f"{_DATA}/runs/stageB1/step-039536-base"),
    (45184, f"{_DATA}/runs/stageB1/step-045184-base"),
    (45190, f"{_DATA}/runs/stageB1/step-045190-base"),
]
PER_DIALECT = 40
SEED = 13
RATE = 1.95


def logged_vals():
    step = None
    rows = []
    with open(LOG, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            matched = re.search(r"\bstep (\d+)/", line)
            if matched:
                step = int(matched.group(1))
            if "val extra" in line or "val main loss" not in line:
                continue
            matched = re.search(r"val main loss ([0-9]+\.[0-9]+)", line)
            if matched and step is not None:
                rows.append({"step": step, "val": float(matched.group(1))})
    return rows


def nearest(rows, step):
    return min(rows, key=lambda row: (abs(row["step"] - step), row["step"] < step))


def sample_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("dialect") or "?"].append(row)
    picked = []
    for dialect in sorted(grouped):
        pool = sorted(grouped[dialect], key=lambda row: row.get("clip_id") or "")
        rng = random.Random(SEED)
        rng.shuffle(pool)
        take = pool[:PER_DIALECT]
        for row in take:
            row["_sample_dialect"] = dialect
        picked.extend(take)
    return picked


def lang_emb_from_checkpoint(model, dialects, lang_map):
    import torch

    weight = model.talker.model.codec_embedding.weight
    rows = [weight[int(lang_map[name])].detach().float() for name in dialects]
    emb = torch.nn.Embedding(len(dialects), weight.shape[1])
    emb.weight.data.copy_(torch.stack(rows))
    return emb.to(device=weight.device, dtype=weight.dtype)


def eval_checkpoint(path, rows):
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from transformers import AutoConfig

    import train_dialects as td

    tts = Qwen3TTSModel.from_pretrained(
        path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model = tts.model.eval()
    config = AutoConfig.from_pretrained(path)
    lang_map = dict(getattr(config.talker_config, "codec_language_id", None) or {})
    missing = [name for name in DIALECTS if name not in lang_map]
    if missing:
        raise RuntimeError(f"{path} is missing dialect ids {missing}")
    lang_emb = lang_emb_from_checkpoint(model, DIALECTS, lang_map)
    d2i = {name: index for index, name in enumerate(DIALECTS)}
    dataset = td.build_dataset_cls()(rows, tts.processor, config, d2i, 0.0, 375, 0.0)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row.get("dialect") or "?"].append(index)
    by = {}
    batch_losses = []
    for dialect, indexes in sorted(grouped.items()):
        losses = []
        for start in range(0, len(indexes), 4):
            batch = dataset.collate_fn([dataset[index] for index in indexes[start:start + 4]])
            with torch.inference_mode():
                _total, main, _sub, _spk = td.forward_loss(model, lang_emb, batch, 0.3)
            losses.append(float(main))
            batch_losses.append(float(main))
        by[dialect] = {
            "n": len(indexes),
            "main": round(sum(losses) / len(losses), 4),
        }
    overall = round(sum(batch_losses) / len(batch_losses), 4)
    del tts, model, lang_emb
    torch.cuda.empty_cache()
    return {"overall_sample": overall, "by_dialect": by}


def run(commit):
    import sys
    import time

    sys.path.insert(0, "/recipe")
    import train_dialects as td

    started = time.time()
    logged = logged_vals()
    best = min(logged, key=lambda row: row["val"])
    saved = []
    for step, path in CKPTS:
        near = nearest(logged, step)
        saved.append({
            "step": step,
            "ckpt": path,
            "logged_step": near["step"],
            "logged_val": near["val"],
            "best": near["step"] == best["step"],
        })
    rows = sample_rows(td.load_rows(DEV))
    report = {
        "best_logged_step": best["step"],
        "best_logged_val": best["val"],
        "sample_per_dialect": PER_DIALECT,
        "seed": SEED,
        "sample_n": len(rows),
        "note": (
            "logged_val is the validation main loss written during training. "
            "The best value 1.2941 is at step 45000, which was not saved. "
            "Per-dialect numbers are a fixed sample of the dev set, seed 13, up to 40 clips each."
        ),
        "checkpoints": saved,
        "scored": [],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for item in saved:
        t0 = time.time()
        cell = eval_checkpoint(item["ckpt"], rows)
        cell["step"] = item["step"]
        cell["seconds"] = round(time.time() - t0, 1)
        report["scored"].append(cell)
        report["wall_s"] = round(time.time() - started, 1)
        report["dollars"] = round(report["wall_s"] / 3600 * RATE, 4)
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, OUT)
        commit()
        print(json.dumps({
            "step": item["step"], "sample": cell["overall_sample"],
            "logged": item["logged_val"], "seconds": cell["seconds"],
            "dollars": report["dollars"],
        }), flush=True)
        if report["dollars"] > 12:
            report["stopped"] = "past $12"
            stop = OUT.with_suffix(".json.tmp")
            stop.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(stop, OUT)
            commit()
            break
    return {
        "best_logged_step": best["step"],
        "best_logged_val": best["val"],
        "dollars": report.get("dollars"),
        "scored_steps": [cell["step"] for cell in report["scored"]],
    }
