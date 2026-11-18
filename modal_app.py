"""Modal launcher for Stage A on the YouTube Arabic corpus (volume qwen3-arabic-tts-v2 at /data).

One-time:  modal volume put qwen3-arabic-tts-v2 ../data/edge_silence_bad_clips.xlsx /edge_silence_bad_clips.xlsx

Steps (run from this folder, in order):
  modal run modal_app.py::build                 # filtered train/val/eval jsonl + build_report.json
  modal run modal_app.py::encode                # audio -> codec tokens, sharded over GPUs
  modal run modal_app.py::check                 # prints token IDs; must show 11 dialect IDs, no errors
  modal run modal_app.py::probe                 # alignment table on the base model; no training
  modal run modal_app.py::e0                    # after approval: 1 H100 into /data/runs/e0_fix
  modal run --detach modal_app.py::train        # full Stage A on 8x H100 (resumable)
  modal run modal_app.py::evaluate --ckpt /data/runs/stageA/step-XXXXXX-base

If train stops (24 h limit, preemption), run the same train command again: it resumes.
Get files:  modal volume get qwen3-arabic-tts-v2 /runs/e0_fix/eval ./e0_fix_eval
"""
import glob
import os
import shutil
import subprocess
import threading

import modal
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

VOL_NAME = os.environ.get("TTS_VOLUME", "qwen3-arabic-tts-v2")
RECIPE = "/recipe"
WORK = f"{_DATA}/tts_stageA"
MODELS = f"{_DATA}/models"
BASE = f"{MODELS}/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER = f"{MODELS}/Qwen3-TTS-Tokenizer-12Hz"
DIALECTS = "ar_msa,ar_egy,ar_sau,ar_uae,ar_gulf,ar_omn,ar_lev,ar_irq,ar_mar,ar_tun,ar_dza"

vol = modal.Volume.from_name(VOL_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "sox", "ffmpeg", "libsndfile1")
    .pip_install("torch", "torchaudio")
    .pip_install("qwen-tts", "transformers==4.57.3", "accelerate", "safetensors", "huggingface_hub",
                 "soundfile", "librosa", "pyloudnorm", "numpy", "pandas", "openpyxl", "tensorboard")
    .run_commands("git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git /opt/Qwen3-TTS")
    .env({"HF_HOME": "/data/hf_cache", "QWEN3_TTS_DIR": "/opt/Qwen3-TTS",
          "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(os.path.dirname(os.path.abspath(__file__)), RECIPE,
                   ignore=["*.zip", "runs", "data", "open_data", "__pycache__", ".venv",
                           "e0_eval", "e0_fix_eval", "stageA_eval", "tb", "phase0_en"])
)
app = modal.App("qwen3-arabic-tts-stageA", image=image)


def sh(cmd, commit_every=None):
    """Run a shell command in the recipe dir; commit the volume periodically and at the end."""
    print("+", cmd, flush=True)
    stop = threading.Event()
    if commit_every:
        def loop():
            while not stop.wait(commit_every):
                try:
                    vol.commit()
                except Exception as e:  # noqa
                    print("volume commit failed (will retry):", e, flush=True)
        threading.Thread(target=loop, daemon=True).start()
    rc = subprocess.call(cmd, shell=True, cwd=RECIPE)
    stop.set()
    vol.commit()
    if rc:
        raise RuntimeError(f"command failed ({rc}): {cmd}")


def ensure_models():
    from huggingface_hub import snapshot_download
    for repo, path in (("Qwen/Qwen3-TTS-Tokenizer-12Hz", TOKENIZER), ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", BASE)):
        if not os.path.exists(os.path.join(path, "config.json")):
            snapshot_download(repo, local_dir=path)
    vol.commit()


# ------------------------------------------------------------------ 1. build the dataset
@app.function(cpu=16, memory=65536, timeout=8 * 3600, volumes={"/data": vol})
def build_remote(bad_list: str, extra: str):
    sh(f"python build_yt_stageA.py --manifest {_DATA}/data/yt_full.jsonl "
       f"--audio_root_24k {_DATA}/yt/yt_clean_24k --bad_list {bad_list} --out_dir {WORK} {extra}")
    print(open(f"{WORK}/build_report.json").read())


@app.local_entrypoint()
def build(bad_list: str = f"{_DATA}/edge_silence_bad_clips.xlsx", extra: str = "--f_channel speaker"):
    build_remote.remote(bad_list, extra)


# ------------------------------------------------------------------ 2. codec encoding
@app.function(cpu=8, memory=65536, timeout=3 * 3600, volumes={"/data": vol})
def make_shards(n: int):
    ensure_models()
    lines = open(f"{WORK}/stageA_train.jsonl", encoding="utf-8").readlines()
    os.makedirs(f"{WORK}/shards", exist_ok=True)
    k = -(-len(lines) // n)
    paths = []
    for i in range(n):
        p = f"{WORK}/shards/train_{i:03d}.jsonl"
        if not os.path.exists(p.replace(".jsonl", "_codes.jsonl")):
            with open(p, "w", encoding="utf-8") as f:
                f.writelines(lines[i * k:(i + 1) * k])
            paths.append(p)
    if not os.path.exists(f"{WORK}/stageA_val_codes.jsonl"):
        paths.append(f"{WORK}/stageA_val.jsonl")
    vol.commit()
    return paths


@app.function(gpu="L40S", cpu=8, memory=32768, timeout=12 * 3600, volumes={"/data": vol}, retries=2)
def encode_shard(path: str):
    out = path.replace(".jsonl", "_codes.jsonl")
    if os.path.exists(out):
        return out
    sh(f"DEVICE=cuda:0 TOKENIZER={TOKENIZER} bash prepare_codes.sh {path} {out}.tmp 2000 && mv {out}.tmp {out}")
    return out


@app.function(cpu=4, memory=16384, timeout=3 * 3600, volumes={"/data": vol})
def merge_codes():
    vol.reload()
    parts = sorted(glob.glob(f"{WORK}/shards/train_*_codes.jsonl"))
    inputs = sorted(p for p in glob.glob(f"{WORK}/shards/train_*.jsonl") if not p.endswith("_codes.jsonl"))
    missing = [p for p in inputs if not os.path.exists(p.replace(".jsonl", "_codes.jsonl"))]
    if missing or not os.path.exists(f"{WORK}/stageA_val_codes.jsonl"):
        raise RuntimeError(f"{len(missing)} shard(s) not encoded yet (or val missing): run encode again. {missing[:3]}")
    n = 0
    with open(f"{WORK}/stageA_train_codes.jsonl", "w", encoding="utf-8") as g:
        for p in parts:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    g.write(line); n += 1
    vol.commit()
    print(f"merged {len(parts)} shards, {n:,d} rows")


@app.local_entrypoint()
def encode(shards: int = 40):
    paths = make_shards.remote(shards)
    print(f"encoding {len(paths)} files")
    for out in encode_shard.map(paths, return_exceptions=True):
        print("done:", out)
    merge_codes.remote()


# ------------------------------------------------------------------ 3. checks + sanity run
@app.function(gpu="L40S", memory=32768, timeout=1800, volumes={"/data": vol})
def check_remote():
    ensure_models()
    sh(f"python train_dialects.py --init_model_path {BASE} --dialects {DIALECTS} "
       f"--train_jsonl x --check_only")


@app.local_entrypoint()
def check():
    check_remote.remote()


@app.function(gpu="L40S", memory=65536, timeout=3 * 3600, volumes={"/data": vol})
def probe_remote(ckpt: str, n: int):
    ensure_models()
    ckpt_arg = f"--ckpt {ckpt}" if ckpt else ""
    sh(f"python probe_alignment.py --init_model_path {BASE} --val_jsonl {WORK}/stageA_val_codes.jsonl "
       f"--n {n} {ckpt_arg}")


@app.local_entrypoint()
def probe(ckpt: str = "", n: int = 240):
    probe_remote.remote(ckpt, n)


@app.function(gpu="H100", cpu=12, memory=131072, timeout=8 * 3600, volumes={"/data": vol})
def e0_remote(rows: int, lr: float):
    ensure_models()
    out = f"{_DATA}/runs/e0_fix"
    sub = f"{WORK}/e0_fix_train_codes.jsonl"
    sh(f"python probe_alignment.py --init_model_path {BASE} --val_jsonl {WORK}/stageA_val_codes.jsonl --n 240")
    sh(f"head -n {rows} {WORK}/stageA_train_codes.jsonl > {sub}")
    sh(f"python train_dialects.py --init_model_path {BASE} --dialects {DIALECTS} "
       f"--train_jsonl {sub} --val_jsonl {WORK}/stageA_val_codes.jsonl --output_dir {out} "
       f"--lr {lr} --epochs 2 --batch_size 8 --grad_accum 4 --eval_every 100 --val_batches 20",
       commit_every=600)
    ckpts = sorted(glob.glob(f"{out}/step-*-base"))
    if not ckpts:
        raise RuntimeError(f"no checkpoints in {out}")
    for ckpt in ckpts:
        sh(f"python eval_dialects.py --ckpt {ckpt} --eval_jsonl {WORK}/eval.jsonl --out_dir {out}/eval "
           f"--max_per_dialect 10 --swap_n 5", commit_every=120)


@app.local_entrypoint()
def e0(rows: int = 8000, lr: float = 5e-6):
    e0_remote.remote(rows, lr)


# ------------------------------------------------------------------ 4. full Stage A
@app.function(gpu="H100:8", cpu=64, memory=393216, timeout=24 * 3600, volumes={"/data": vol},
              retries=modal.Retries(max_retries=3, initial_delay=30.0))
def train_remote(out: str, lr: float, epochs: float, batch_size: int, extra: str):
    import torch
    n = torch.cuda.device_count()
    sh(f"accelerate launch --multi_gpu --num_processes {n} --num_machines 1 --mixed_precision bf16 "
       f"--dynamo_backend no train_dialects.py --init_model_path {BASE} --dialects {DIALECTS} "
       f"--train_jsonl {WORK}/stageA_train_codes.jsonl --val_jsonl {WORK}/stageA_val_codes.jsonl "
       f"--output_dir {out} --lr {lr} --epochs {epochs} --batch_size {batch_size} --grad_accum 1 "
       f"--eval_every 500 --resume 1 --export_speakers 0 {extra}",
       commit_every=900)


@app.local_entrypoint()
def train(out: str = f"{_DATA}/runs/stageA", lr: float = 5e-6, epochs: float = 2, batch_size: int = 4,
          extra: str = "--save_every 2000"):
    train_remote.remote(out, lr, epochs, batch_size, extra)


# ------------------------------------------------------------------ 5. evaluation
@app.function(gpu="L40S", memory=32768, timeout=6 * 3600, volumes={"/data": vol})
def evaluate_remote(ckpt: str, max_per_dialect: int, extra: str):
    vol.reload()
    sh(f"python eval_dialects.py --ckpt {ckpt} --eval_jsonl {WORK}/eval.jsonl "
       f"--out_dir {os.path.dirname(ckpt)}/eval --max_per_dialect {max_per_dialect} {extra}",
       commit_every=120)


@app.local_entrypoint()
def evaluate(ckpt: str, max_per_dialect: int = 30, extra: str = ""):
    evaluate_remote.remote(ckpt, max_per_dialect, extra)


# ------------------------------------------------------------------ 6. phase 0 only: data recount and English smoke clips
@app.function(cpu=2, memory=8192, timeout=3600, volumes={"/data": vol})
def data_check_remote():
    vol.reload()
    sh("python - <<'PY'\n"
       "import json\n"
       f"r = json.load(open('{WORK}/build_report.json'))\n"
       "t = r['totals']\n"
       "print('train_rows', t['train_rows'])\n"
       "print('val_rows', t['val_rows'])\n"
       "print('eval_rows', t['eval_rows'])\n"
       "print('train_hours_effective', t['train_hours_effective'])\n"
       "for lab, d in r['labels'].items():\n"
       "    band = r['bandwidth'][lab]['band_limited_pct']\n"
       "    print(f\"label {lab} channels={d['train_channels']} eff_h={d['train_hours_effective']} band={band}\")\n"
       "n = 0\n"
       f"with open('{WORK}/stageA_train_codes.jsonl', encoding='utf-8') as f:\n"
       "    for line in f:\n"
       "        if line.strip():\n"
       "            n += 1\n"
       "print('codes_rows', n)\n"
       "PY")


@app.local_entrypoint()
def data_check():
    data_check_remote.remote()


@app.function(gpu="L40S", memory=65536, timeout=3600, volumes={"/data": vol})
def smoke_en_remote():
    ensure_models()
    sh("python - <<'PY'\n"
       "import json, soundfile as sf\n"
       f"row = json.loads(open('{WORK}/stageA_val.jsonl', encoding='utf-8').readline())\n"
       "print('ref_audio', row['audio'])\n"
       "print('ref_dialect', row.get('dialect'))\n"
       "info = sf.info(row['audio'])\n"
       "print(f'ref_dur_s={info.duration:.2f}')\n"
       "open('/tmp/ref_audio.txt', 'w').write(row['audio'])\n"
       "PY")
    sh(f"python infer.py --ckpt {BASE} --ref_audio \"$(cat /tmp/ref_audio.txt)\" "
       f"--language english --sentences eval_sentences_en.txt --out_dir {_DATA}/runs/phase0_en")
    sh("python - <<'PY'\n"
       "from pathlib import Path\n"
       "import soundfile as sf\n"
       "import torch\n"
       "from transformers import pipeline\n"
       "asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',\n"
       "               torch_dtype=torch.float16, device=0)\n"
       f"for p in sorted(Path('{_DATA}/runs/phase0_en').rglob('*.wav')):\n"
       "    info = sf.info(p)\n"
       "    text = asr(str(p), generate_kwargs={'language': 'english', 'task': 'transcribe',\n"
       "                                       'return_timestamps': True})['text']\n"
       "    print(f'FILE {p} dur_s={info.duration:.2f} text={text}', flush=True)\n"
       "PY")


@app.local_entrypoint()
def smoke_en():
    smoke_en_remote.remote()


# ------------------------------------------------------------------ 7. blind listening kit
BLIND_CANDS = (f"C1={_DATA}/runs/e0_fix/step-000500-base,C2={_DATA}/runs/stageA/step-024000-base,"
               f"C3={_DATA}/runs/stageA/step-032406-base")


@app.function(gpu="L40S", cpu=8, memory=65536, timeout=6 * 3600, volumes={"/data": vol})
def blind_kit_remote(cands: str, out_dir: str, extra: str):
    vol.reload()
    sh(f"python make_blind_kit.py --eval_jsonl {WORK}/eval.jsonl --cands {cands} --out_dir {out_dir} {extra}",
       commit_every=600)


@app.local_entrypoint()
def blind_kit(cands: str = BLIND_CANDS, out_dir: str = f"{_DATA}/blind_kit", extra: str = ""):
    """Render anonymised clips. Share only the kit. Never share private/."""
    blind_kit_remote.remote(cands, out_dir, extra)


# ------------------------------------------------------------------ 8. automatic eval: loss per dialect, then paired generation
AUTO_CKPTS = [
    (BASE, f"{_DATA}/runs/base_eval"),
    (f"{_DATA}/runs/e0_fix/step-000500-base", f"{_DATA}/runs/e0_fix/eval"),
    (f"{_DATA}/runs/stageA/step-008000-base", f"{_DATA}/runs/stageA/eval"),
    (f"{_DATA}/runs/stageA/step-016000-base", f"{_DATA}/runs/stageA/eval"),
    (f"{_DATA}/runs/stageA/step-024000-base", f"{_DATA}/runs/stageA/eval"),
    (f"{_DATA}/runs/stageA/step-032406-base", f"{_DATA}/runs/stageA/eval"),
]
VALLOSS_CKPTS = ",".join(ckpt for ckpt, _out in AUTO_CKPTS)
AUTO_EVAL_EXTRA = "--n_samples 3 --temperature 0.7 --no_swap"


@app.function(gpu="L40S", cpu=8, memory=65536, timeout=6 * 3600, volumes={"/data": vol})
def valloss_remote(ckpts: str, per_dialect: int):
    ensure_models()
    vol.reload()
    sh(f"python val_by_dialect.py --ckpts {ckpts} --data {WORK}/stageA_val_codes.jsonl "
       f"--per_dialect {per_dialect} --out {WORK}/val_by_dialect.csv",
       commit_every=300)


@app.local_entrypoint()
def valloss(ckpts: str = VALLOSS_CKPTS, per_dialect: int = 150):
    """Teacher-forced loss per dialect on the same held-out clips, including the untouched base."""
    valloss_remote.remote(ckpts, per_dialect)


@app.function(gpu="L40S", memory=65536, timeout=10 * 3600, volumes={"/data": vol})
def auto_eval_remote(max_per_dialect: int):
    vol.reload()
    for ckpt, out_dir in AUTO_CKPTS:
        sh(f"python eval_dialects.py --ckpt {ckpt} --eval_jsonl {WORK}/eval.jsonl "
           f"--out_dir {out_dir} --max_per_dialect {max_per_dialect} {AUTO_EVAL_EXTRA}",
           commit_every=180)


@app.local_entrypoint()
def auto_eval(max_per_dialect: int = 30):
    """Three draws at temperature 0.7 for the base model and the five fine-tunes. No swap set."""
    auto_eval_remote.remote(max_per_dialect)


@app.function(gpu="L40S", memory=65536, timeout=4 * 3600, volumes={"/data": vol})
def base_eval_remote(max_per_dialect: int):
    """Untouched base only. Dialect ids are absent, so eval falls back to language=auto."""
    vol.reload()
    sh(f"python eval_dialects.py --ckpt {BASE} --eval_jsonl {WORK}/eval.jsonl "
       f"--out_dir {_DATA}/runs/base_eval --max_per_dialect {max_per_dialect} {AUTO_EVAL_EXTRA}",
       commit_every=180)


@app.local_entrypoint()
def base_eval(max_per_dialect: int = 30):
    base_eval_remote.remote(max_per_dialect)


@app.function(gpu="L40S", memory=65536, timeout=4 * 3600, volumes={"/data": vol})
def step32406_eval_remote(max_per_dialect: int):
    """Only the last checkpoint. The earlier five already have their _n3_t0.7 results."""
    vol.reload()
    sh(f"python eval_dialects.py --ckpt {_DATA}/runs/stageA/step-032406-base "
       f"--eval_jsonl {WORK}/eval.jsonl --out_dir {_DATA}/runs/stageA/eval "
       f"--max_per_dialect {max_per_dialect} {AUTO_EVAL_EXTRA}",
       commit_every=180)


@app.local_entrypoint()
def step32406_eval(max_per_dialect: int = 30):
    step32406_eval_remote.remote(max_per_dialect)


@app.function(cpu=2, memory=4096, timeout=1800, volumes={"/data": vol})
def compare_remote(runs: str, kind: str, out_path: str):
    vol.reload()
    sh(f"python compare_evals.py --kind {kind} --runs {runs} | tee {out_path}")


@app.local_entrypoint()
def compare(runs: str, kind: str = "short", out_path: str = f"{_DATA}/tts_stageA/compare.txt"):
    """runs = space-separated label=/data/.../results.csv, baseline first."""
    compare_remote.remote(runs, kind, out_path)


@app.function(cpu=1, memory=2048, timeout=600, volumes={"/data": vol})
def pair_report_remote():
    vol.reload()
    sh(f"python pair_report.py --testers {_DATA}/blind_kit/private/testers "
       f"--log {_DATA}/blind_kit/private/answer_log --pairs {_DATA}/blind_kit/private/pairs_v2.csv "
       f"--out {_DATA}/blind_kit/private/pair_report.md")


@app.local_entrypoint()
def pair_report():
    """Write the listening results from the saved answers. Does not change the answers."""
    pair_report_remote.remote()
