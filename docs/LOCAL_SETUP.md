# Local GPU setup

You run training and T1 on your own machine. Modal wrappers are still in the repo. Leave `QWEN_DATA_ROOT` unset when you `modal run`. Set it for every local command. Manifests on the volume store absolute `/data/...` paths. `train_dialects.py` and `t1_local.py` rewrite those onto `QWEN_DATA_ROOT`. The default root is `/data`, so an unset variable still reads the volume.

## Software

- Linux, NVIDIA driver new enough for CUDA 12.4. The box that ran `docs/SMOKE_TEST.md` had driver 580.178 and an RTX 4090.
- Python 3.11.
- A checkout of https://github.com/QwenLM/Qwen3-TTS (the image clones `--depth 1` to `/opt/Qwen3-TTS`). `train_dialects.py` imports `finetuning/dataset.py` from `$QWEN3_TTS_DIR`. That module is not in this repo. It is in that checkout. Confirmed: `finetuning/dataset.py` defines `TTSDataset`.

Train stack and score stack pin different Transformers. Use two venvs.

```bash
python3.11 -m venv .venv-train
.venv-train/bin/pip install -U pip
.venv-train/bin/pip install -r requirements-train.txt

python3.11 -m venv .venv-score
.venv-score/bin/pip install -U pip
.venv-score/bin/pip install -r requirements-score.txt
```

`requirements-train.txt` is torch 2.6.0 (cu124), transformers 4.57.3, qwen-tts, accelerate. `requirements-score.txt` is torch 2.6.0, transformers >= 5.4.0, and UTMOSv2 from git. System packages: `ffmpeg`, `libsndfile`.

Copy `paths.env.example` to `paths.env`.

## What to download

Create a read-only B2 key. Do not commit it. The names in the lab env file are `B2_COMPANY_KEY_ID` and `B2_COMPANY_APPLICATION_KEY`. An older key only opens `arabicDatav1-pretraining` and cannot read these prefixes.

```bash
export RCLONE_CONFIG_B2DS_TYPE=b2
export RCLONE_CONFIG_B2DS_ACCOUNT="$B2_COMPANY_KEY_ID"
export RCLONE_CONFIG_B2DS_KEY="$B2_COMPANY_APPLICATION_KEY"
rclone copy b2ds:companyNames-dataset/11labs_gen_audios/arabic_dialects_voxcpm/audio "$QWEN_DATA_ROOT/exp_11labs/audio"
rclone copy b2ds:companyNames-dataset/YT_datadownload/arabic "$QWEN_DATA_ROOT/yt_b2"
```

These are not on B2, as far as the data note goes. They are on Modal volume `qwen3-arabic-tts-v2`. Download with the Modal CLI (read only):

```bash
modal volume get qwen3-arabic-tts-v2 exp_11labs/runs/step-005482-base "$QWEN_DATA_ROOT/exp_11labs/runs/step-005482-base"
modal volume get qwen3-arabic-tts-v2 runs/stageB1/step-045190-base "$QWEN_DATA_ROOT/runs/stageB1/step-045190-base"
modal volume get qwen3-arabic-tts-v2 exp_11labs/train_codes.jsonl "$QWEN_DATA_ROOT/exp_11labs/train_codes.jsonl"
modal volume get qwen3-arabic-tts-v2 exp_11labs/val_codes.jsonl "$QWEN_DATA_ROOT/exp_11labs/val_codes.jsonl"
modal volume get qwen3-arabic-tts-v2 plan13/tier1_v1.jsonl "$QWEN_DATA_ROOT/plan13/tier1_v1.jsonl"
modal volume get qwen3-arabic-tts-v2 tasks/t1_eval/testset "$QWEN_DATA_ROOT/tasks/t1_eval/testset"
modal volume get qwen3-arabic-tts-v2 retx/plan12/partD/weights "$QWEN_DATA_ROOT/retx/plan12/partD/weights"
modal volume get qwen3-arabic-tts-v2 models/Jais-ASR "$QWEN_DATA_ROOT/models/Jais-ASR"
```

Sizes measured while packing the smoke test are filled in at the bottom of `docs/SMOKE_TEST.md` (checkpoint, train jsonl, Cohere, Jais). They are not in a B2 listing.

## VRAM

Training at the historical batch of 4 used 41 GB on an H100 and only 22% of the GPU. Use an 80 GB GPU for that batch. A 24 GB card is below that measurement. The smoke test on the 4090 uses batch 1 and records the peak. If it runs out of memory, that is the result. Do not switch the smoke test to Modal.

T1 scoring loads Cohere, then Jais, then WavLM, then UTMOSv2, and drops each model before the next. The peak on the smoke machine is in `docs/SMOKE_TEST.md`. WavLM is `microsoft/wavlm-base-plus-sv` (small). Cohere and Jais are the large ones. Jais is three safetensors shards.

## Local commands

```bash
# 50 steps from s5482, batch 1, nothing kept
export QWEN_DATA_ROOT=/path/to/data
export QWEN3_TTS_DIR=/opt/Qwen3-TTS
export INIT=$QWEN_DATA_ROOT/exp_11labs/runs/step-005482-base
export TRAIN_JSONL=/tmp/smoke_train.jsonl   # a few dozen rows, not the full file
export VAL_JSONL=$QWEN_DATA_ROOT/exp_11labs/val_codes.jsonl
export OUT=/tmp/qwen-smoke
export BATCH=1 MAX_STEPS=50 LR=2e-6 EPOCHS=1
bash scripts/train_local.sh
rm -rf /tmp/qwen-smoke
```

```bash
export CKPT=$QWEN_DATA_ROOT/exp_11labs/runs/step-005482-base
export EXAM=$QWEN_DATA_ROOT/tasks/t1_eval/testset/partB.jsonl
export OUT=/tmp/t1-smoke
export LIMIT=20 SEEDS=1 MODES=xvec BATCH=1
bash scripts/score_local.sh
```
