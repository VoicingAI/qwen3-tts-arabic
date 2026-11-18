#!/usr/bin/env bash
# One GPU (or whatever CUDA_VISIBLE_DEVICES shows). Does not call Modal.
# Source paths.env first, or export QWEN_DATA_ROOT and QWEN3_TTS_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f paths.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source paths.env
  set +a
fi
: "${QWEN_DATA_ROOT:?set QWEN_DATA_ROOT}"
if [[ -z "${QWEN3_TTS_DIR:-}" ]]; then
  if [[ ! -f /opt/Qwen3-TTS/finetuning/dataset.py ]]; then
    git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git /opt/Qwen3-TTS
  fi
  QWEN3_TTS_DIR=/opt/Qwen3-TTS
fi
export QWEN3_TTS_DIR
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

INIT="${INIT:?set INIT to a checkpoint directory}"
TRAIN_JSONL="${TRAIN_JSONL:?set TRAIN_JSONL}"
VAL_JSONL="${VAL_JSONL:-}"
OUT="${OUT:?set OUT}"
DIALECTS="${DIALECTS:-ar_msa,ar_egy,ar_sau,ar_irq,ar_tun,ar_mar,ar_omn,ar_jor,ar_syr,ar_sdn}"
LR="${LR:-2e-6}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
SAVE_EVERY="${SAVE_EVERY:-1000000}"
EXTRA_VAL="${EXTRA_VAL:-}"

args=(
  --init_model_path "$INIT"
  --dialects "$DIALECTS"
  --train_jsonl "$TRAIN_JSONL"
  --require_split train
  --output_dir "$OUT"
  --lr "$LR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH"
  --grad_accum "$ACCUM"
  --eval_every 1000000
  --save_every "$SAVE_EVERY"
  --save_at ""
  --resume 0
  --export_speakers 0
  --id_dropout 0.1
  --warmup_frac 0.03
  --sub_talker_weight 0.3
  --diac_prob 0.5
  --lang_lr_mult 20
  --fp32_master 1
  --num_workers 0
  --max_steps "$MAX_STEPS"
)
if [[ -n "$VAL_JSONL" ]]; then
  args+=(--val_jsonl "$VAL_JSONL")
fi
if [[ -n "$EXTRA_VAL" ]]; then
  args+=(--extra_val_jsonl "$EXTRA_VAL")
fi
python -m train_dialects "${args[@]}"
