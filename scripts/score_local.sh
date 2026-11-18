#!/usr/bin/env bash
# T1 scoring on this machine. Does not call Modal.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f paths.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source paths.env
  set +a
fi
: "${QWEN_DATA_ROOT:?set QWEN_DATA_ROOT}"
: "${CKPT:?set CKPT}"
: "${EXAM:?set EXAM to a jsonl of exam rows}"
: "${OUT:?set OUT}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
LIMIT="${LIMIT:-0}"
SEEDS="${SEEDS:-1,2,3}"
MODES="${MODES:-xvec,icl}"
BATCH="${BATCH:-1}"
python -m t1_local \
  --ckpt "$CKPT" \
  --exam "$EXAM" \
  --out "$OUT" \
  --limit "$LIMIT" \
  --seeds "$SEEDS" \
  --modes "$MODES" \
  --batch-size "$BATCH"
