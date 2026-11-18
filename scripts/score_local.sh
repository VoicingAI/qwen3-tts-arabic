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
GEN_PYTHON="${GEN_PYTHON:-python}"
SCORE_PYTHON="${SCORE_PYTHON:-python}"
args=(
  --ckpt "$CKPT"
  --exam "$EXAM"
  --out "$OUT"
  --limit "$LIMIT"
  --seeds "$SEEDS"
  --modes "$MODES"
  --batch-size "$BATCH"
)
# qwen-tts does not import on the score pin (transformers >= 5.4). Generation
# uses the train venv. Scoring uses the score venv. One python runs both phases.
if [[ "$GEN_PYTHON" == "$SCORE_PYTHON" ]]; then
  "$GEN_PYTHON" -m t1_local --phase all "${args[@]}"
else
  "$GEN_PYTHON" -m t1_local --phase generate "${args[@]}"
  "$SCORE_PYTHON" -m t1_local --phase score "${args[@]}"
fi
