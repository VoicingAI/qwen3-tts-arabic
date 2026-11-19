# Smoke test

Run on 2026-11-18. Machine: NVIDIA GeForce RTX 4090, 24564 MiB, driver 580.178.04. Python 3.11. Torch 2.6.0+cu124. No Modal GPU was started. Spend is $0.

The train step used commit `6b7e375` with `torchaudio==2.6.0+cu124` already installed. That pin is committed in `22c65e2`. The score step used `22c65e2`.

`dataset.py` is not in the `qwen-tts` wheel. The train script imported it from a depth-1 checkout of https://github.com/QwenLM/Qwen3-TTS (`finetuning/dataset.py`, class `TTSDataset`).

Sizes copied for this run, from Modal volume `qwen3-arabic-tts-v2`, not from B2:

| path | size |
| --- | ---: |
| `exp_11labs/runs/step-005482-base` | 4.3 GB |
| `exp_11labs/train_codes.jsonl` | 422 MB |
| `retx/plan12/partD/weights` (Cohere) | 3.9 GB |
| `models/Jais-ASR` | 11 GB |

The smoke train file was 64 rows cut from that jsonl. Reference wavs for those rows are 12 MB. The score used the first 20 lines of `partB.jsonl` (all Emirati), seed 1, x-vector only, and the two prompt flacs those lines name.

## Train

```bash
export QWEN_DATA_ROOT=/tmp/qwen_smoke_data
export QWEN3_TTS_DIR=/tmp/Qwen3-TTS
export INIT=$QWEN_DATA_ROOT/exp_11labs/runs/step-005482-base
export TRAIN_JSONL=$QWEN_DATA_ROOT/exp_11labs/smoke_train.jsonl
export OUT=/tmp/qwen-smoke
export BATCH=1 ACCUM=1 MAX_STEPS=50 EPOCHS=1 LR=2e-6 SAVE_EVERY=1000000
bash scripts/train_local.sh
```

Exit 1. The 4090 ran out of memory on the first AdamW step, while allocating the fp32 moment buffer. Peak reported by PyTorch: 22.43 GiB in use of 23.51 GiB, 70 MiB free, the failed allocation was 16 MiB. `/tmp/qwen-smoke` held a TensorBoard directory only. It was deleted. No checkpoint was kept.

The `sox` lines are the Python `sox` package noticing that the `sox` binary is not installed. The loader still read the wavs with librosa. The failure is the out-of-memory error below.

```
COMMAND 2026-11-18T10:36:43Z
NVIDIA GeForce RTX 4090, 580.178.04, 24564 MiB
Warning: flash-attn is not installed. Will only run the manual PyTorch version.
/bin/sh: 1: sox: not found
SoX could not be found!
`torch_dtype` is deprecated! Use `dtype` instead!
codec vocab=3072 dim=2048
base language ids: {'chinese': 2055, 'english': 2050, 'german': 2053, 'italian': 2070, 'portuguese': 2071, 'spanish': 2054, 'japanese': 2058, 'korean': 2064, 'french': 2061, 'russian': 2069, 'ar_msa': 2072, 'ar_egy': 2073, 'ar_sau': 2074, 'ar_uae': 2075, 'ar_gulf': 2076, 'ar_omn': 2077, 'ar_lev': 2078, 'ar_irq': 2079, 'ar_mar': 2080, 'ar_tun': 2081, 'ar_dza': 2082, 'unknown': 2083, 'ar_jor': 2084, 'ar_syr': 2085, 'ar_sdn': 2086}
dialect ids: {'ar_msa': 2072, 'ar_egy': 2073, 'ar_sau': 2074, 'ar_irq': 2079, 'ar_tun': 2081, 'ar_mar': 2080, 'ar_omn': 2077, 'ar_jor': 2084, 'ar_syr': 2085, 'ar_sdn': 2086}
special: think=2154 think_bos=2156 think_eos=2157 nothink=2155 pad=2148 bos=2149 eos=2150
has text_projection: True
train rows 64 (0.1 h) loaded in 0s | rows trained as auto (unknown dialect): 0
GPUs 1 | per-GPU batch 1 x accum 1 = 1 clips/step | steps/epoch 64 | total 64 | save every 1000000 | save at []
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 70.38 MiB is free. Including non-PyTorch memory, this process has 22.43 GiB memory in use. Of the allocated memory 21.47 GiB is allocated by PyTorch, and 508.23 MiB is reserved by PyTorch but unallocated.
TRAIN_EXIT 1
```

A bare `pip install` of qwen-tts had pulled `torchaudio==2.11.0`, which needs `libcudart.so.13` and cannot load next to torch 2.6 cu124. `requirements-train.txt` now pins `torchaudio==2.6.0`. The run above used that pin (`2.6.0+cu124`).

## Score

```bash
export QWEN_DATA_ROOT=/tmp/qwen_smoke_data
export CKPT=$QWEN_DATA_ROOT/exp_11labs/runs/step-005482-base
export EXAM=$QWEN_DATA_ROOT/tasks/t1_eval/testset/partB.jsonl
export OUT=/tmp/t1-smoke
export LIMIT=20 SEEDS=1 MODES=xvec BATCH=1
export GEN_PYTHON=$PWD/.venv-train/bin/python
export SCORE_PYTHON=$PWD/.venv-score/bin/python
bash scripts/score_local.sh
```

Generation finished. nvidia-smi every 2 seconds peaked at 12583 MiB during the score that followed.

```
{"tag": "run", "shard": 0, "clips": 20, "load_s": 4.5, "prompt_s": 1.7, "copy_errors": 0}
{"tag": "run", "shard": 0, "seed": 1, "made": 20, "seed_s": 48.2, "stopped": ""}
{"tag": "run", "shard": 0, "done": true, "n": 20, "seeds_done": [1], "dollars": 0.0295, "fallbacks": 0, "stopped": ""}
{"generate": {"done": true, "n": 20, "errors": 0, "wall_s": 54.5, "load_s": 4.5}}
```

`dollars` here is `wall_s / 3600 * 1.95` inside `noise_generate`. It is not a charge. Nothing was billed.

The first score process then loaded Cohere and Jais and died while UTMOSv2's `wget` followed a redirect to `us.aws.cdn.hf.co` (`Temporary failure in name resolution`). `score_0.jsonl` was not written. The weight was fetched with `huggingface_hub` (`sarulab-speech/UTMOSv2`, `fold0_s42_best_model.pth`) and the score phase was run again on the same 20 wavs:

```
{"n": 20, "n_usable": 20, "wer_mean": 0.08045634920634921, "wall_s": 41.5, "score_path": "/tmp/t1-smoke/run/score_0.jsonl"}
```

Those 20 rows are the first lines of Part B, all Emirati, seed 1, x-vector only. `wer_mean` is the Cohere mean. Jais ran on all 20 (Emirati is Gulf). Mean Jais WER 0.052. Mean of the two, which is the T1 Gulf rule, is 0.066. Mean WavLM cosine 0.856. Mean UTMOSv2 2.94. Broken 0, errors 0. This is not the full-exam Part B number.

Weight-loading progress bars are omitted above. They are redraws of the same line.

## LoRA on the 4090

Full fine-tune at batch 1 does not fit, above. The same card does fit LoRA. 50 steps, batch 2, from `step-005482-base`, rank 64, alpha 64, learning rate 1e-4, base bf16, LoRA weights fp32, gradient checkpointing on. `/tmp/qwen-smoke` held a TensorBoard file only. It was deleted. No checkpoint was kept. Spend $0.

```bash
export QWEN_DATA_ROOT=/tmp/qwen_smoke_data
export INIT=$QWEN_DATA_ROOT/exp_11labs/runs/step-005482-base
export TRAIN_JSONL=$QWEN_DATA_ROOT/exp_11labs/smoke_train.jsonl
export OUT=/tmp/qwen-smoke
export LORA=1 BATCH=2 ACCUM=1 MAX_STEPS=50 EPOCHS=2 LR=1e-4 SAVE_EVERY=1000000
bash scripts/train_local.sh
```

Exit 0. 231 LoRA modules, 76,939,264 trainable parameters, 33 decoder layers checkpointed. PyTorch peak allocated 7001 MiB. nvidia-smi, sampled every 2 seconds, peaked at 9650 MiB. 50 steps in 21.4 s, 0.43 s/step (2.34 steps/s, the average from the start of the loop).

Loss went down. The four printed steps, total loss then main loss:

| step | loss | main |
| --- | ---: | ---: |
| 1 | 1.8334 | 0.5875 |
| 20 | 2.4764 | 1.1473 |
| 40 | 1.5895 | 0.3296 |
| 50 | 1.3850 | 0.1688 |

Step 20 is a longer bucket than step 1, so that one point is higher. Step 50 is below step 1. Gradient norm stayed above 1, so the LoRA weights received a gradient.

```
lora modules 231 rank 64 alpha 64.0 trainable 76,939,264 | checkpointed decoder layers 33 | base bf16, lora fp32, lr 0.0001
GPUs 1 | per-GPU batch 2 x accum 1 = 2 clips/step | steps/epoch 32 | total 64 | save every 1000000 | save at []
ep 0 step 1/64 loss 1.8334 main 0.5875 sub 4.1532 lr 1.00e-04 | 0.105 steps/s | 0 audio-h/h | ETA 0.2 h diac 0.00 grad 3.46
ep 0 step 20/64 loss 2.4764 main 1.1473 sub 4.4302 lr 7.92e-05 | 1.332 steps/s | 9 audio-h/h | ETA 0.0 h diac 0.00 grad 3.10
ep 1 step 40/64 loss 1.5895 main 0.3296 sub 4.1999 lr 3.17e-05 | 2.030 steps/s | 15 audio-h/h | ETA 0.0 h diac 0.00 grad 1.25
ep 1 step 50/64 loss 1.3850 main 0.1688 sub 4.0541 lr 1.17e-05 | 2.342 steps/s | 16 audio-h/h | ETA 0.0 h diac 0.00 grad 1.11
peak allocated 7001 MiB
training finished steps 50 wall 21.4s
TRAIN_EXIT 0
```
