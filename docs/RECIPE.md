# How step 5482 was trained

Read this in order. The checkpoint we ship as the current best model is `/data/exp_11labs/runs/step-005482-base` (tag `internal_11labs`). It was not trained from the public base in one go. Three runs were stacked.

Paths below are the Modal volume paths. On your machine, set `QWEN_DATA_ROOT` and the same relative path is used. Leave `QWEN_DATA_ROOT` unset when you launch Modal. The volume is always mounted at `/data` inside the container.

Speaker encoder is frozen in `train_dialects.py`. `freeze_code_predictor` is 0, so the code predictor trains. New dialect rows in the codec embedding use `--lang_lr_mult 20`. Everything else that is not the speaker encoder trains. Precision is bf16 autocast with fp32 master weights (`--fp32_master 1`). Optimizer is AdamW, weight decay 0.01 (the script default; the launchers do not override it), gradient clip 1.0, cosine schedule, warmup fraction 0.03. Attention is sdpa. Max frames 375 (30 s at 12.5 Hz).

`diac_prob 0.5` means each row uses `text_diac2` if that field is set, otherwise `text_diac`, on half the draws. The other half uses plain `text`. Rows with `no_diac` stay on plain `text`.

## Base

Repo id `Qwen/Qwen3-TTS-12Hz-1.7B-Base`. Local path used by Stage B1: `/data/models/Qwen3-TTS-12Hz-1.7B-Base`. Tokenizer `/data/models/Qwen3-TTS-Tokenizer-12Hz`. The Stage B1 `run.json` does not record a Hugging Face revision. Do not invent one.

The trainer imports `TTSDataset` from a checkout of [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) `finetuning/dataset.py`. That file is not in this repo. `modal_app.py` clones it to `/opt/Qwen3-TTS`. Locally, set `QWEN3_TTS_DIR` to that checkout.

## Stage A3

Start weights `/data/runs/stageA/step-016000-base`. One epoch. End checkpoint `/data/runs/stageA3/step-020408-base`.

| | |
| --- | --- |
| Train file | `/data/tts_stageA/stageA3/train_codes.jsonl` |
| Val | `/data/tts_stageA/stageA3/a2_val_codes.jsonl` |
| Extra val | `/data/tts_stageA/stageA3/uae100k_val_codes.jsonl` |
| Hours | 653,064 clips, 1,517.21 h in `runs/stageA3/report.md`. Per-dialect hours for this file are not in that report |
| Text | `diac_prob` 0.5 |
| lr | 5e-6, cosine, warmup 0.03 |
| Batch | 4 per GPU, grad accum 1, 8×H100, 1 epoch |
| Steps | ends at 20,408 |
| Region | `train_stagea3` sets `region=["us", "eu"]` |
| Wall time | not stored. There is no `run.json` or `spend_train.json` in `runs/stageA3/` |
| Cost | not stored |

Modal:

```bash
modal run --detach stageA3_app.py::train_stagea3
```

Local, one GPU, same flags except the GPU count (set `BATCH` and `QWEN_DATA_ROOT` first):

```bash
INIT=$QWEN_DATA_ROOT/runs/stageA/step-016000-base \
TRAIN_JSONL=$QWEN_DATA_ROOT/tts_stageA/stageA3/train_codes.jsonl \
VAL_JSONL=$QWEN_DATA_ROOT/tts_stageA/stageA3/a2_val_codes.jsonl \
EXTRA_VAL=$QWEN_DATA_ROOT/tts_stageA/stageA3/uae100k_val_codes.jsonl \
OUT=$QWEN_DATA_ROOT/runs/stageA3 \
LR=5e-6 EPOCHS=1 BATCH=4 ACCUM=1 MAX_STEPS=0 \
DIALECTS=ar_msa,ar_egy,ar_sau,ar_uae,ar_gulf,ar_omn,ar_lev,ar_irq,ar_mar,ar_tun,ar_dza \
bash scripts/train_local.sh
```

`train_local.sh` forces `--resume 0` and a large `--save_every`. The historical Modal command uses `--resume 1` and `--save_every 2000`. For a real Stage A3 resume, call `python -m train_dialects` with those two flags instead of the smoke defaults in the shell script.

## Stage B1 (step 45190)

From the base model, not from step 20408. Job name in `/data/runs/stageB1/run.json` is `track5`. Launcher `plan13_train.py`. Checkpoint `/data/runs/stageB1/step-045190-base`. That directory is read only. Do not write into it.

| | |
| --- | --- |
| Train file | `/data/plan13/tier1_v1.jsonl` |
| Dev | `/data/plan13/tier1_dev.jsonl` |
| Extra vals | Stage A2 val and UAE100k val, same paths as Stage A3 |
| Hours | Tier 1 1,504.54 h. Per dialect in `docs/DATA.md` |
| Text | `diac_prob` 0.5 |
| lr | 5e-6, cosine, warmup 0.03 |
| Batch | 4 per GPU, grad accum 1, 8×H100, 3 epochs |
| Region | null in `run.json`. No region pin |
| Wall clock | `spend_train.json` `total_s` 19554.8 seconds (5.43 h). No dollar field |
| Cost | not stored. 8 GPUs × 5.43 h × $3.95 (`H100_RATE` in `plan13_train.py`) is $171.65. That is arithmetic on the stored clock, not an invoice |

Stop rule in `plan13_train.py`: Cohere word error on the scored checkpoints is compared with Stage A3 at the paired step. The stop uses that word error, not the dev loss. The dev loss is logged beside it. Pause if projected spend reaches $396. Approved cap in the file is $330.

Modal:

```bash
modal run --detach plan13_train.py::go
```

`go` spawns `run`, which is the Track 5 orchestrator (train plus the scorer). It will not start if you only wanted the inner `train_dialects.py` line. The inner command, from `train_remote`, is:

```bash
accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no train_dialects.py \
  --init_model_path /data/models/Qwen3-TTS-12Hz-1.7B-Base \
  --dialects ar_msa,ar_egy,ar_sau,ar_uae,ar_gulf,ar_omn,ar_lev,ar_irq,ar_mar,ar_tun,ar_dza,unknown \
  --train_jsonl /data/plan13/tier1_v1.jsonl \
  --val_jsonl /data/plan13/tier1_dev.jsonl \
  --extra_val_jsonl /data/tts_stageA/stageA3/a2_val_codes.jsonl \
  --extra_val_jsonl2 /data/tts_stageA/stageA3/uae100k_val_codes.jsonl \
  --require_split train --output_dir /data/runs/stageB1 \
  --lr 5e-6 --epochs 3 --batch_size 4 --grad_accum 1 \
  --eval_every 500 --save_every <from the orchestrator> --save_at <from the orchestrator> \
  --resume 1 --export_speakers 0 --id_dropout 0.1 --warmup_frac 0.03 \
  --sub_talker_weight 0.3 --diac_prob 0.5 --lang_lr_mult 20 --fp32_master 1
```

Do not resume this run.

## Plan 14 (step 5482)

Init `/data/runs/stageB1/step-045190-base`, read only. Train `/data/exp_11labs/train_codes.jsonl`. Val `/data/exp_11labs/val_codes.jsonl`. Tag written into each checkpoint: `internal_11labs`. Launcher `plan14_train.py`. End checkpoint `/data/exp_11labs/runs/step-005482-base`.

| | |
| --- | --- |
| Hours | `exp_11labs/manifest/counts.json`: train 99.04 h (87,717 clips), val 1.18 h (1,024). Together 100.22 h, 88,741 clips |
| Per dialect | Measured on the whole set for T6a: ar_egy 10.009, ar_irq 10.025, ar_jor 10.021, ar_mar 10.015, ar_msa 10.035, ar_omn 10.032, ar_sau 10.018, ar_sdn 10.031, ar_syr 10.020, ar_tun 10.008. `counts.json` itself has no per-dialect split |
| Dialects | `ar_msa,ar_egy,ar_sau,ar_irq,ar_tun,ar_mar,ar_omn,ar_jor,ar_syr,ar_sdn` |
| Text | `diac_prob` 0.5 |
| lr | 2e-6, cosine, warmup 0.03 |
| Batch | 4 per GPU, grad accum 1, 8×H100, so 32 rows per step. 3 epochs |
| Saves | every 1,370 steps, plus 2741, 4000, 5482, 8223. Three passes are 8,223 steps (`counts.json`) |
| Region | null |
| Encode | `encode.json` `gpu_dollars` 4.0364 |
| Wall clock | `runs/spend_train.json` `total_s` 3804.2 seconds (1.06 h). No dollar field |
| Cost | not stored. 8 × 1.06 h × $3.95 is $33.39, arithmetic on the stored clock, not an invoice. `counts.json` `train_dollars` 13.05 is an estimate written before launch. Do not use it as the bill |

Stop rule: if both step 1370 and step 2740 are more than 10% worse than step 45190 on priority WER, stop. Checkpoints exist through step 8223, so that rule did not end the run. Cap in the file is $60, pause at $72.

Modal, 8×H100 (defaults of `run`):

```bash
modal run --detach plan14_train.py::run
```

`run` checks that batch × accum × GPUs is 32, then spawns `train_remote`. The inner command is:

```bash
accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no train_dialects.py \
  --init_model_path /data/runs/stageB1/step-045190-base \
  --dialects ar_msa,ar_egy,ar_sau,ar_irq,ar_tun,ar_mar,ar_omn,ar_jor,ar_syr,ar_sdn \
  --train_jsonl /data/exp_11labs/train_codes.jsonl \
  --val_jsonl /data/exp_11labs/val_codes.jsonl \
  --require_split train --output_dir /data/exp_11labs/runs \
  --lr 2e-6 --epochs 3 --batch_size 4 --grad_accum 1 \
  --eval_every 500 --save_every 1370 --save_at 2741,4000,5482,8223 \
  --resume 1 --export_speakers 0 --id_dropout 0.1 --warmup_frac 0.03 \
  --sub_talker_weight 0.3 --diac_prob 0.5 --lang_lr_mult 20 --fp32_master 1
```

Local equivalent: set `INIT` to the 45190 checkpoint, `LR=2e-6`, `EPOCHS=3`, `BATCH=4`, and call `python -m train_dialects` with `--resume 1` and `--save_every 1370`. `scripts/train_local.sh` is the one-GPU form and defaults to resume 0 so a smoke test cannot overwrite a real run.

There is no separate `launch` entry point for 8×H100 besides `run`. `launch4`, `launch2`, `launch_a100`, and `launch_l40s` are the other shapes. They keep 32 rows per step by raising grad accumulation when the GPU count drops.

## Known issues

- One H100 at batch 4, the real Stage B step, used about 22% of the GPU and 41 GB. Steady rate after step 40 was 3.80 steps/s. The card is waiting on the loader, not on math. See `docs/FIRST_TASKS.md`.
- Stage B1 Part B, x-vector only: Emirati "too long" is 0.325 and Kuwaiti "too long" is 0.275. Those match the broken rates. Step 5482 on the same rows is 0.000 too long. Source: the T1 compare.
- UTMOS is reported and is not a reason to pick a checkpoint. On the T6a codec check, ElevenLabs resynthesis scored higher than the original (drop −0.14). Do not tune on it.
