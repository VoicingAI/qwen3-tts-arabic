# Score a checkpoint with T1

T1 is a frozen exam. Quality numbers use clips that are not broken. Broken rates count every clip and are split into silent, cut off, too long, and repeating.

## What the two parts are

Part A is held-out Tier 1 material. Generation is x-vector only (a speaker vector from the reference clip, no transcript of the reference). The frozen file is `tasks/t1_eval/testset/partA.jsonl`. SHA-1 `b139e7c1214c562ac93407c537dab48146d0b3d7` (`partA.sha1`). `t1_eval.py` refuses to run if the file hash changes.

Part B is 270 prompted lines (40 each of Emirati, Kuwaiti, Najdi, Hijazi, Egyptian, Levantine, and 30 MSA). The reference is a short prompt clip plus its transcript. Generation is both x-vector only and in-context (the prompt transcript is shown to the model). SHA-1 of the current `partB.jsonl` is `9d648341bfafce6bef5917d0b81a56745c22a507`.

Kuwaiti and Najdi are one speaker each. The Kuwaiti prompt transcript has CER 0.10, so the in-context column can read a bit low.

Seeds are 1, 2, and 3. The same seeds are used for every checkpoint.

## Metrics

- WER: Cohere word error after the Plan 12 text normaliser. On Part B Gulf (Emirati, Kuwaiti, Najdi, Hijazi) and Levantine, the reported WER is the mean of Cohere and Jais. Other dialects are Cohere only.
- Letter: character error from `plan12_letter.py`.
- Dropped endings / dropped words: fraction of reference words lost at the end, or lost anywhere, on the alignment.
- Same speaker: cosine of WavLM `microsoft/wavlm-base-plus-sv` between the reference and the hypothesis. Different speaker is a different prompt from the same part, so the score should be lower.
- UTMOS: UTMOSv2. Reported only. Do not pick a winner with it.
- Broken: the clip is silent, cut off, longer than twice the expected duration, or repeating. A broken clip is left out of the quality means and counted in the broken rate.

Priority WER on Part A is the unweighted mean of MSA, Egyptian, Saudi, and Emirati. Lower is better. A difference is "no real difference" when the three-seed ranges overlap.

Cohere weights must be the commit `c3e911b42149bf7a1e53d5cef9878aee87515a23`. Jais must be revision `1323995fa189c479369a3929be0e95039eba5cba`. `t1_eval.py` checks `model_sha.txt` and `models/Jais-ASR/revision.txt`.

## One command

Modal, full exam, up to 10 GPUs. This call uses 2 shards. Raise `--shards` only if you stay at or under 10 GPUs in total, including anything else already running.

```bash
unset QWEN_DATA_ROOT
modal run t1_eval.py::main --ckpt /data/exp_11labs/runs/step-005482-base --name my-run --part both --shards 2
```

`--ckpt real` scores the Part A reference recordings and does not synthesise. Part B has no "real" column in that command.

Local, same metrics, your GPU:

```bash
export QWEN_DATA_ROOT=/path/to/data
export CKPT=$QWEN_DATA_ROOT/exp_11labs/runs/step-005482-base
export EXAM=$QWEN_DATA_ROOT/tasks/t1_eval/testset/partB.jsonl
export OUT=$QWEN_DATA_ROOT/tasks/t1_eval/results/my-run
export SEEDS=1,2,3
export MODES=xvec,icl
bash scripts/score_local.sh
```

`score_local.sh` calls `python -m t1_local`. Results land in `$OUT/run/` (`manifest_0.jsonl`, `audio_0.tar`, `score_0.jsonl`).

A full Modal checkpoint is about $8 of L40S time at $1.95 per GPU-hour. That is the planning figure. `modal run t1_eval.py::timing` writes a fresh estimate to `tasks/t1_eval/timing_estimate.md` from a 50-clip probe. Print that number before you launch a full score. Do not start the full score if the estimate plus what you have already spent breaks the cap you were given.

Modal results are written to `/data/tasks/t1_eval/results/<name>/` and the comparison prose to `/data/tasks/t1_eval/compare.md`.

## How to read compare.md

The first lines name the best checkpoint on Part A priority WER and the best checkpoint per dialect. Then each dialect has a table. A cell marked `no real difference` overlaps step 45190 across the three seeds. Read Part B x-vector and in-context as separate tables. Step 5482 wins Part B as x-vector only, not as in-context.

## Baseline

From the frozen T1 compare. Part A priority WER:

| checkpoint | Part A priority WER | Part B x-vector | Part B in-context |
| --- | ---: | ---: | ---: |
| s45190 (step 45190) | 0.221 (mean of the four priority dialects below) | 0.262 (0.259–0.267) | 0.268 (0.264–0.274) |
| s5482_11labs | 0.144 (0.141–0.146) | 0.076 | 0.184 |
| s1370_11labs | 0.149 (mean of MSA 0.075, Egyptian 0.184, Saudi 0.152, Emirati 0.185) | 0.079 | 0.184 |
| real (Part A recordings) | 0.205 |  |  |

Part A WER by dialect, seed range in parentheses:

| dialect | s45190 | s5482 | s1370 | real |
| --- | ---: | ---: | ---: | ---: |
| MSA | 0.110 (0.101–0.115) | 0.068 (0.064–0.070) | 0.075 (0.073–0.076) | 0.101 |
| Egyptian | 0.212 (0.209–0.213) | 0.179 (0.173–0.185) | 0.184 (0.181–0.186) | 0.213 |
| Saudi | 0.279 (0.274–0.287) | 0.153 (0.146–0.161) | 0.152 (0.149–0.157) | 0.240 |
| Emirati | 0.281 (0.275–0.286) | 0.176 (0.174–0.178) | 0.185 (0.180–0.188) | 0.264 |
| Iraqi | 0.262 (0.254–0.267) | 0.157 (0.146–0.165) | 0.165 (0.159–0.170) | 0.286 |
| Algerian | 0.371 (0.366–0.378) | 0.250 (0.237–0.263) |  |  |
| Moroccan |  | 0.309 (0.299–0.318) |  |  |

s5482 is the best Part A priority number. A new run is better only when its priority WER is lower than 0.144 and the seed range does not overlap 0.141–0.146, and the broken rate does not rise. Voice similarity should not drop by more than 0.01. UTMOS does not count.

Part B x-vector WER for s5482: Emirati 0.115, Kuwaiti 0.078, Najdi is better at s1370 (0.028), Hijazi 0.081, Egyptian is better at s1370 (0.075), Levantine 0.140, MSA 0.007.

Stage B1 Part B x-vector is the broken one: Emirati too-long 0.325, Kuwaiti too-long 0.275. s5482 on those rows is 0.
