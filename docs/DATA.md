# Data you will touch

`QWEN_DATA_ROOT` replaces the `/data` prefix. On Modal the prefix is `/data` and the volume name is `qwen3-arabic-tts-v2`.

## Training

| set | path | what it is | rate |
| --- | --- | --- | --- |
| Tier 1 manifest | `/data/plan13/tier1_v1.jsonl` | Stage B1 train file. 1,504.54 h | Codes point at the 24 kHz clean set. The raw YouTube audio on disk is 16 kHz and was upsampled; it has nothing above about 8 kHz |
| Tier 1 hours | see table | Our channels plus Darbuka. `unknown` is Darbuka with no dialect label. It is not folded into MSA | |
| Stage A3 codes | `/data/tts_stageA/stageA3/train_codes.jsonl` | 653,064 clips, 1,517.21 h | codec tokens, not wavs |
| Stage A2 val | `/data/tts_stageA/stageA3/a2_val_codes.jsonl` | 4,208 clips, 10.17 h | |
| UAE100k val | `/data/tts_stageA/stageA3/uae100k_val_codes.jsonl` | 300 clips used, 0.55 h. 9,335 other kept UAE val clips were not used | |
| UAE100k audio | under the Stage A3 tree and the YouTube clean folders | full-band relative to the YouTube 16 kHz dump; T6a measured UAE originals around 11 kHz | |
| ElevenLabs | `/data/exp_11labs/train_codes.jsonl`, `val_codes.jsonl`, audio under `/data/exp_11labs/audio` | 99.04 h train, 1.18 h val, 88,741 clips, tag `internal_11labs` | Generated audio, not YouTube. T6a: most of a 50-clip draw stays at or above 11 kHz through the codec |
| Darbuka | `/data/retx/plan12/darbuka_parts` and the Tier 1 rows whose dialect is `unknown` or a mapped dialect | Tier 1 Darbuka is 842.20 h of the 1,504.54 h. All Darbuka in the tier report is 1,940.58 h | CC-BY segments, not full videos. Many files are 48 kHz |
| YouTube clean | `/data/yt/yt_clean_24k/<folder>/` | what training reads | stored as 24 kHz, content is 16 kHz upsampled |
| YouTube raw | `/data/yt/raw/<folder>/manifest_single.transcribed.jsonl` | manifests | 16 kHz wavs |

Tier 1 hours (the Stage B1 set):

| dialect | hours |
| --- | ---: |
| unknown | 489.98 |
| ar_egy | 337.04 |
| ar_sau | 237.92 |
| ar_msa | 214.67 |
| ar_gulf | 104.00 |
| ar_uae | 69.38 |
| ar_omn | 16.01 |
| ar_lev | 12.84 |
| ar_irq | 12.43 |
| ar_dza | 3.91 |
| ar_tun | 3.55 |
| ar_mar | 2.80 |
| all | 1504.54 |

ElevenLabs hours (whole set, T6a, about 10 h each):

| dialect | hours |
| --- | ---: |
| ar_egy | 10.009 |
| ar_irq | 10.025 |
| ar_jor | 10.021 |
| ar_mar | 10.015 |
| ar_msa | 10.035 |
| ar_omn | 10.032 |
| ar_sau | 10.018 |
| ar_sdn | 10.031 |
| ar_syr | 10.020 |
| ar_tun | 10.008 |

B2, read-only key, bucket `companyNames-dataset`:

- ElevenLabs audio: `11labs_gen_audios/arabic_dialects_voxcpm/audio`
- YouTube: `YT_datadownload/arabic/<dialect>/segments_single/`

The checkpoints and the frozen test set are not in that B2 note. They are on the Modal volume. See `docs/LOCAL_SETUP.md`.

## Test

| file | path |
| --- | --- |
| Part A | `/data/tasks/t1_eval/testset/partA.jsonl` SHA-1 `b139e7c1214c562ac93407c537dab48146d0b3d7` |
| Part B | `/data/tasks/t1_eval/testset/partB.jsonl` SHA-1 `9d648341bfafce6bef5917d0b81a56745c22a507` |
| Prompts | `/data/tasks/t1_eval/testset/prompts/` |

## Rules

- Never train on the test set. Part A and Part B stay out of every train jsonl.
- ElevenLabs rows stay tagged `internal_11labs`. Do not mix them into a checkpoint that is described as Stage B1.
- `/data/runs/stageB1/step-045190-base` is read only.
- Never more than 10 GPUs at once, counting jobs that are already up.
- Print the expected cost before any GPU job, and stop if it would pass the cap you were given.
- Do not call ElevenLabs. Do not use the Modal secret named `elevenlabs`. The secrets this code references by name are `huggingface` and `qwen3-b2`.
