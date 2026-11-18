# First tasks

Do them in this order. Print the cost and stop before any GPU job that would pass the cap you were given. Never more than 10 GPUs. Do not train on the test set. Do not write into `step-045190-base`.

Costs below use the T5 probe rate of $4.84 per hour for one H100 with 8 CPU cores and 64 GiB, the H100 GPU price of $3.95 per hour, and about $8 for one full T1 checkpoint on L40S. They are estimates. Replace them with a measurement before you launch.

## 1. GPU utilisation

The Stage B step at batch 4 on one H100 ran at about 22% GPU and 3.80 steps/s after the first 40 steps, using 41 GB.

Change the loader or the batch until a 300-step probe, same bf16 and fp32 master setup, stays above 70% GPU utilisation. Report the new steps/s and the peak memory. Do not save a checkpoint.

Done when: utilisation mean after the first step is above 70%, and steps/s is written next to the old 3.80.

Cost: one 15-minute 1×H100 probe is about $1.20 at the $4.84 rate. A few of those while you change the loader is about $5. An 8×H100 hour, if you insist on the production shape, is 8 × $3.95 = $31.60. Start with one GPU.

## 2. T6b speaker inventory and a single-voice ceiling

Inventory is CPU. List speakers in the ElevenLabs set with hours per speaker. Pick two voices that each have at least 5 hours.

Train each voice on its own from step 45190 (read only). Score each with T1, x-vector and in-context, seeds 1–3.

Done when: the inventory is a table of speaker, dialect, hours, and both runs have a T1 `compare` line against s5482. The claim "a single voice is enough" is only allowed if that voice's Part A priority WER is at or below 0.144 without a higher broken rate.

Cost: inventory is $0 of GPU. Plan 14's stored clock for ~100 h and 8,223 steps was 1.06 h on 8×H100, which is about $33 at $3.95. A 5 h slice is a fraction of that data. Budget about $15 per voice, including one T1 score at $8, and do not launch if your own step estimate is higher. Two voices, about $30.

## 3. T6c scaling curve

From step 45190, three separate runs on 25 h, 50 h, and 100 h of the ElevenLabs set. Same recipe as Plan 14 (lr 2e-6, diac_prob 0.5, batch 4, 32 rows per step). Score each with T1.

Done when: one table of hours, steps, Part A priority WER, Part B x-vector WER, broken rate, dollars. 100 h should land near s5482 (0.144) if the recipe matches. 25 h and 50 h tell you whether the last half of the data is worth it.

Cost: scale the Plan 14 clock. 100 h about $33 of H100 time plus $8 to score. 50 h about half, 25 h about a quarter, if steps scale with rows. Three scores are $24. The whole curve is about $80 to $110. Write the three estimates, then launch one at a time.

## 4. Low-band and full-band tags

The YouTube audio used for Tier 1 is 16 kHz content in a 24 kHz file. ElevenLabs and a lot of Darbuka are wider. Add a field on each train row, `band=low` or `band=full`, set from the source, and teach the loader to read it. Do not drop either set. Do not change the loss until a later task says so.

Done when: `python -m train_dialects --check_only` on a tiny jsonl prints the count of each tag, and a row with no tag fails the check instead of training silently.

Cost: $0. No GPU.
