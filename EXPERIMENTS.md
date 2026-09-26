# Experiments

One row per run. Metrics per `CLAUDE_v2.md`: blocking recall ceiling, avg candidates per S1,
validation macro F0.5 (overall + per country), wall-clock runtime. Validation = held-out 20% of
train S1 ids (seed 42), searched against ALL train S2/S3 so distractors stay in.

## Blocking

| date | config | recall @ k | avg cands/S1 | runtime | notes |
|------|--------|-----------|--------------|---------|-------|

## Features (B)

| date | config | pairs | throughput | 40M extrapolation | runtime | notes |
|------|--------|-------|-----------|-------------------|---------|-------|
| 2026-09-27 | `dev_candidates` 5,000 val S1, 20 hard negs | 117,287 (17,287 pos / 100,000 neg) | — | — | 69s | 99,919 overlap + 81 pool negs; 3.46 pos/S1 matches EDA |
| 2026-09-27 | `features.py` v1, 31 float32 feats | 117,287 | 22,209 pairs/s | ~30.0 min | 5.3s | peak RAM 1.3 GB; cpdist workers=-1 |
| 2026-09-27 | `dev_candidates --side fit` 20,000 S1 | 468,900 (68,900 pos / 400,000 neg) | — | — | 77s | trainable set; 0 S1 overlap with the val set |
| 2026-09-27 | `features.py` v2 (D review fixes) | 468,900 | 20,538 pairs/s | ~32.5 min | 22.8s | Arrow-backed records: 193 B/row vs 952 B/row for dicts |

Strongest label separation (dev set, hard negatives): `addr_norm_tfidf` +0.803, `addr_phon_tfidf`
+0.801, `num_conflict` -0.707, `num_agree` +0.653, `acronym_match` +0.599, `addr_norm_token_set`
+0.557. Near-dead: `name_missing` (0.000), `is_s3` (+0.001), `addr_blank` (+0.004),
`postal_conflict` (-0.005), `name_alt_token_set` (+0.018) — postal is 95% missing on this data.

Caveat: these negatives come from token overlap, not from blocking top-k. Real blocking
candidates are nearer neighbours, so expect every gap to compress on the true candidate set.
The dev set also keeps 100% of true matches at ~15% positive, against ~8-9% on real candidates,
so it must not be used to pick a threshold.

### v2 — changes from D's review (2026-09-27)

- **Record memory.** `load_records` returned a dict of per-row dicts. Measured at 952 B/row,
  which is ~8.9 GB at 10M distinct targets and would very likely OOM a 16 GB box once the
  40M-pair arrays were added. Now Arrow-backed at 193 B/row (~1.8 GB at 10M), with lookups via
  `pc.index_in` instead of a Python dict.
- **Address TF-IDF vocabulary.** Was fitted on a 200k sample of the dev candidates, i.e. train
  text only, leaving test-only French addresses with an all-zero vector on ~15% of the test set.
  Now sampled across every normalized parquet present; the cache records its sources and warns
  when new ones appear. Dev-set means are unchanged (these are all train rows) — the fix is
  only observable once the test parquets exist.
- **Retrieval + competition features** (14, added when the candidate file carries
  channel/score/rank): per-channel flags, channel count, best score/rank, rank fraction,
  candidate-list size, gap and ratio to the S1's best, and the cross-S1 contest for a shared
  target (reverse rank, contender count, score gap).
- **Duplicate-pair collapse.** blocking.py emits one row per (pair, channel), so a pair found by
  three channels appears three times; `collapse()` folds them (score = max, rank = min, one flag
  per channel) before any feature is computed.
- **`--side fit|val|both`** on dev_candidates, so a training set can be drawn from the 80% the
  model is actually fitted on.
- **`feature_columns(table)`** is now the train/predict column contract. "Numeric columns" is
  not a safe test: `label` is int8 and passes it.

## Model

| date | config | val F0.5 | US | India | France | threshold | runtime | notes |
|------|--------|----------|----|-------|--------|-----------|---------|-------|

## Submissions

| date | leaderboard F0.5 | local val F0.5 | avg cands/S1 | what changed |
|------|------------------|----------------|--------------|--------------|
