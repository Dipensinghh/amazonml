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

Strongest label separation (dev set, hard negatives): `addr_norm_tfidf` +0.803, `addr_phon_tfidf`
+0.801, `num_conflict` -0.707, `num_agree` +0.653, `acronym_match` +0.599, `addr_norm_token_set`
+0.557. Near-dead: `name_missing` (0.000), `is_s3` (+0.001), `addr_blank` (+0.004),
`postal_conflict` (-0.005), `name_alt_token_set` (+0.018) — postal is 95% missing on this data.

Caveat: these negatives come from token overlap, not from blocking top-k. Real blocking
candidates are nearer neighbours, so expect every gap to compress on the true candidate set.

## Model

| date | config | val F0.5 | US | India | France | threshold | runtime | notes |
|------|--------|----------|----|-------|--------|-----------|---------|-------|

## Submissions

| date | leaderboard F0.5 | local val F0.5 | avg cands/S1 | what changed |
|------|------------------|----------------|--------------|--------------|
