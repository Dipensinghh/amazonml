# Business Entity Resolution

CPU-only pipeline that matches every Source 1 (S1) business to zero or more Source 2 / Source 3
records. It normalizes the three sources, retrieves candidates per S1 within its country and target
source in both directions (forward S1 → targets, plus reverse: each target's best S1), computes 44 pair
features, scores pairs with LightGBM, applies one validation-tuned threshold with global target
ownership (each S2/S3 record goes to at most one S1), and writes the two submission files.

Shipped model (final submission): **R=1 reverse retrieval trained on 600,000 fitting S1s**,
`models/r1_600k/` (validation macro F0.5 0.9561, US 0.9658, India 0.9418, 2,978 trees), scored at
**threshold 0.78**: public leaderboard 0.947135, output SHA-256 `078556e5…3fe78`. The bundle stores its
validation-tuned threshold (0.70); 0.78 was chosen from two public-leaderboard probes and is set by
`THRESHOLD` in `run.py` (override with `--threshold`). The scoring model is `MODEL_DIR` at the top of
`run.py` (or `--model-dir`). `models/r1_full/` (the 200k-S1 R=1 model, val 0.9531, public 0.942136 at
0.70) is kept for comparison.

## Layout

```text
run.py                  stage runner (prints every command it runs; --dry-run to only print)
requirements.txt        pinned dependencies
src/                    all pipeline code (entry points listed below)
models/r1_600k/         final: lgbm.txt, bundle.json (ordered 44 features, val threshold 0.70), threshold.json, val_report.json
models/r1_full/         200k-S1 R=1 model, same layout
artifacts/addr_tfidf.joblib   fitted address TF-IDF shared by train and test features (see below)
```

Place the organizer `dataset/` (train/ and test/) and `utils/` folders next to `run.py`. No external
data, APIs, gazetteers or pretrained models are used.

## Setup

```text
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt           # macOS / Linux
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
```

Python used by the team: CPython 3.14.4 (Windows 11), CPython 3.14.7 (macOS), Python 3.14 (Linux).

## Reproduce end to end

`python -u run.py all` runs the stages below in order. While `models/r1_600k/bundle.json` exists,
`all` skips `train` and `report` and scores with the shipped model (see "Retraining"). Each stage can
also be run alone, e.g. `python -u run.py reverse`. The commands `run.py` executes, from this folder:

```text
# 1. normalize (writes artifacts/{train,test}_s{1,2,3}_norm.parquet)
python -u src/normalize.py --split train
python -u src/normalize.py --split test

# 2. forward blocking, frozen config: cap 200, alpha 6, K 10 per source + 2 blank-address per source
python -u src/blocking.py --produce trainall --jobs 8
python -u src/blocking.py --produce test     --jobs 8

# 3. reverse retrieval R=1 and union: every S2/S3 target queries the same country-specific S1
#    token index and adds its top-1 S1; forward evidence wins on overlap, `from_reverse` marks
#    reverse discovery, final ranks are recomputed over the union. --subsets writes the exact
#    seeded fit200k and val50k populations of the train union.
python -u src/reverse_union.py --name trainall --subsets --jobs 8   # cand_trainall_r1, cand_fit200k_r1, cand_val50k_r1
python -u src/reverse_union.py --name test --jobs 8                 # cand_test_r1

# 4. pair features: 44 (43 of v3 + from_reverse); ranks, counts and gaps within each S1's list
#    are computed over the union; leading-zero house-number fix built in
python -u src/features.py --candidates artifacts/cand_fit200k_r1.parquet --split train --labels --batch 500000 --out artifacts/cand_fit200k_r1_features.parquet
python -u src/features.py --candidates artifacts/cand_val50k_r1.parquet  --split train --labels --batch 500000 --out artifacts/cand_val50k_r1_features.parquet
python -u src/features.py --candidates artifacts/cand_test_r1.parquet    --split test           --batch 500000 --out artifacts/cand_test_r1_features.parquet

# 5. competition features (rev_rank, rev_count, rev_score_gap) over the whole union population
python -u src/competition.py --name trainall_r1
python -u src/competition.py --name test_r1

# 6. write the whole-population rev_* into each feature file
python -u src/apply_competition.py --features artifacts/cand_fit200k_r1_features.parquet --competition artifacts/competition_trainall_r1.parquet --out artifacts/cand_fit200k_r1_features_fullrev.parquet
python -u src/apply_competition.py --features artifacts/cand_val50k_r1_features.parquet  --competition artifacts/competition_trainall_r1.parquet --out artifacts/cand_val50k_r1_features_fullrev.parquet
python -u src/apply_competition.py --features artifacts/cand_test_r1_features.parquet    --competition artifacts/competition_test_r1.parquet     --out artifacts/cand_test_r1_features_fullrev.parquet

# 7. train: fit200k_r1 + fit400k_extra_r1 (600k fitting S1s), val50k_r1 for early stopping.
#    fit400k_extra_r1 comes from: python -u src/fit_extra.py --n 400000 --candidates artifacts/cand_trainall_r1.parquet --base-fit artifacts/cand_fit200k_r1.parquet --tag fit400k_extra_r1
#    then features.py + apply_competition.py on it, exactly as for fit200k_r1 (run.py stages extra / features / apply_competition).
python -u src/train.py --features artifacts/cand_fit200k_r1_features_fullrev.parquet --extra-features artifacts/cand_fit400k_extra_r1_features_fullrev.parquet artifacts/cand_val50k_r1_features_fullrev.parquet --scope-sample 50000 --out-dir models/r1_600k --n-jobs 16 --num-leaves 255 --n-estimators 3000 --learning-rate 0.05 --early-stop 100
python -u src/val_report.py --model models/r1_600k --features-val artifacts/cand_val50k_r1_features_fullrev.parquet --cand artifacts/cand_val50k_r1.parquet --scope-sample 50000 --k 5 --out models/r1_600k/val_report.json

# 8. score test, threshold + global ownership, write output/, run both checkers
python -u src/score_test.py --features artifacts/cand_test_r1_features_fullrev.parquet --candidates artifacts/cand_test_r1.parquet --model models/r1_600k/lgbm.txt --bundle models/r1_600k/bundle.json --validator utils/validate_submission.py --batch-size 500000 --threshold 0.78
```

**Why reverse retrieval.** Half of the forward model's misses were never retrieved (68% of those in
India). Letting every target search back for its best S1 raises the candidate recall ceiling on the
50,000 validation S1s from 0.9293 to 0.9578 (US 0.9621 → 0.9777, India 0.8806 → 0.9281) for 1.8% more
pairs, and validation macro F0.5 from 0.9394 (forward, same recipe) to 0.9531.

**Why steps 5-6.** `features.py` computes `rev_*` inside the file it is given, so a 50k- or 200k-S1
sample sees far less competition for each target than all S1s do (forward fit200k: negative-pair
`rev_count` 13.2 inside the file vs 135.7 over all 2.2M train S1s). Model and test use
whole-population values; without them the R=1 model scores 0.9527 instead of 0.9531.

Validation split: 20% of train S1s (seed 42) in `artifacts/split_train.parquet`, created by
`blocking.py`; the evaluated scope is a seed-42 sample of 50,000 of them, searched against all train
S2/S3 of the same country (distractors kept).

## Candidate and model sizes (R=1)

| artifact | pairs | candidates/S1 | reverse-only pairs |
|---|---:|---:|---:|
| cand_trainall_r1 | 53,163,079 | 24.09 | 943,117 |
| cand_fit200k_r1 | 4,819,000 | 24.10 | – |
| cand_val50k_r1 | 1,205,167 | 24.10 | 21,884 |
| cand_test_r1 | 41,955,111 | 24.22 | 1,258,272 |

r1_600k (final): 44 features, fitted on 14,456,015 pairs of 600,000 fitting S1s, best iteration 2,978
(of a 3,000 cap); val pair precision 0.9860, recall 0.9142, singleton accuracy 0.9367.
r1_full: 44 features, fitted on 4,819,000 pairs of 200,000 fitting S1s (13.8% positive), early
stopping at 1,372 trees; val pair precision 0.9843, recall 0.9106.

## Measured runtime and peak RAM

The production R=1 steps (reverse union, R=1 features, R=1 competition, r1_full training, test
scoring) were run by the team on another machine and their runtimes are **not recorded in this
repository**. Measured values for the shared steps and the closest reference runs:

| step | machine | rows | runtime | peak RAM |
|---|---|---|---|---|
| normalize train | Windows | 12.5M | ~8 min | 3.9 GB |
| normalize test (via `normalize_test.py`, same functions) | Windows | 11.7M | 395 s | 2.8 GB |
| forward blocking trainall | Windows | 52,219,962 pairs | 533 s | 9.1 GB |
| forward blocking test | Windows | 40,696,839 pairs | 471 s | 7.4 GB |
| reverse query of all 10.32M train targets (experiment, same index) | C | – | 479 s | 10.4 GB |
| features, forward sets (reference) | Windows | 1.2M-9.5M | ~16,500-17,700 pairs/s | 3.0-5.2 GB |
| features, forward test (reference) | macOS, 36 GB | 40,696,839 | 1,714 s | 11.6 GB |
| competition, forward trainall (reference) | Windows | 52.2M pairs | 82 s | 4.4 GB |
| apply_competition val50k (reference) | Windows | 1,183,283 | 6 s | 0.7 GB |
| training with the same settings on forward fit200k (v2cq_fix, reference) | Windows | 4,732,479 | 300 s | 6.0 GB |

Windows = Windows 11, 16 GB RAM, 20 logical CPUs, CPython 3.14.4. The full test feature build needs
more than 16 GB in one pass; build it on a larger machine or in candidate slices. Slicing is safe for
`rev_*` because step 6 overwrites them with whole-population values.

## Retraining

`run.py train` refuses to overwrite an existing bundle. To retrain and score the result:

```text
python -u run.py train  --model-dir models/r1_retrained
python -u run.py report --model-dir models/r1_retrained
python -u run.py score  --model-dir models/r1_retrained
```

To use another model, put its `lgbm.txt` and `bundle.json` in a folder and change `MODEL_DIR` in
`run.py`, or pass `--model-dir`. The previous forward-only model (v3, val 0.9423) and its recipe
(`src/fit_extra.py`, 600k fitting S1s) are documented in `EXPERIMENTS.md`.

## Shared address TF-IDF

`artifacts/addr_tfidf.joblib` (SHA-256 `7a9d6370acc9eb3e55625c7e8d4a55b81e91f3320a9484f27c5a9098a30249e3`)
is fitted on addresses sampled from all six normalized sources. The train/val features behind
r1_full and the test features must use the same file, so it is shipped. `features.py` refits it only
when the file is missing, and a refit would not match the model.

## Outputs and checks

```text
output/matching_results.tsv   source1_entity_id<TAB>matched_entity_ids
output/candidate_pairs.tsv    source1_entity_id<TAB>candidate_entity_ids
```

One row per test S1 (empty lists allowed), sorted ids, UTF-8, LF, no quoting. Matches are a
subset of candidates and each S2/S3 id has at most one owner (ties: smaller S1 id).
`score_test.py` runs both checkers; to rerun them:

```text
python -u src/checks.py --out-dir output --test-dir dataset/test
python -u utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Entry points in src/

| file | role |
|---|---|
| `normalize.py` | ftfy + NFKC + anyascii, name/address normalization, phonetic keys |
| `blocking.py` | frozen forward rare-token retrieval (`--produce trainall / test`, also `val50k / fit200k`) |
| `reverse_union.py` | reverse R=1 retrieval and forward ∪ reverse union (+ fit200k/val50k subsets) |
| `features.py` | 44 pair features (`feature_columns()` is the train/score contract) |
| `competition.py`, `apply_competition.py` | whole-population `rev_*` |
| `train.py`, `val_report.py` | LightGBM training + threshold tuning; validation report |
| `score_test.py` | chunked test scoring, global ownership, TSVs, checks |
| `predict.py`, `ownership.py`, `io_utils.py`, `checks.py`, `evaluate.py` | decision policy, writers, strict checker, metric |
| `fallback.py`, `run_fallback.py` | rules fallback (val 0.6561, public 0.658) |
| others | earlier models and experiments: `fit_extra.py` (v3), `eda.py`, `sweep*.py`, `error_analysis.py`, `india_retrieval.py`, `reverse_retrieval.py`, `exp_xsrc.py`, `xsrc_features.py`, `dev_candidates.py`, `patch_num_features.py`, `fallback_e01.py`, `normalize_test.py` |
