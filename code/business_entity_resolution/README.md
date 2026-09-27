# Business Entity Resolution

CPU-only pipeline that matches every Source 1 (S1) business to zero or more Source 2 / Source 3
records. It normalizes the three sources, retrieves a bounded candidate list per S1 within its
country and target source, computes 43 pair features, scores pairs with LightGBM, applies one
validation-tuned threshold with global target ownership (each S2/S3 record goes to at most one
S1), and writes the two submission files.

Shipped model: **v3** in `models/v3/` (validation macro F0.5 0.9423, threshold 0.66). The
scoring model is one config value, `MODEL_DIR` at the top of `run.py` (or `--model-dir`).

## Layout

```text
run.py                  stage runner (prints every command it runs; --dry-run to only print)
requirements.txt        pinned dependencies
src/                    all pipeline code (entry points listed below)
models/v3/              lgbm.txt, bundle.json (ordered features, threshold), threshold.json, val_report.json
artifacts/addr_tfidf.joblib   fitted address TF-IDF shared by train and test features (see below)
```

Place the organizer `dataset/` (train/ and test/) and `utils/` folders next to `run.py`. No
external data, APIs, gazetteers or pretrained models are used.

## Setup

```text
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt           # macOS / Linux
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
```

Python used: CPython 3.14.4 on Windows 11 (all runtimes below unless marked), CPython 3.14.7
on macOS, Python 3.14 on Linux. The v3 run on Windows had joblib 1.5.3 installed; the pin is
1.6.0; both loaded the same `addr_tfidf.joblib`.

## Reproduce end to end

`python -u run.py all` runs the stages below in order (training is skipped while
`models/v3/bundle.json` exists; see "Retraining"). Each stage can also be run alone, e.g.
`python -u run.py features`. The commands `run.py` executes, from this folder:

```text
# 1. normalize (writes artifacts/{train,test}_s{1,2,3}_norm.parquet)
python -u src/normalize.py --split train
python -u src/normalize.py --split test

# 2. blocking, frozen config: cap 200, alpha 6, K 10 per source + 2 blank-address per source
python -u src/blocking.py --produce val50k   --jobs 8
python -u src/blocking.py --produce fit200k  --jobs 8
python -u src/blocking.py --produce trainall --jobs 8
python -u src/blocking.py --produce test     --jobs 8

# 3. 400k more fitting S1s (seed 42, disjoint from fit200k and validation), rows from cand_trainall
python -u src/fit_extra.py --n 400000

# 4. pair features (43; features.py includes the leading-zero house-number fix, no patch step)
python -u src/features.py --candidates artifacts/cand_fit200k.parquet       --split train --labels --batch 500000 --out artifacts/cand_fit200k_features.parquet
python -u src/features.py --candidates artifacts/cand_fit400k_extra.parquet --split train --labels --batch 500000 --out artifacts/cand_fit400k_extra_features.parquet
python -u src/features.py --candidates artifacts/cand_val50k.parquet        --split train --labels --batch 500000 --out artifacts/cand_val50k_features.parquet
python -u src/features.py --candidates artifacts/cand_test.parquet          --split test           --batch 500000 --out artifacts/cand_test_features.parquet

# 5. competition features (rev_rank, rev_count, rev_score_gap) over the whole population
python -u src/competition.py --name trainall
python -u src/competition.py --name test

# 6. write the whole-population rev_* into each feature file
python -u src/apply_competition.py --features artifacts/cand_fit200k_features.parquet       --competition artifacts/competition_trainall.parquet --out artifacts/cand_fit200k_features_fullrev.parquet
python -u src/apply_competition.py --features artifacts/cand_fit400k_extra_features.parquet --competition artifacts/competition_trainall.parquet --out artifacts/cand_fit400k_extra_features_fullrev.parquet
python -u src/apply_competition.py --features artifacts/cand_val50k_features.parquet        --competition artifacts/competition_trainall.parquet --out artifacts/cand_val50k_features_fullrev.parquet
python -u src/apply_competition.py --features artifacts/cand_test_features.parquet          --competition artifacts/competition_test.parquet     --out artifacts/cand_test_features_fullrev.parquet

# 7. train v3: fit600k (fit200k + fit400k_extra) + val50k for early stopping and the threshold
python -u src/train.py --features artifacts/cand_fit200k_features_fullrev.parquet --extra-features artifacts/cand_fit400k_extra_features_fullrev.parquet artifacts/cand_val50k_features_fullrev.parquet --scope-sample 50000 --out-dir models/v3 --n-jobs 16 --num-leaves 255 --n-estimators 3000 --learning-rate 0.05 --early-stop 100
python -u src/val_report.py --model models/v3 --features-val artifacts/cand_val50k_features_fullrev.parquet --cand artifacts/cand_val50k.parquet --scope-sample 50000 --k 5 --out models/v3/val_report.json

# 8. score test, threshold + global ownership, write output/, run both checkers
python -u src/score_test.py --features artifacts/cand_test_features_fullrev.parquet --candidates artifacts/cand_test.parquet --model models/v3/lgbm.txt --bundle models/v3/bundle.json --validator utils/validate_submission.py --batch-size 500000
```

Why steps 5-6: `features.py` computes the `rev_*` competition features inside the file it is
given, so a 50k- or 200k-S1 sample sees far less competition for each target than the 1.73M
test S1s do (negative-pair `rev_count` 13.2 inside fit200k vs 135.7 over all 2.2M train S1s).
The model is trained and scored on whole-population values. `apply_competition.py` reproduced
the v3 validation features exactly (all 43 columns identical, 1,183,283 rows).

Why `fit_extra.py` reads `cand_trainall`: each S1 query is retrieved independently against the
same target index, so the rows of `cand_trainall` for those S1s are their blocking output. It
reproduces the exact 400,000 ids used for v3.

Validation split: 20% of train S1s (seed 42) in `artifacts/split_train.parquet`, created by
`blocking.py`; the evaluated scope is a seed-42 sample of 50,000 of them, searched against all
train S2/S3 of the same country (distractors kept).

## Measured runtime and peak RAM

Windows 11, 16 GB RAM, 20 logical CPUs, unless marked. "-" = not measured on its own.

| step | rows | runtime | peak RAM |
|---|---|---|---|
| normalize train | 12.5M | ~8 min | 3.9 GB |
| normalize test (via `normalize_test.py`, same functions) | 11.7M | 395 s | 2.8 GB |
| blocking fit200k | 4,732,479 pairs | 205 s | 6.4 GB |
| blocking trainall | 52,219,962 pairs | 533 s | 9.1 GB |
| blocking test | 40,696,839 pairs | 471 s | 7.4 GB |
| blocking val50k | 1,183,283 pairs | - | - |
| fit_extra | 9,464,503 pairs | ~11 s | - |
| features fit200k | 4,732,479 | ~17,500 pairs/s | >= 3.6 GB |
| features fit400k_extra | 9,464,503 | ~16,500 pairs/s | >= 5.2 GB |
| features val50k | 1,183,283 | ~17,700 pairs/s | >= 3.0 GB |
| features test (macOS, 36 GB) | 40,696,839 | 1,714 s | 11.6 GB |
| competition trainall | 52.2M pairs | 82 s | 4.4 GB |
| competition test | 40.7M pairs | - | - |
| apply_competition fit200k / fit400k_extra / val50k | | 26 s / 49 s / 6 s | val50k 0.7 GB |
| train v3 (16 threads) | 15,380,265 pairs, 650,000 S1 | 973 s | 9.7 GB |
| score_test | 40.7M pairs | - | - |

Feature throughput is from the run logs; ">=" is the last peak RAM the log printed. The full
test feature build needs more than 16 GB in one pass (11.6 GB peak on macOS); build it on a
larger machine or in candidate slices. Slicing is safe for `rev_*` because step 6 overwrites
them with whole-population values.

## Retraining

`run.py train` refuses to overwrite an existing bundle. To retrain and score the result:

```text
python -u run.py train  --model-dir models/v3_retrained
python -u run.py report --model-dir models/v3_retrained
python -u run.py score  --model-dir models/v3_retrained
```

To use another model (e.g. a reverse-retrieval variant), put its `lgbm.txt` and `bundle.json` in
a folder and change `MODEL_DIR` in `run.py`, or pass `--model-dir`.

## Shared address TF-IDF

`artifacts/addr_tfidf.joblib` (SHA-256 `7a9d6370acc9eb3e55625c7e8d4a55b81e91f3320a9484f27c5a9098a30249e3`)
is fitted on addresses sampled from all six normalized sources. The train/val features behind
v3 and the test features must use the same file, so it is shipped. `features.py` refits it only
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
| `blocking.py` | frozen rare-token retrieval (`--produce val50k / fit200k / trainall / test`) |
| `fit_extra.py` | extra fitting S1s for fit600k |
| `features.py` | 43 pair features (`feature_columns()` is the train/score contract) |
| `competition.py`, `apply_competition.py` | whole-population `rev_*` |
| `train.py`, `val_report.py` | LightGBM training + threshold tuning; validation report |
| `score_test.py` | chunked test scoring, global ownership, TSVs, checks |
| `predict.py`, `ownership.py`, `io_utils.py`, `checks.py`, `evaluate.py` | decision policy, writers, strict checker, metric |
| `fallback.py`, `run_fallback.py` | rules fallback (val 0.6561) used for the first submission |
| others | experiments: `eda.py`, `sweep*.py`, `error_analysis.py`, `india_retrieval.py`, `reverse_retrieval.py`, `dev_candidates.py`, `patch_num_features.py`, `fallback_e01.py`, `normalize_test.py` |
