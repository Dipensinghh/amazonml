# Business Entity Resolution

CPU-only reproducible pipeline for matching every Source 1 business to zero or more Source 2
and Source 3 records. It normalizes the three sources, retrieves bounded candidates by country
and target source, computes lexical/retrieval features, trains LightGBM, tunes the decision
threshold with global target ownership, and writes the two required TSV files.

## Setup

Tested with CPython 3.14.7 on macOS and CPython 3.14.4 on Windows 11. From this directory:

```text
python -m venv .venv
```

macOS/Linux:

```text
.venv/bin/python -m pip install -r requirements.txt
```

Windows PowerShell:

```text
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the organizer `dataset/` and `utils/` directories here. Do not add external data. The
package must contain the integrated `src/` directory supplied in the final ZIP.

## Reproduce

Run one stage at a time so every artifact can be checked before the next long job:

```text
python -u run.py normalize
python -u run.py candidates --jobs 8
python -u run.py features --batch-size 500000
python -u run.py train
python -u run.py score --batch-size 500000
```

`python -u run.py all` runs the same sequence end to end. Intermediate Parquets are written to
`artifacts/`; final files are written to `output/`.

Measured on a 36 GiB, 14-logical-CPU Mac: frozen test retrieval took 433 seconds and 10.6 GB
peak RAM; 40,696,839 test feature rows took about 31 minutes and 11.6 GB peak RAM.

## Final checks

`score_test.py` writes every score in candidate order, applies the tuned threshold and global
target ownership, writes both TSVs, and runs both validators. They can also be rerun directly:

```text
python -u src/checks.py --out-dir output --test-dir dataset/test
python -u utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Expected files:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

Each contains one row per test S1, including empty outputs. Matches are a subset of candidates;
each target receives at most one predicted owner, with exact ties resolved by smaller S1 ID.
