Also read AGENTS.md if present

# Amazon ML Challenge 2026 — Business Entity Resolution

## Environment
Windows + PowerShell. Use Python scripts, never bash one-liners (`&&`, `head`, `grep` fail).
Write files with LF line endings via the Write tool, not shell edits.
Deadline: **27 Sept 2026, 11:59 PM IST**. Max 5 leaderboard submissions/day.

## Task
Three sources of noisy business records (entity_id, business_name, business_address, country).
Source 1 (S1-) is deduplicated. For EVERY S1 entity, output all matching S2-/S3- records (0, 1 or many).
Train: US + India. Test: US + India + **France (unseen, ~15% of test)**. Treat `country` as an open string set.

## Data
- `dataset/train/train_source{1,2,3}.tsv`, `train_ground_truth.tsv`; `dataset/test/test_source{1,2,3}.tsv`
- Read with `sep="\t"`, all columns as str, no quoting (pyarrow). ~2.5 GB extracted.

## EDA findings (verified)
- Train: S1 2.2M, S2 5.0M, S3 5.3M rows (~60/40 US/India). Test: S1 1.7M, S2 4.9M, S3 5.1M.
- No nulls except blank addresses (~3% of S2/S3, none in S1). No duplicate IDs.
- Singletons only 5.6%. Mean 3.46 matches per S1 (1.67 S2, 1.79 S3), max 11; 85% of matched S1s have both S2 and S3 matches.
- **Each S2/S3 ID matches at most one S1** → enforce at prediction time. ~26% of S2/S3 rows match nothing (distractors).
- S1 is 100% Latin script. Non-Latin text only in India S2/S3 (S2 names 23.5%, S3 names 13%, addresses ~23%): mostly Devanagari, some Telugu/Kannada/Tamil, mixed-script addresses.
- Noise: reordered address parts, uppercase, abbreviations (LN, DR, R.), leading junk (`##`, `--`), name typos, word reordering, dba / f/k/a prefixes, acronyms ("IAG" = Indian Ace Global).
- France: "number R./Rue street, city, region"; suffixes SARL/SAS/SASU/EURL.

## Metric
Macro F0.5 per S1 entity (singletons included: empty on a true singleton = 1.0, anything else = 0.0).
Since 94% of S1s have matches, recall matters a lot; precision still weighted 2x.

## Pipeline
1. **normalize.py** — ftfy+NFKC, anyascii transliteration, lowercase, strip accents/junk/punct.
   Name: split dba/fka into separate fields, legal-suffix dict (pvt, ltd, llc, inc, corp, co, sarl, sas, sasu, eurl, sa, gmbh), & → and.
   Outputs: name_norm, name_core, name_acronym, name_sorted. Address: abbreviation dict, numeric tokens, postal-like tokens (regex only).
2. **blocking.py** — per country, per source (S2 and S3 separately):
   channel A = char 3-gram TF-IDF top-k on name_core; channel B = address TF-IDF top-k; channel C = acronym exact match.
   Union, dedupe. Chunked sparse matmul (sparse_dot_topn if available). Pick the smallest k whose recall ceiling is near max.
   Output parquet: s1_id, cand_id, source, channel, tfidf_name, tfidf_addr, rank.
3. **features.py** — per pair, vectorized (rapidfuzz.process.cpdist, multithreaded):
   name ratio / token_set / token_sort / partial / Jaro-Winkler on norm & core; acronym match; address token_set & TF-IDF cosine;
   numeric-token and postal-token agreement (+ missing flags); rank and score gap within the S1's list; reverse rank (cand's rank among S1s competing for it); blank-address flag.
   No country one-hot. Script flag (was_non_latin) is fine.
4. **train.py** — LightGBM binary on candidate pairs of train S1s (subsample S1s if slow), validated on the held-out 20% S1s.
5. **predict.py** — score candidates → keep p > t → assign each S2/S3 ID only to its highest-scoring S1 → write both TSVs.
   Tune t (and optionally a per-S1 relative-gap rule) for macro F0.5 on validation.
6. **evaluate.py** — exact macro F0.5, blocking recall, avg candidates/S1, per-country breakdown.

Validation: hold out 20% of train S1s (seed 42), search against ALL train S2/S3 so distractors stay in.

## Optional (only if time and the transliteration check fails)
Multilingual embedding cosine as an extra feature (e.g. intfloat/multilingual-e5-small, MIT, verify license) — restricted to India rows with non-Latin text; CPU-heavy, skip if no GPU.

## Outputs (both in `output/`, tab-separated, no quoting)
- `matching_results.tsv`: `source1_entity_id<TAB>matched_entity_ids` (scored)
- `candidate_pairs.tsv`: `source1_entity_id<TAB>candidate_entity_ids` = exact set fed to the model
One row per test S1, empty lists allowed, S2-/S3- test IDs only, no duplicates, matches ⊆ candidates.
Smaller candidate sets rank higher in final judging.
Validate: `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Hard rules
- No external data: no geocoding, libpostal, postal/city/gazetteer tables, translation APIs, hosted LLM APIs.
- Allowed: rapidfuzz, jellyfish, scikit-learn, LightGBM, pyarrow/pandas, anyascii, indic-transliteration, ftfy, small hand-written dicts.
- Any pretrained model: MIT/Apache-2.0, ≤8B params each, offline. Record model + license in docs.

## Final package
```
<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt (pinned)}
└── Documentation_template.md (filled)
```
README must reproduce end-to-end: data → normalize → blocking → features → train → predict. Comment all functions.

## Conventions
Artifacts in `artifacts/` as parquet; seed 42; process by country and in chunks to bound RAM.
Log every experiment in EXPERIMENTS.md: blocking recall, avg candidates/S1, val F0.5 (overall + per country), runtime.

## Git
Repo: https://github.com/Dipensinghh/amazonml (branch `main`). Never commit data (dataset/, sample/, artifacts/, output/, *.tsv, *.parquet, *.zip) or files >50 MB.
Never commit organizer materials (PDFs, video, organizer README.md, Documentation_template.md, utils/) — repo may go public.
Commit after each pipeline stage; the message must include the key metric where relevant,
e.g. `blocking: TF-IDF name+addr, recall 0.973 @ 38 cands/S1` or `train: LGBM v1, val F0.5 0.912`.
