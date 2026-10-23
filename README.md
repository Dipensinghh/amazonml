# Business Entity Resolution · Amazon ML Challenge 2026

**Team Commit&Pray:** Dipen Singh, Zaid Khan, Aryan Jha, Diljit Singh

Given 1.73 million businesses from one source, find every record that refers to the same business across two other sources holding about 10 million rows, in the US, India and France. Names are misspelled, reordered, abbreviated or written in Devanagari; addresses are truncated, padded or blank; and France has no training labels at all.

The pipeline runs on CPU only, with no external data, APIs or pretrained models.

## Results

| Metric | Score |
|---|---|
| **Public leaderboard (macro F0.5)** | **0.947135** |
| Validation macro F0.5 | 0.9561 (US 0.9658 · India 0.9418) |
| Candidate recall ceiling | 0.9578 at 24.2 candidates per business |
| Candidate retrieval, full test set | 471 s, against an estimated ~22 h for the TF-IDF approach it replaced |

Validation scores are tuned: early stopping and the threshold were chosen on the same 50,000 validation businesses, so they are not untouched test estimates. The LightGBM models lost about 0.01 between validation and the public leaderboard.

## How it works

1. **Normalise and transliterate.** Unicode cleanup, transliteration to Latin script, and phonetic name keys, so a Devanagari name meets its English original. On true pairs with a non-Latin name, similarity rises from 66.7 on the normalised name to 93.1 on the phonetic key.
2. **Retrieve candidates on rare token pairs.** Single address words are too common to be useful, but pairs of them are close to unique. Scoring shared rare pairs with one sparse matrix product cut retrieval from an estimated 22 hours to under 8 minutes.
3. **Search in reverse too.** Every target record also searches for its best match, which recovers 40% of the true pairs the forward search missed, for only 1.8% more candidates.
4. **Score pairs with LightGBM** on 44 features: name and address similarity, house-number and postcode agreement, retrieval rank, and competition features that compare each candidate with every other business that retrieved the same record.
5. **Assign globally.** The labels show every record belongs to at most one business, so each record goes to its highest-scoring owner, with one threshold tuned on the exact competition metric.

## What we tried and dropped

Kept honest in [`EXPERIMENTS.md`](EXPERIMENTS.md) and [`docs/section5_results.md`](docs/section5_results.md):

- **Character-trigram TF-IDF retrieval:** stopped after 45 minutes on validation alone.
- **Cross-source corroboration features:** +0.0019, below our pre-set +0.005 bar, so not adopted.
- **A non-Latin phonetic index:** +0.1 point of recall for 0.5 more candidates per business. Not worth the cost.

Error analysis found a real bug: zero-padded house numbers were being read as conflicts. Fixing it gained +0.0024.

## My role (Dipen)

- Built the whole-population competition features over 52.2 million candidate pairs (82 s, 4.4 GB peak memory)
- Trained and evaluated the v2 and v3 model generations, including the leading-zero fix
- Made the submission reproducible end to end: packaging, the build script and its tests

## Repository layout

```text
src/                            pipeline code (normalise, blocking, features, train, score)
code/business_entity_resolution submission package: run.py, pinned requirements, final model bundle
models/                         trained LightGBM bundles
tests/                          pytest suite (23 tests)
Documentation_filled.md         full methodology write-up
EXPERIMENTS.md                  every experiment, kept or rejected
```

## Run it

```bash
cd code/business_entity_resolution
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python.exe -m pip ...
python -u run.py all                                    # or one stage: normalize, candidates, reverse, features, train, score
```

Place the organiser `dataset/` and `utils/` folders next to `run.py`. The competition data is not redistributed here. Full commands, runtimes and memory use are in [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).

**Stack:** Python, LightGBM, scikit-learn, SciPy sparse, pandas, PyArrow, RapidFuzz, ftfy, AnyAscii
