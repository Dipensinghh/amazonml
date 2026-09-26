# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Commit&Pray  
**Team Members:** [fill in]  
**Submission Date:** 27 September 2026

> Status legend: **[measured]** = produced by code in this repo; **[pending]** = not run yet, do not treat as a result.

---

## 1. Executive Summary
Blocking + classifier pipeline. Names and addresses are normalized and transliterated to Latin script; three retrieval channels (character-trigram TF-IDF on the name, a phonetic name view, and address tokens) plus an exact-acronym channel shortlist candidates per Source 1 entity within each country and target source; a LightGBM classifier scores each pair on string-similarity features; a single validation-tuned threshold and a global "one Source 1 owner per Source 2/3 record" rule produce the final matches. **[pending: final scores]**

---

## 2. Methodology

### 2.1 Problem Analysis (verified on the training files)
- Train: 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3 rows (US 60% / India 40%). Test: 1,732,544 / 4,887,273 / 5,082,316 rows; test adds **France** (15% of test S1), absent from training.
- Ground truth: 3.46 matches per S1 on average (max 11); 5.585% of S1 have no match; 85% of matched S1 have matches in both S2 and S3.
- **Every labeled S2/S3 record belongs to exactly one S1** (7,638,365 distinct targets, 0 repeats). About a quarter of S2/S3 rows match nothing (distractors).
- S1 is entirely Latin script. In India, S2/S3 names are ~23.5% / ~13% non-Latin (mostly Devanagari, some Telugu/Kannada/Tamil); addresses are often mixed-script.
- Noise seen: typos, reordered address parts, abbreviations, leading junk (`##`, `--`), dba/fka names, acronyms, missing addresses (~3% of S2/S3 rows).
- Team finding: Indic names in S2/S3 are **transliterations** of the S1 name, not translations, so transliteration plus a phonetic key recovers them. In India, address blocking recall (~92%) is much higher than name blocking recall (~58%), so the address channel is essential. **[teammate measurement; exact numbers to be copied from EXPERIMENTS.md]**
- The organizer metric is macro F0.5 per S1, with singletons scoring 1.0 only for an empty prediction.

### 2.2 Solution Strategy
**Approach Type:** Blocking + supervised classifier + global assignment policy.  
**Core Innovation:** (1) transliteration + phonetic views so Indic and Latin text meet in one space; (2) address as a first-class retrieval channel; (3) global target-ownership after scoring, which follows from the one-owner structure of the labels; (4) all decisions tuned on the exact macro F0.5 metric after ownership.

**Validation.** 20% of train S1s (seed 42, `artifacts/split_train.parquet`) are held out; queries search **all** train S2/S3 of the same country so distractors stay in. Model fitting uses only the other 80% of S1s. Early stopping and threshold selection use the validation S1s, so the reported validation score is a *tuned* score, not an untouched test estimate. Validation competition is thinner than at test time (20% of S1s).

---

## 3. Candidate Generation (Blocking)
- **Partitioning:** per observed country string (open set, no US/India whitelist) and per target source (S2, S3 separately).
- **Channels:** A = char 3-gram TF-IDF on `name_core`; P = char 3-gram TF-IDF on the phonetic name skeleton; B = word TF-IDF on the phonetic address; C = exact acronym match (bounded bucket size). Names with dba/fka parts are indexed twice.
- **Blocking keys used:** normalized/transliterated name, phonetic name, address tokens, acronym.
- **Candidate pairs generated / recall / cost:** **[pending: table from EXPERIMENTS.md - k per channel and source, pair recall, macro recall, oracle F0.5, total / mean / p95 candidates, runtime, peak RAM]**
- **How true matches are protected:** union of independent channels; the smallest k near the recall plateau is chosen from a measured recall-versus-cost sweep. `candidate_pairs.tsv` contains every pair the matcher scored, including rejected ones.

---

## 4. Matching Model

**Features used:** **[pending: final schema from features.py]** Planned families: name (ratio, token-set, token-sort, partial, Jaro-Winkler on normalized/core views; acronym match), address (token-set, TF-IDF cosine, missing flags), numeric/postal-token agreement and conflict, retrieval evidence (channel flags, score, rank, within-list gap), competition (reverse rank), script and blank-address flags. No country one-hot (France is unseen).

**Model type:** LightGBM binary classifier (`n_estimators=800`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `reg_lambda=1`, early stopping 50 rounds on validation log loss, seed 42). Fitted on labeled candidate pairs of fitting S1s; positives = pairs in the ground-truth list, negatives = all other retrieved candidates (hard negatives included).

**Decision policy:** score every candidate; give each S2/S3 record to the S1 with the highest score (ties: smallest S1 id), keep it if score >= t; one S1 may keep many records. Threshold swept 0.10-0.95 in 0.05 steps, then refined in 0.01 steps, choosing the maximum **macro F0.5 after ownership** over all validation S1s (including those with no candidates). Ownership is resolved globally, never inside chunks.

**Threshold selection method:** F0.5 optimization on the validation set, reported with and without ownership.

**Rules-based fallback, D sample version (E01) [measured]:** word-level TF-IDF retrieval (name and address, top-10 each) with a hand-weighted score `w*name_cos + (1-w)*addr_cos`, one tuned threshold and the same ownership rule. On 2,000 random validation S1s searched against all train S2/S3 of the same country: pair recall 0.930 at 37.9 candidates per S1; best name weight 0.2 (address-heavy), t = 0.73, **macro F0.5 = 0.662** (US 0.706, India 0.574) versus 0.056 for predicting no matches. Ownership changed the score by only +0.001 at this sample size (collision rate 0.2%), because 2,000 S1s give almost no competition for targets. A's fallback on the real frozen candidates (`cand_val50k`, 23.7 candidates per S1, recall ceiling 0.929) scored macro F0.5 **0.6561** with ownership versus **0.6049** without, so global ownership added +0.051 there. That is the baseline LightGBM must beat on the same candidates.

---

## 5. Results & Error Analysis
- **F0.5 Score (macro), validation:** rules fallback 0.662 (2k-S1 sample) **[measured]**; LightGBM **[pending]**.
- **Ownership effect (constrained vs unconstrained):** **[pending: full-scale validation]**
- **Per-country:** US / India **[measured for fallback: 0.706 / 0.574]**; France has no labels, so only coverage, candidate counts and score distributions can be reported, not accuracy.
- **Common false positives (wrong merges):** **[pending]** Expected: same-chain branches and shared-address businesses with similar names.
- **Common false negatives (missed matches):** **[pending]** Expected: Indic-script names with poor transliteration, names made only of common words, blank addresses.
- **Limitations:** France is unlabeled; validation is shared-corpus and has reduced S1 competition; the threshold is tuned on the same validation labels used for early stopping.

---

## 6. Conclusion
**[pending]** To be written from the final measured results.

---

## Appendix A: Code structure and reproduction
`src/normalize.py` (normalization and transliteration), `src/blocking.py` (candidate generation), `src/features.py` (pair features), `src/evaluate.py` (exact macro F0.5, candidate diagnostics), `src/train.py` (labels, LightGBM, model bundle), `src/predict.py` (scoring, global ownership, threshold sweep, TSV writer), `src/fallback_e01.py` (D sample rules baseline E01; A owns the submission fallback `src/fallback.py`), `src/valsplit.py` (shared split). Pinned environment in `requirements.txt`. **[pending: single entry point and run instructions]**

## Appendix B: Provenance
No external data, APIs, gazetteers or pretrained models are used. Libraries: scikit-learn, LightGBM, RapidFuzz, sparse_dot_topn, anyascii, ftfy, pyarrow, pandas. **[pending: final versions, runtime, hardware, hashes of the submitted outputs]**
