# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Commit&Pray  
**Team Members:** Dipen Singh, Zaid Khan, Aryan Jha, Diljit Singh  
**Submission Date:** 27 September 2026

All scores marked "val" are tuned validation scores: early stopping and the threshold were chosen on
the same 50,000 validation S1s, so they are not untouched test estimates. "Public" is the leaderboard.

---

## 1. Executive Summary

Blocking + classifier. Names and addresses are normalized and transliterated to Latin script;
a rare-token retriever proposes about 23.5 candidates per Source 1 (S1) entity within its country and
target source; LightGBM scores each pair on 43 string, retrieval and cross-S1 competition features; one
threshold plus a global "each S2/S3 record has at most one S1 owner" rule gives the matches. The final
model v3 reaches **val macro F0.5 0.9423** (US 0.9578, India 0.9192) against a blocking recall ceiling
of 0.9293; the earlier v1b scored 0.9262 val / 0.9164 public.

---

## 2. Methodology

### 2.1 Problem Analysis

- Train: 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3 rows (US ~60%, India ~40%). Test: 1,732,544 /
  4,887,273 / 5,082,316 rows, adding **France** (259,452 test S1s, ~15%), which has no training labels.
- Ground truth: 3.46 matches per S1 on average (max 11); 5.6% of S1s have none; 85% of matched S1s have
  matches in both S2 and S3. **Every labelled S2/S3 record belongs to exactly one S1**, and about a
  quarter of S2/S3 rows match nothing (distractors).
- No labelled pair crosses countries, so partitioning retrieval by country is lossless on train.
- S1 is fully Latin script. In India, ~23.5% of S2 and ~13% of S3 names are non-Latin (mostly
  Devanagari), and addresses are often mixed-script. They are **transliterations** of the English
  name, not translations: on 551,240 true pairs with a non-Latin name, mean token-set similarity is
  66.7 on the normalized core name but 93.1 on a phonetic skeleton of it.
- Noise: typos, reordered and abbreviated address parts, leading junk (`##`, `--`), dba/fka names,
  acronyms, zero-padded or truncated house numbers, blank addresses (~3% of S2/S3).
- Names are weakly identifying (many near-identical business names); the address is the stronger
  retrieval signal (below).

### 2.2 Solution Strategy

**Approach Type:** Blocking + supervised classifier + global assignment.  
**Core Innovation:** (1) retrieval on rare *pairs* of address/name tokens instead of TF-IDF nearest
neighbours: 471 s measured for the full test set against an estimated ~22 h for the TF-IDF
channels; (2) phonetic, script-agnostic name keys so
Indic transliterations meet their Latin originals; (3) cross-S1 competition features computed over the
whole population, matching the scale of the 1.73M-S1 test set; (4) global target ownership, which
follows from the one-owner structure of the labels, with the threshold tuned on exact macro F0.5 after
ownership.

**Validation.** 20% of train S1s held out (seed 42). Evaluation uses a seed-42 sample of 50,000 of
them (US 29,914, India 20,086; 2,702 singletons), searched against **all** train S2/S3 of the same
country, so distractors stay in. Models are fitted only on the other 80%.

---

## 3. Candidate Generation (Blocking)

- **Normalization:** ftfy + Unicode NFKC + AnyAscii transliteration, lowercasing, punctuation cleanup,
  small hand-written legal-suffix and address-abbreviation dictionaries; dba/fka/aka aliases kept
  separately; derived acronym, sorted-name, numeric, postal-like and phonetic fields.
- **Blocking keys used (v3, frozen):** per country × target source, hashed tokens: address letter/digit
  runs (leading zeros stripped), house numbers truncated by one digit (they are perturbed), all
  unordered pairs of non-common address tokens, name words, name-word pairs and the concatenated name.
  Tokens whose target document frequency exceeds 200 are dropped. Score = sum of IDF over shared tokens
  (address weight 1.0, name 0.5) via one sparse matrix product; a 30-record pool per source is reranked
  by score + 6 × name token-set similarity; the top **10 per source** are kept, plus up to **2 per
  source** from a separate name-only index for blank-address targets.
- **Candidate pairs generated (test):** 40,696,839 = **23.49 per S1** (US 23.76, India 23.53, France
  22.68); one France S1 has no candidate. Runtime 471 s, peak RAM 7.4 GB.

**Evolution (50k validation S1s; recall = share of true pairs present in the candidates):**

| version | method | recall @ candidates/S1 | test cost |
|---|---|---|---|
| v1 | char-trigram TF-IDF top-k (sparse_dot_topn threads) | not completed | stopped after 45 min on validation |
| v2 | TF-IDF channels: name trigram (A), address words (B), k=10 per channel and source | A 0.598 @ 20.0; **A+B 0.943 @ 37.8** | ~22 h estimated (all channels) |
| v3 | rare-token pairs (above), K=10 | **0.9293 @ 23.7** (US 0.9621, India 0.8806) | **471 s** measured |

v3 sweep (cap = document-frequency cap, α = name-similarity weight in the rerank):

| cap | α | K=5 | K=10 | K=15 |
|---:|---:|---|---|---|
| 200 | 0 | 0.8955 @ 13.7 | 0.9214 @ 23.7 | 0.9315 @ 33.6 |
| 200 | 6 | 0.9082 @ 13.7 | **0.9293 @ 23.7** | 0.9373 @ 33.6 |
| 500 | 6 | 0.9073 @ 13.9 | 0.9286 @ 23.9 | 0.9365 @ 33.9 |
| 2000 | 6 | 0.9069 @ 14.0 | 0.9287 @ 24.0 | 0.9364 @ 33.9 |

- **How true matches were protected:** the address carries retrieval (in v2 the address channel alone
  found 28.7% of true pairs no other channel found; adding an acronym channel gained ~0); single
  address words are too common (document frequency 100-5,000) while token pairs are near-unique; the
  blank-address index fixed the largest miss class (before it, 53% of US and 23% of India misses were
  blank-address targets). The 30-per-source pool reaches 0.9475 @ 69.3.
- **Not adopted:** a non-Latin phonetic index (+0.1 point overall for +0.5 candidates/S1); four India
  variants (higher caps, transliteration variants, phonetic address pairs), all 0.02-2.42 points worse.
  Reverse retrieval (each target searches the S1 index, top R added) passed its gate: R=1 0.9579 @
  24.10, R=3 0.9627 @ 28.64; it is not part of the v3 model.

---

## 4. Matching Model

**Features used (43, float32; `features.feature_columns()` is the train/score contract):**
- Name: ratio, token-set, token-sort, partial ratio and Jaro-Winkler on the normalized, core and
  phonetic names (15); acronym match; alias (dba/fka) token-set; name-missing flag.
- Address: token-set on normalized and phonetic address; character-trigram TF-IDF cosine on both
  (vectorizer fitted on addresses from all six sources, shared by train and test); house-number
  agreement / conflict / missing (leading zeros stripped); postal agreement / conflict / missing;
  blank-address flag.
- Other: was-non-Latin and is-S3 flags; retrieval score, rank, rank fraction, candidate count, gap and
  ratio to the S1's best candidate, token score, name similarity, from-blank-index flag;
  **competition**: the pair's rank among all S1s that retrieved the same target (`rev_rank`), the number
  of such S1s (`rev_count`) and the score gap to the best of them (`rev_score_gap`).
- No country feature (France is unseen).

**Model type:** LightGBM binary classifier. v3: 255 leaves, learning rate 0.05, up to 3,000 trees with
early stopping (100 rounds) on validation log loss (stopped at 2,554), min_child_samples 50,
reg_lambda 1, seed 42. Fitted on 14,196,982 candidate pairs of 600,000 fitting S1s (13.6% positive);
positives are candidates in the ground truth, negatives are all other retrieved candidates.

**Competition at the right scale.** Computed inside a sampled candidate file, `rev_*` see only that
file's S1s (negative-pair `rev_count` 13.2 inside fit200k). Recomputed over all 2.2M train S1s it is
135.7, and the share of wrong pairs with `rev_rank = 0` falls from 68.5% to 10.1%. The model is trained
on whole-population values and test values are computed over all 1.73M test S1s.

**Threshold selection method:** each S2/S3 record goes to the S1 that scores it highest (ties: smaller
S1 id) and is kept if score ≥ t; t is swept from 0.10 to 0.95 in 0.05 steps, then refined in 0.01 steps
around the best, maximizing macro F0.5 after ownership over all 50,000 validation S1s. v3: t = 0.66.

---

## 5. Results & Error Analysis

### Model lineup (val50k, global ownership; blocking v3 candidates, K = 10)

| model | features | leaves / trees | fit S1s | val F0.5 | t | US | India | pair P / R | singleton acc. | public |
|---|---|---|---|---|---|---|---|---|---|---|
| rules fallback | 3 token-set scores | – | – | 0.6561 | 0.85 | | | | | 0.658 |
| v1 | 43 (rev_* within file) | 31 / 800 | 200k | 0.9256 | 0.71 | 0.9439 | 0.8984 | 0.9794 / 0.8504 | 0.9175 | |
| v1b | 40 (no rev_*) | 31 / 800 | 200k | 0.9262 | 0.68 | 0.9448 | 0.8985 | 0.9813 / 0.8478 | 0.9171 | 0.9164 |
| v2q | 40 (no rev_*) | 255 / 1,648 | 200k | 0.9326 | 0.70 | 0.9491 | 0.9079 | 0.9833 / 0.8602 | 0.9178 | |
| v2cq | 43 (whole-population rev_*) | 255 / 1,661 | 200k | 0.9370 | 0.68 | 0.9529 | 0.9133 | 0.9825 / 0.8763 | 0.9130 | |
| v2cq_fix | 43, leading-zero fix | 255 / 1,503 | 200k | 0.9394 | 0.73 | 0.9561 | 0.9145 | 0.9859 / 0.8739 | 0.9393 | |
| **v3** | 43, as v2cq_fix | 255 / 2,554 | 600k | **0.9423** | 0.66 | 0.9578 | 0.9192 | 0.9841 / 0.8844 | 0.9352 | |

- **F_0.5 Score (macro):** **0.9423** val (v3). Predicting no matches scores 0.054; the blocking recall
  ceiling is 0.9293 (US 0.9621, India 0.8806).
- **What moved the score:** supervised model over rules +0.27 on the same candidates; capacity (31 → 255
  leaves, early stopping) +0.0064; whole-population competition features +0.0044 (within-file rev_*
  added nothing: v1 0.9256 vs v1b 0.9262); leading-zero house-number fix +0.0024 (true pairs flagged as
  a house-number conflict fell 11,907 → 7,305; singleton accuracy 0.913 → 0.939); 600k instead of 200k
  fitting S1s +0.0029, mostly recall (0.8739 → 0.8844) and India (0.9145 → 0.9192); v3's tuned
  threshold fell from 0.73 to 0.66, trading a little singleton accuracy (0.9393 → 0.9352; 175 of 2,702
  singletons get a false match) for that recall.
- **Validation vs public:** fallback 0.6561 val / 0.658 public; v1b 0.9262 / 0.9164 (−0.01).
- **Fewer candidates:** re-scoring v3 on K = 5 per source (9.6 candidates/S1 instead of 23.7) gives
  0.9232 (−0.019); K = 10 was kept.
- **Ownership:** at 50,000 validation S1s it changes v3 by < 0.0001 (collision rate 0.011%); on the
  less confident rules fallback it added +0.051 (0.6049 → 0.6561). It stays on for test, where 1.73M
  S1s compete for targets.
- **Cross-source corroboration (tried, not adopted).** 85% of matched S1s have matches in both S2 and S3,
  so a true candidate should resemble one of the S1's candidates from the other source. For each pair,
  the target was compared with the S1's top-5 other-source candidates (best name token-set, best
  address token-set, best mean of the two, and that partner's rank). The signal is real
  (`xsrc_best_both` 0.870 on true pairs vs 0.588 on non-matches), but on v2q's 40 features with the
  leading-zero fix it added only **+0.0019** (0.9355 → 0.9374): below-threshold misses fell 10.7%
  (11,153 → 9,959) and pair recall rose 0.0069, while precision moved −0.0006. It cannot reach pairs
  that blocking never retrieved (half of all misses), so it fell short of the +0.004 adoption bar.
- **Other:** dropping 6 low-signal features cost −0.0011, so all 43 were kept.

**Error analysis** (v2q, t = 0.70, before the leading-zero fix). Of 173,059 true pairs 24,194 were
missed; of 151,394 predicted pairs 2,529 were wrong.

| error type | count | share |
|---|---|---|
| Miss: never retrieved by blocking | 12,237 | 50.6% of misses (68% India) |
| Miss: retrieved, scored below t | 11,950 | 49.4% of misses (41% with a blank candidate address) |
| Miss: lost to another S1 under ownership | 7 | ~0% |
| False merge: record belongs to no S1 (distractor) | 1,690 | 66.8% of false merges |
| False merge: record belongs to another S1 | 839 | 33.2% of false merges |

- **Common false positives (wrong merges):** "twins": 60% have near-identical names (token-set ≥ 0.9),
  usually on the same street with a neighbouring house, flat or plot number; 9.6% share an identical
  address with an unrelated name (another business in the same building); records with a blank address
  whose name matches an S1 elsewhere (17.4% of false merges).
- **Common false negatives (missed matches):** India addresses reduced to common tokens (city, state,
  a short number) are never retrieved; blank candidate addresses (only 22.8% of such true pairs are
  recalled); zero-padded house numbers read as conflicts (38.6% of true pairs flagged with a
  house-number conflict, fixed in v2cq_fix); name words replaced or reordered while the address is
  truncated.

---

## 6. Conclusion

Rare-token-pair blocking made full-test retrieval take 8 minutes instead of an estimated 22 hours at
0.929 recall and 23.5 candidates per S1, and a LightGBM matcher with whole-population competition
features, global ownership and a metric-tuned threshold reaches 0.9423 val macro F0.5. The main lessons:
the address, not the name, carries retrieval; features must be computed at the scale they are used
(competition features only helped once computed over all S1s); and small data bugs (zero-padded house
numbers) were worth as much as model tuning. The remaining gap is mostly retrieval in India.

**Limitations.** France (15% of test) is unlabelled and unseen in training, so only coverage can be
checked there. All scores are bounded by the blocking recall ceiling (0.9293; India 0.8806). Validation
scores are tuned on the same labels used for early stopping; v1b lost 0.01 from validation to public.
Validation has 50,000 S1s competing for targets against 1.73M at test, so ownership is exercised far
less in validation than at test.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains `src/` (all pipeline code), `run.py`, `README.md`,
pinned `requirements.txt`, the v3 bundle (`models/v3/`: `lgbm.txt`, `bundle.json` with the ordered
features and threshold) and the shared address TF-IDF (`artifacts/addr_tfidf.joblib`, SHA-256
`7a9d6370acc9eb3e55625c7e8d4a55b81e91f3320a9484f27c5a9098a30249e3`). Entry point: `python -u run.py
all` (or one stage at a time: normalize, candidates, fit_extra, features, competition,
apply_competition, train, report, score); the scoring model is the single value `MODEL_DIR` in
`run.py`. `src/score_test.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`
and runs the strict checker (`src/checks.py`) and the organizer validator. The README lists every
command with measured runtime and peak RAM (CPython 3.14.4, Windows 11, 16 GB RAM).

No external data, APIs, gazetteers or pretrained models are used. Libraries: numpy, scipy, pandas,
pyarrow, scikit-learn, joblib, sparse_dot_topn, rapidfuzz, ftfy, anyascii, LightGBM, psutil.

### B. Additional Results

Per-country blocking recall at the frozen config: US 0.9621, India 0.8806 (India misses: 4,727 other
noise or pool/rerank limits, 2,049 transliteration mismatch, 1,074 shared tokens above the cap, 477
blank target address). Full experiment log: `EXPERIMENTS.md`; retrieval write-up:
`docs/section2_retrieval.md`; results and 20 + 20 error examples: `docs/section5_results.md`.
