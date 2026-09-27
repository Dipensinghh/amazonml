# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Commit&Pray  
**Team Members:** Dipen Singh, Zaid Khan, Aryan Jha, Diljit Singh  
**Submission Date:** 27 September 2026

All scores marked "val" are tuned validation scores: early stopping and the threshold were chosen on
the same 50,000 validation S1s, so they are not untouched test estimates. "Public" is the leaderboard.

---

## 1. Executive Summary

Blocking + classifier. Names and addresses are normalized and transliterated to Latin script. A
rare-token retriever proposes candidates in both directions: forward (each Source 1 (S1) entity
searches the S2/S3 records of its country) and reverse (each S2/S3 record adds its best S1), about
24.2 candidates per S1 on test. LightGBM scores each pair on 44 string, retrieval and cross-S1
competition features; one threshold plus a global "each S2/S3 record has at most one S1 owner" rule
gives the matches. The final model (R=1 reverse retrieval, trained on 600k fitting S1s) reaches
**val macro F0.5 0.9561** (US 0.9658, India 0.9418) against a candidate recall ceiling of 0.9578, and
**0.947135 on the public leaderboard** at threshold 0.78.

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
channels; (2) **reverse retrieval**: every S2/S3 record also searches the S1 index and adds its best
S1, which recovers 40% of the true pairs forward search missed for 1.8% more candidates on validation
(3.1% more on test); (3)
phonetic, script-agnostic name keys so Indic transliterations meet their Latin originals; (4) cross-S1
competition features computed over the whole population, matching the scale of the 1.73M-S1 test set;
(5) global target ownership, which follows from the one-owner structure of the labels, with the
threshold tuned on exact macro F0.5 after ownership.

**Validation.** 20% of train S1s held out (seed 42). Evaluation uses a seed-42 sample of 50,000 of
them (US 29,914, India 20,086; 2,702 singletons), searched against **all** train S2/S3 of the same
country, so distractors stay in. Models are fitted only on the other 80%.

---

## 3. Candidate Generation (Blocking)

- **Normalization:** ftfy + Unicode NFKC + AnyAscii transliteration, lowercasing, punctuation cleanup,
  small hand-written legal-suffix and address-abbreviation dictionaries; dba/fka/aka aliases kept
  separately; derived acronym, sorted-name, numeric, postal-like and phonetic fields.
- **Forward blocking keys (frozen):** per country × target source, hashed tokens: address letter/digit
  runs (leading zeros stripped), house numbers truncated by one digit (they are perturbed), all
  unordered pairs of non-common address tokens, name words, name-word pairs and the concatenated name.
  Tokens whose target document frequency exceeds 200 are dropped. Score = sum of IDF over shared tokens
  (address weight 1.0, name 0.5) via one sparse matrix product; a 30-record pool per source is reranked
  by score + 6 × name token-set similarity; the top **10 per source** are kept, plus up to **2 per
  source** from a separate name-only index for blank-address targets.
- **Reverse retrieval (R = 1):** every S2/S3 record queries the same country-specific token-pair index
  built over the S1s and keeps its single best S1 (same scoring and rerank). The result is unioned with
  the forward candidates: forward evidence is kept where both directions found the pair, reverse-only
  pairs carry the reverse scores and a `from_reverse` flag, and final ranks are recomputed over the union.
- **Candidate pairs generated (test):** **41,955,111 = 24.22 per S1**: 40,696,839 forward (23.49 per S1;
  US 23.76, India 23.53, France 22.68) plus 1,258,272 reverse-only.

**Evolution (50k validation S1s; recall = share of true pairs present in the candidates):**

| version | method | recall @ candidates/S1 | test cost |
|---|---|---|---|
| v1 | char-trigram TF-IDF top-k (sparse_dot_topn threads) | not completed | stopped after 45 min on validation |
| v2 | TF-IDF channels: name trigram (A), address words (B), k=10 per channel and source | A 0.598 @ 20.0; **A+B 0.943 @ 37.8** | ~22 h estimated (all channels) |
| v3 forward | rare-token pairs (above), K=10 | 0.9293 @ 23.7 (US 0.9621, India 0.8806) | 471 s measured |
| **v3 forward ∪ reverse R=1** | + each target's top-1 S1 | **0.9578 @ 24.10** (US 0.9777, India 0.9281) | 41.96M test pairs (24.22/S1) |

Forward sweep (cap = document-frequency cap, α = name-similarity weight in the rerank):

| cap | α | K=5 | K=10 | K=15 |
|---:|---:|---|---|---|
| 200 | 0 | 0.8955 @ 13.7 | 0.9214 @ 23.7 | 0.9315 @ 33.6 |
| 200 | 6 | 0.9082 @ 13.7 | **0.9293 @ 23.7** | 0.9373 @ 33.6 |
| 500 | 6 | 0.9073 @ 13.9 | 0.9286 @ 23.9 | 0.9365 @ 33.9 |
| 2000 | 6 | 0.9069 @ 14.0 | 0.9287 @ 24.0 | 0.9364 @ 33.9 |

Reverse quota (experiment; ranks 1-R of each target's reverse search added to the forward pairs):
R=1 0.9579 @ 24.10, R=2 0.9608 @ 26.02, R=3 0.9627 @ 28.64 (predeclared gate: +2 recall points at
≤ 30 candidates/S1; all pass). R=1 recovers 40.4% of the forward misses for +0.44 candidates/S1 and was
chosen for production (0.9578 @ 24.10 in the production union).

- **How true matches were protected:** the address carries retrieval (in v2 the address channel alone
  found 28.7% of true pairs no other channel found; adding an acronym channel gained ~0); single
  address words are too common (document frequency 100-5,000) while token pairs are near-unique; the
  blank-address index fixed the largest miss class (before it, 53% of US and 23% of India misses were
  blank-address targets); reverse retrieval catches true pairs that lose the forward top-10 cut,
  mostly in India (0.8806 → 0.9281).
- **Not adopted:** a non-Latin phonetic index (+0.1 point overall for +0.5 candidates/S1); four India
  variants (higher caps, transliteration variants, phonetic address pairs), all 0.02-2.42 points worse.

---

## 4. Matching Model

**Features used (44, float32; `features.feature_columns()` is the train/score contract):**
- Name: ratio, token-set, token-sort, partial ratio and Jaro-Winkler on the normalized, core and
  phonetic names (15); acronym match; alias (dba/fka) token-set; name-missing flag.
- Address: token-set on normalized and phonetic address; character-trigram TF-IDF cosine on both
  (vectorizer fitted on addresses from all six sources, shared by train and test); house-number
  agreement / conflict / missing (leading zeros stripped); postal agreement / conflict / missing;
  blank-address flag.
- Retrieval: score, rank, rank fraction, candidate count, gap and ratio to the S1's best candidate,
  token score, name similarity, from-blank-index flag, **`from_reverse`** (pair found only by reverse
  retrieval); all list-dependent values are computed over the forward ∪ reverse union.
- **Competition**: the pair's rank among all S1s that retrieved the same target (`rev_rank`), the number
  of such S1s (`rev_count`) and the score gap to the best of them (`rev_score_gap`).
- Other: was-non-Latin and is-S3 flags. No country feature (France is unseen).

**Model type:** LightGBM binary classifier. Final model R=1 600k (`models/r1_600k`): 255 leaves,
learning rate 0.05, up to 3,000 trees with early stopping (100 rounds) on validation log loss (best
iteration 2,978), min_child_samples 50, reg_lambda 1, seed 42. Fitted on 14,456,015 union candidate
pairs of 600,000 seeded fitting S1s (the 200,000 of fit200k plus 400,000 drawn by `fit_extra.py` with
the same seed-42 recipe; 13.8% positive); positives are candidates in the ground truth, negatives are
all other retrieved candidates. Scored at threshold **0.78**: the validation-tuned threshold is 0.70,
and two public probes of the 200k R=1 model (0.62: 0.940067, 0.70: 0.942136, 0.78: 0.943093) showed a
consistent gain from a stricter threshold; applied to the 600k model it moved the public score from
0.946504 to 0.947135. The 200k R=1 model (`models/r1_full`, 1,372 trees, 4,819,000 pairs) is kept for
comparison.

**Competition at the right scale.** Computed inside a sampled candidate file, `rev_*` see only that
file's S1s (forward fit200k: negative-pair `rev_count` 13.2 inside the file). Recomputed over all 2.2M
train S1s it is 135.7, and the share of wrong pairs with `rev_rank = 0` falls from 68.5% to 10.1%. The
model is trained on whole-population values (over the train union) and test values are computed over
all 1.73M test S1s. Without `rev_*`, the R=1 model scores 0.9527 instead of 0.9531.

**Threshold selection method:** each S2/S3 record goes to the S1 that scores it highest (ties: smaller
S1 id) and is kept if score ≥ t; t is swept from 0.10 to 0.95 in 0.05 steps, then refined in 0.01 steps
around the best, maximizing macro F0.5 after ownership over all 50,000 validation S1s. R=1: t = 0.70.

---

## 5. Results & Error Analysis

### Model lineup (val50k, global ownership)

v1 to v3 use the forward candidates (K = 10 per source, recall ceiling 0.9293); the R=1 models use the
forward ∪ reverse union (ceiling 0.9578).

| model | features | leaves / trees | fit S1s | val F0.5 | t | US | India | pair P / R | singleton acc. | public |
|---|---|---|---|---|---|---|---|---|---|---|
| rules fallback | 3 token-set scores | – | – | 0.6561 | 0.85 | | | | | 0.658 |
| v1 | 43 (rev_* within file) | 31 / 800 | 200k | 0.9256 | 0.71 | 0.9439 | 0.8984 | 0.9794 / 0.8504 | 0.9175 | |
| v1b | 40 (no rev_*) | 31 / 800 | 200k | 0.9262 | 0.68 | 0.9448 | 0.8985 | 0.9813 / 0.8478 | 0.9171 | 0.9164 |
| v2q | 40 (no rev_*) | 255 / 1,648 | 200k | 0.9326 | 0.70 | 0.9491 | 0.9079 | 0.9833 / 0.8602 | 0.9178 | |
| v2cq | 43 (whole-population rev_*) | 255 / 1,661 | 200k | 0.9370 | 0.68 | 0.9529 | 0.9133 | 0.9825 / 0.8763 | 0.9130 | |
| v2cq_fix | 43, leading-zero fix | 255 / 1,503 | 200k | 0.9394 | 0.73 | 0.9561 | 0.9145 | 0.9859 / 0.8739 | 0.9393 | |
| v3 | 43, as v2cq_fix | 255 / 2,554 | 600k | 0.9423 | 0.66 | 0.9578 | 0.9192 | 0.9841 / 0.8844 | 0.9352 | |
| R=1 without rev_* | 41 | 255 / 1,378 | 200k | 0.9527 | 0.69 | 0.9634 | 0.9367 | | | |
| R=1 (`models/r1_full`) | 44 = v2cq_fix + `from_reverse` | 255 / 1,372 | 200k | 0.9531 | 0.70 | 0.9640 | 0.9369 | 0.9843 / 0.9106 | 0.9301 | 0.942136 |
| **R=1 600k** (`models/r1_600k`, **final**) | 44, as R=1 | 255 / 2,978 | 600k | **0.9561** | 0.70 | 0.9658 | 0.9418 | 0.9860 / 0.9142 | 0.9367 | 0.946504 @ 0.70; **0.947135 @ 0.78** |

- **F_0.5 Score (macro):** **0.9561** val (R=1 600k); **0.947135** public at threshold 0.78. Predicting
  no matches scores 0.054; the candidate recall ceiling is 0.9578 (US 0.9777, India 0.9281). Every
  LightGBM model lost about 0.01 from validation to public, and the public order matched the
  validation order.
- **What moved the score:** supervised model over rules +0.27 on the same candidates; capacity (31 → 255
  leaves, early stopping) +0.0064; whole-population competition features +0.0044 (within-file rev_*
  added nothing: v1 0.9256 vs v1b 0.9262); leading-zero house-number fix +0.0024 (true pairs flagged as
  a house-number conflict fell 11,907 → 7,305; singleton accuracy 0.913 → 0.939); 600k instead of 200k
  fitting S1s +0.0029 (v3), mostly recall (0.8739 → 0.8844) and India (0.9145 → 0.9192), with the tuned
  threshold falling from 0.73 to 0.66 (singleton accuracy 0.9393 → 0.9352); **reverse retrieval
  +0.0137** (v2cq_fix → R=1, both on 200k fitting S1s): the ceiling rises 0.9293 → 0.9578 for 1.8% more
  pairs and the model turns most of it into matches (pair recall 0.8739 → 0.9106, India 0.9145 →
  0.9369), the largest gain after the move to a supervised model.
- **Test predictions (R=1):** 5,603,577 matches (3.234 per S1) with 6.144% of S1s empty; France 3.320
  per S1 and 5.531% empty, India 3.151 and 6.608%, US 3.302 and 5.818%, close to the training rates
  (3.46 matches, 5.6% singletons). `checks.py` and the organizer validator with `--check-ids` passed.
- **Validation vs public:** fallback 0.6561 val / 0.658 public; v1b 0.9262 / 0.9164 (−0.01); R=1
  pending.
- **Fewer candidates:** re-scoring v3 on K = 5 per source (9.6 forward candidates/S1 instead of 23.7)
  gives 0.9232 (−0.019); K = 10 was kept.
- **Ownership:** at 50,000 validation S1s it changes R=1 by < 0.0001 (collision rate 0.001%); on the
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
- **Other:** dropping 6 low-signal features cost −0.0011, so all were kept.

**Error analysis** (v2q, t = 0.70, forward candidates, before the leading-zero fix and reverse
retrieval; its findings led to both). Of 173,059 true pairs 24,194 were missed; of 151,394 predicted
pairs 2,529 were wrong.

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
  a short number) were never retrieved forward (reverse retrieval later raised India's ceiling from
  0.8806 to 0.9281); blank candidate addresses (only 22.8% of such true pairs were recalled);
  zero-padded house numbers read as conflicts (38.6% of true pairs flagged with a house-number
  conflict, fixed in v2cq_fix); name words replaced or reordered while the address is truncated.

---

## 6. Conclusion

Rare-token-pair blocking made full-test retrieval take 8 minutes instead of an estimated 22 hours, and
searching in both directions (forward from each S1, reverse from each S2/S3 record) raised the candidate
recall ceiling from 0.929 to 0.958 for 1.8% more candidates on validation (3.1% on test, 24.2 per S1).
A LightGBM matcher
with whole-population competition features, global ownership and a metric-tuned threshold turns that
into 0.9561 val macro F0.5 (US 0.9658, India 0.9418) and 0.947135 on the public leaderboard. The main lessons: the address, not the name,
carries retrieval, and the biggest late gain came from retrieval (reverse search, +0.0137), not the
model; features must be computed at the scale they are used (competition features only helped once
computed over all S1s); and small data bugs (zero-padded house numbers) were worth as much as model
tuning.

**Limitations.** France (15% of test) is unlabelled and unseen in training, so only coverage and
prediction rates can be checked there. All scores are bounded by the candidate recall ceiling (0.9578;
India 0.9281). Validation scores are tuned on the same labels used for early stopping; v1b lost 0.01
from validation to public. Validation has 50,000 S1s competing for targets against 1.73M at test, so
ownership is exercised far less in validation than at test. The final R=1 model was fitted on 200,000
S1s; combining reverse retrieval with v3's 600,000 fitting S1s is untested. The error analysis predates
reverse retrieval.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains `src/` (all pipeline code), `run.py`, `README.md`,
pinned `requirements.txt`, the final R=1 600k bundle (`models/r1_600k/`: `lgbm.txt`, `bundle.json` with the
ordered 44 features and validation threshold; scoring threshold 0.78 is `THRESHOLD` in `run.py`), the
200k R=1 bundle (`models/r1_full/`) and the shared address TF-IDF (`artifacts/addr_tfidf.joblib`,
SHA-256 `7a9d6370acc9eb3e55625c7e8d4a55b81e91f3320a9484f27c5a9098a30249e3`). Entry point: `python -u
run.py all` (or one stage at a time: normalize, candidates, reverse, features, competition,
apply_competition, train, report, score); the scoring model is the single value `MODEL_DIR` in
`run.py`. `src/score_test.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`
and runs the strict checker (`src/checks.py`) and the organizer validator. The README lists every
command, with measured runtime and peak RAM where recorded.

No external data, APIs, gazetteers or pretrained models are used. Libraries: numpy, scipy, pandas,
pyarrow, scikit-learn, joblib, sparse_dot_topn, rapidfuzz, ftfy, anyascii, LightGBM, psutil.

### B. Additional Results

Per-country forward blocking recall at the frozen config: US 0.9621, India 0.8806 (India misses: 4,727
other noise or pool/rerank limits, 2,049 transliteration mismatch, 1,074 shared tokens above the cap,
477 blank target address); with reverse R=1: US 0.9777, India 0.9281. R=1 union sizes: train 53,163,079
pairs (24.09/S1, 943,117 reverse-only), test 41,955,111 (24.22/S1, 1,258,272 reverse-only). Full
experiment log: `EXPERIMENTS.md`; retrieval write-up: `docs/section2_retrieval.md`; results and 20 + 20
error examples: `docs/section5_results.md`.
