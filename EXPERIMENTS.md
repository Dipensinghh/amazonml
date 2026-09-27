# Experiments

Validation protocol: 20% of train S1 ids held out (seed 42, `artifacts/split_train.parquet`).
Blocking studies use a fixed random sample of 50,000 val S1s (seed 42) searched against ALL
train S2/S3 rows of the same country (distractors kept). Recall = share of GT (S1, match)
pairs present in the candidate set. Machine: 16 GB RAM, 20 logical CPUs, Windows.

---

## 2026-09-27 — Stage 1: normalization (`src/normalize.py`)

- ftfy + NFKC + anyascii on distinct non-ASCII values only; the rest is vectorized in pyarrow.
- Fields: name_norm / name_core (legal suffixes removed) / name_alt_core (after dba/fka/aka) /
  name_acronym / name_sorted / name_phon; addr_norm / addr_phon / addr_nums / addr_postal.
- Runtime all train (12.5M rows): ~8 min, peak RAM 3.9 GB.

**Transliteration check** (551,240 matched pairs whose S2/S3 name is non-Latin = 18% of India pairs):
the non-Latin names are *transliterations* of English business names into Indic scripts
(परफेक्ट फूड्स → "prphekt phuds" = Perfect Foods), not translations. anyascii recovers the
sound, but spelling drifts systematically (ph/f, soft g→j, anusvara m, dropped vowels,
"praivet/piraivet limitet"). Added a script-agnostic phonetic skeleton `name_phon`.

Mean rapidfuzz token_set_ratio, S1 vs matched S2/S3:

| group | pairs | name_norm | name_core | ratio(core) | name_phon |
|---|---:|---:|---:|---:|---:|
| India non-Latin | 551,240 | 73.5 | 66.7 | 69.9 | **93.1** |
| India Latin-Latin | 2,508,603 | 91.2 | 90.9 | 87.9 | 91.0 |
| all Latin-Latin | 7,087,125 | 92.7 | 91.6 | 88.4 | 91.7 |

Caveat: skeletons are shorter, so non-matches also score higher on name_phon — use it as an
extra feature, not a replacement.

---

## 2026-09-27 — Stage 2: blocking v2 (`src/blocking.py`, 50k val sample)

Channels per country × source: A = char_wb 3-gram TF-IDF on name_core (max_df 0.2);
P = same on name_phon; B = word TF-IDF on addr_phon (max_df 0.1); C = exact acronym
(bucket cap 10). Top-20 once per channel, k sliced. GT pairs crossing countries: **0.000%**
(per-country blocking loses nothing).

Recall @ avg candidates per S1 (S2+S3 combined; k = per source per channel), overall:

| k | A | P | A+P | A+B | A+P+B | A+B+C | A+P+B+C |
|---:|---|---|---|---|---|---|---|
| 3 | 0.467 @ 6.0 | 0.427 @ 6.0 | 0.484 @ 7.8 | 0.858 @ 10.6 | 0.863 @ 12.3 | 0.858 @ 12.8 | 0.863 @ 14.5 |
| 5 | 0.535 @ 10.0 | 0.494 @ 10.0 | 0.554 @ 13.5 | 0.914 @ 18.2 | 0.917 @ 21.7 | 0.914 @ 20.4 | 0.917 @ 23.9 |
| 10 | 0.598 @ 20.0 | 0.556 @ 20.0 | 0.620 @ 28.5 | **0.943 @ 37.8** | 0.947 @ 46.2 | 0.943 @ 40.0 | 0.947 @ 48.4 |
| 20 | 0.650 @ 40.0 | 0.603 @ 40.0 | 0.676 @ 59.2 | 0.960 @ 77.5 | 0.963 @ 96.7 | 0.960 @ 79.7 | 0.963 @ 98.8 |

Per country (A+B): India 0.840 / 0.893 / 0.923 / 0.942 and US 0.871 / 0.927 / 0.957 / 0.972
at k = 3 / 5 / 10 / 20.

Channel contribution at k=20 (share of GT pairs found by that channel only):
A 0.72%, P 0.32%, **B 28.67%**, C 0.002%; missed by all 3.71%.
- B over A: +34.5 pp recall at k=10 (0.598 → 0.943) for +17.8 cands/S1.
- C over A+B: ~0 (+0.0001) for +2.2 cands/S1 → drop C.
- P over A+B: +0.3 pp at k=10 for +8.4 cands/S1.

Timing (10 worker processes; sparse_dot_topn threading does not scale here: 122 q/s at 1
thread vs 161 at 20; process pool is memory-bandwidth bound):

| channel | fit | query (50k S1) | q/s | extrapolated full test (1.73M S1) |
|---|---:|---:|---:|---:|
| A | 112 s | 1148 s | 87 | ~665 min |
| P | 73 s | 790 s | 127 | ~457 min |
| B | 50 s | 328 s | 305 | ~190 min |
| C | 6 s | – | – | ~0 min |

Total runtime 2588 s, peak RAM 6.4 GB (main process).

**Findings.** Names are weakly identifying in this data (many entities share near-identical
names); the address is the main blocking signal. TF-IDF top-k via sparse matmul is far too
slow for test (~22 h all channels). Next: cheaper candidate generation driven by rare
address tokens (e.g. house number + street/locality keys) with name similarity used for
ranking, targeting A+B-level recall at a fraction of the cost.

Earlier attempt (blocking v1, full 441k val, sparse_dot_topn threads): stopped after 45 min;
India/S2/A alone took 1162 s (152 q/s).

---

## 2026-09-27 — Stage 2: blocking v3, rare-token pairs (`src/blocking.py`) — FROZEN

Replaces the A/P/B/C trigram search. Per country × source:
- tokens (hashed, 2^26 buckets): address letter/digit runs (leading zeros stripped), number
  prefixes truncated by one digit ("1125"→"112#": house numbers are perturbed), and all
  unordered pairs of non-common address words/numbers (common = df > 1% of targets);
  name words, unordered name-word pairs, concatenated name (+ web affixes stripped).
- drop tokens with target df > cap; score = Σ IDF over shared kept tokens (address 1.0,
  name 0.5) via one sparse matmul (weighted query × binary target), top-30 pool per source.
- blank-address targets get their own small name-only index, quota KB = 2 per source.
- rerank the pool by score + α·token_set_ratio/100 (max over name_core / name_phon; pairs only).

Diagnosis that drove it: single words are too common (street/city df 100–5000, numbers
perturbed); pairs are near-unique. Before the blank-address index, 53% of US / 23% of India
misses were blank-address targets. Phonetic tokens for everything were worse (0.894 vs 0.911 @ 20
on 2k) — collisions outweigh transliteration gains.

50k val sample, overall recall @ avg cands/S1 (K per source + up to 2 blank per source):

| cap | α | K=5 | K=10 | K=15 |
|---:|---:|---|---|---|
| 200 | 0 | 0.8955 @ 13.7 | 0.9214 @ 23.7 | 0.9315 @ 33.6 |
| 200 | 3 | 0.9060 @ 13.7 | 0.9277 @ 23.7 | 0.9363 @ 33.6 |
| 200 | 6 | 0.9082 @ 13.7 | **0.9293 @ 23.7** | 0.9373 @ 33.6 |
| 500 | 6 | 0.9073 @ 13.9 | 0.9286 @ 23.9 | 0.9365 @ 33.9 |
| 2000 | 6 | 0.9069 @ 14.0 | 0.9287 @ 24.0 | 0.9364 @ 33.9 |

Per country at the frozen config (cap 200, α 6, K 10): India 0.8806, US 0.9621 @ ~23.7.
Pool ceiling (30/source): 0.9475 @ 69.3. Cap barely matters; 200 is best and cheapest.

Runtime on 50k: 231 s total for 3 caps (target-side fixed 115 s; query+sim ~20 s per cap,
~10k q/s); peak RAM 7.1 GB. Extrapolated full test: **~12 min** (vs ~22 h for v2).

**Frozen:** cap 200, α 6, K 10, KB 2 → recall 0.929 @ 23.7 cands/S1.
Candidate files (schema: s1_id, cand_id, source, blank_addr_index, tok_score, name_sim, final_rank):
- `artifacts/cand_val50k.parquet` — from the eval pool; recall 0.9293 @ 23.67.
- `artifacts/cand_fit200k.parquet` — 200k non-val train S1 (seed 42); recall 0.9306 @ 23.66; 205 s.
- `artifacts/cand_test.parquet` — all 1,732,544 test S1: 40.7M pairs, 23.49 cands/S1
  (France 22.68, India 23.53, US 23.76); 1 S1 with no candidates; no duplicates.
  **Actual runtime 471 s (7.9 min)**, peak RAM 7.4 GB — France handled with no country-specific code.

### Experiment: non-Latin phonetic index (`--nl-index`, off in the frozen config)

Separate index per country × source over target rows whose original name or address was
non-Latin, tokenized from addr_phon/name_phon, queried with S1 phonetic tokens; quota KN = 2
per source, deduped against the main pool. India targets flagged: S2 836k / S3 686k (~40%,
mixed-script addresses count), so the index is not small.

| config (cap 200, α 6, K 10) | India | overall | test time |
|---|---|---|---|
| frozen | 0.8806 @ 23.5 | 0.9293 @ 23.7 | ~12 min est. / 7.9 min actual |
| + nl-index | 0.8836 @ 24.8 | 0.9305 @ 24.2 | ~18 min est. |

+0.3 pp India / +0.1 pp overall for +0.5 cands/S1 — marginal; not adopted yet.
Also noted: K=5, α 6 with nl-index gives 0.9106 @ 14.4 cands/S1 (≥ 0.91 with ~40% fewer
candidates) — an option if candidate-set size weighs heavily in judging.
(This run overwrote `artifacts/val_blocking_pool_cap200.parquet`; `cand_val50k.parquet` was
built from the frozen-config pool before it.)

## 2026-09-27 — Stage 3: features (`src/features.py`, `src/dev_candidates.py`) — B

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

### v3 adapter — features over A's real candidates (2026-09-27)

Regenerated A's candidate files locally with the frozen config (A's Drive copies never reached
this machine): `cand_val50k` 1,183,283 pairs @ 23.67 cands/S1, `cand_fit200k` 4,732,479 pairs @
23.66 — both matching A's reported 23.7.

| set | pairs | features | positives | throughput | peak RAM |
|-----|-------|----------|-----------|-----------|----------|
| cand_val50k | 1,183,283 | 43 | 160,824 (13.59%) | 21,543 /s | 3.1 GB |
| cand_fit200k | 4,732,479 | 43 | 644,259 (13.61%) | 21,616 /s | 5.3 GB |

**43 features, not 45.** v3 has no A/P/B/C channels, so `ch_*` + `n_channels` (5) are replaced by
`from_blank_index`, `retr_tok_score`, `retr_name_sim` (3). 31 base + 12 retrieval = 43.

Strongest separation is now the retrieval and competition block, ahead of every string feature:
`retr_score` +99.6, `score_gap_best` -94.5, `retr_rank` -10.2, then `rev_score_gap`, `rev_count`,
`rev_rank`. On fit200k a true match sits at `rev_rank` 0.002 against 5.62 for a non-match — i.e.
a true pair is nearly always the top bidder for its target. Best string features are
`acronym_match` +0.543 and `addr_norm_tfidf` +0.490.

Address TF-IDF refitted over all 6 normalized sources (587,799 sampled addresses, vocab 18,776).
French addresses now produce 20-24 nonzero trigrams where they previously scored all-zero.

**Scale caveat on the competition features.** `rev_*` is computed within one candidate set, so it
is not comparable across sets: negatives average `rev_count` 4.0 on val50k (50k S1s) against 13.2
on fit200k (200k S1s). At test scale (1.73M S1s) it will be larger again, so a model trained on
fit200k meets a shifted distribution at inference. `rev_rank` and `rev_score_gap` are more robust
than the raw `rev_count`.

---

## 2026-09-27 — Role C independent full-test reproduction (macOS)

Machine: 36 GiB RAM, 14 logical CPUs, Python 3.14.7. Command:
`python -u src/blocking.py --produce test --jobs 8`.

- Normalized test rows: S1 1,732,544; S2 4,887,273; S3 5,082,316. Peak RAM 5.1 GB.
- Frozen blocker: 40,696,839 pairs in 433 s, peak RAM 10.6 GB, 23.4896 candidates/S1.
- Country means: US 23.7616; India 23.5262; France 22.6806.
- Coverage: 1,732,543 / 1,732,544 S1s; the single zero-candidate query is in France.
- Exact streaming audit passed: expected schema, no nulls, finite scores, `name_sim` in
  [0,100], source/prefix agreement, contiguous ranks, no duplicate target within an S1, and
  query groups in normalized S1 order.
- B integration smoke test on 100,000 real pairs passed exact pair equality and produced 43
  finite feature columns at 23,145 pairs/s (1.4 GB peak RAM).
- Full B feature run: 40,696,839 rows x 43 features, 1,714.4 s feature compute
  (23,738 pairs/s), about 31 minutes end-to-end, 11.6 GB peak RAM. The 2.08 GiB Parquet passed
  exact candidate/feature pair and row-order equality, no-null, and finite-value checks.

B's current feature builder retains the full candidate DataFrame, referenced corpus rows, and
all output chunks until the final write, but the measured full run fits safely on the 36 GiB
Mac. A streaming rewrite is not needed for this submission.

## Model

| date | config | val F0.5 | US | India | France | threshold | runtime | notes |
|------|--------|----------|----|-------|--------|-----------|---------|-------|
| 2026-09-27 | v1: LGBM 43 feats, fit200k → val50k | 0.9256 | 0.9439 | 0.8984 | n/a | 0.71 | 2.4 min, 5.2 GB | 800 trees, no early stop |
| 2026-09-27 | v1b: v1 minus rev_rank/rev_count/rev_score_gap (40) | **0.9262** | 0.9448 | 0.8985 | n/a | 0.68 | 2.8 min, 4.5 GB | 800 trees, no early stop |

### D: scoring, split and rules fallback (E00-E01)

| Run | Question | Setup | Result | Decision |
|---|---|---|---|---|
| E00 | Are scoring/split correct? | toy asserts (0.714 example, singleton rules); split = team split artifacts/split_train.parquet (blocking.make_split, seed 42, default_rng, round) = 441,364 val S1s (my earlier RandomState/ceil split dropped to stay consistent with C's candidates) | asserts pass; all-empty baseline macro F0.5 ~= 0.056 on val | metric + split frozen |
| E01 (`src/fallback_e01.py`) | Does a rules fallback work? | 2,000 random val S1s (seed 42) vs ALL train S2/S3 of same country+source; word TF-IDF (max_df 0.02) name top-10 + address top-10 union; score = w*name_cos + (1-w)*addr_cos (addr blank -> name only); global ownership; one threshold | pair recall 0.930, 37.9 cands/S1 (75,862 pairs); best w_name 0.2 -> t 0.73, **macro F0.5 0.662** (w 0.5: 0.646, w 0.0: 0.619); US 0.706 / India 0.574; ownership effect +0.001 (collision rate 0.2%, thin competition at 2k S1s); runtime 167 s, peak RSS 2.7 GB | fallback ready as a floor; LightGBM must beat 0.662 on the same candidates |

Limits of E01: 2,000-S1 sample (noisy, thin ownership competition); own light normalisation (not B's); word-level retrieval (not C's trigram channels); no acronym channel; names made only of common tokens get no name candidates.

### D: first LightGBM on B's dev feature sets (E03 preview, NOT a real result)

Fit: `dev_candidates_fit_features.parquet` (468,900 pairs, 20,000 fitting S1s). Validation: `dev_candidates_features.parquet` (117,287 pairs, 5,000 val S1s; 14.7% positive; 100% true-match recall; negatives are token-overlap, only 81 random-pool). 31 base features, playbook params, early stop at 727 trees (val log loss 0.0130).

| model (same 117,287 val pairs) | macro F0.5 | threshold |
|---|---|---|
| all-empty | 0.060 | - |
| rules: 0.4*name_core_jw + 0.6*addr_norm_tfidf | 0.935 | 0.59 |
| LightGBM | **0.981** | 0.73 |

LightGBM beats the rules by +0.047 on identical pairs. Ownership effect: 0 (collision rate 0.0 at 5,000 S1s). Top gain: `addr_phon_token_set`, `addr_norm_token_set` (address dominates), then `name_norm_token_sort`, address TF-IDF, `num_conflict`.
Caveat: not comparable to E01 (0.662) - those were real top-k candidates with 93% recall; this set has 100% recall and easier negatives. Threshold 0.73 is not usable. Redo on C's real candidates.

### D: LightGBM v1 / v1b on the frozen blocking candidates (E04)

Data: `cand_fit200k_features.parquet` (4,732,479 pairs, 200k fitting S1, 13.6% positive) and
`cand_val50k_features.parquet` (1,183,283 pairs, 13.6% positive), built by `features.py --labels`
(43 features via `feature_columns()`, no NaN/inf, rows = input rows; `name_missing` constant in
both; cached `addr_tfidf.joblib` sha256 7a9d6370…249e3, not refit). Scope: exactly
`artifacts/val50k_s1_ids.parquet` (50,000 S1, 2,702 singletons, 0 without candidates;
blocking pair-recall ceiling 0.929). Params: 800 trees, lr 0.05, 31 leaves,
min_child_samples 50, early stop 50 on val log loss. Models: `models/v1`, `models/v1b`
(lgbm.txt + bundle.json with ordered features, best_iteration, threshold; val_report.json).

| metric (val50k, global ownership) | v1 (43) | v1b (40, no rev_*) | fallback |
|---|---|---|---|
| macro F0.5 with ownership | 0.9256 | **0.9262** | 0.6561 |
| macro F0.5 without ownership | 0.9256 | 0.9261 | |
| threshold | 0.71 | 0.68 | |
| US / India | 0.9439 / 0.8984 | 0.9448 / 0.8985 | |
| pair precision / recall | 0.9794 / 0.8504 | 0.9813 / 0.8478 | |
| singleton accuracy | 0.9175 (223 false merges / 2,702) | 0.9171 (224) | |
| avg predicted matches per S1 | 3.01 | 2.99 | (gold 3.46) |
| val log loss @ 800 | 0.03587 | 0.03465 | |
| K=5 re-score (9.6 cands/S1) | 0.9074 | 0.9078 | |
| ownership collision rate | 0.005% | 0.012% | |

Top-15 gain (v1): addr_phon_token_set, retr_rank, name_norm_partial, num_agree, addr_phon_tfidf,
name_norm_jw, name_norm_token_sort, name_phon_jw, addr_norm_token_set, name_core_partial,
rank_frac, addr_norm_tfidf, num_conflict, name_core_ratio, name_phon_token_sort.
v1b: same set, near-identical order (name_norm_jw and addr_phon_tfidf swap; name_core_ratio above num_conflict).

Findings: LightGBM +0.27 over the fallback. Dropping the rev_* competition features costs
nothing (v1b ≥ v1): computed within a 200k/50k candidate file they see much thinner competition
than on the full test set, so v1b is the safer choice until they are computed over all S1s
(cand_trainall). Neither run early-stopped — both still improving at 800 trees; more trees are
likely worth a little. Precision (0.98) far exceeds recall (0.85, ceiling 0.93): threshold
already favours precision as F0.5 wants. India trails US by ~0.046. Cutting to K=5 per source
costs ~0.018 F0.5 for 60% fewer candidates.

## Submissions

| date | leaderboard F0.5 | local val F0.5 | avg cands/S1 | what changed |
|------|------------------|----------------|--------------|--------------|

---

## 2026-09-27 — India frozen-blocker miss study (Role C)

Exact India subset of the seeded 50k validation scope: 20,086 S1s, 69,685 labelled pairs.
Frozen cap-200 baseline: 0.8805 recall at 23.51 candidates/S1; 8,327 true pairs missed.

Mutually exclusive miss classification:

| Cause | Missed pairs |
|---|---:|
| Other noise / pool or rerank limit | 4,727 |
| Transliteration mismatch | 2,049 |
| Generic shared tokens dropped by cap | 1,074 |
| Blank target address | 477 |

Controlled alternatives all retained K=10 and stayed near 24 candidates/S1:

| Configuration | Recall | Candidates/S1 | Change |
|---|---:|---:|---:|
| Frozen, cap 200 | 0.8805 | 23.51 | — |
| India cap 500 | 0.8803 | 23.77 | -0.02 pp |
| India cap 1000 | 0.8799 | 23.88 | -0.06 pp |
| ph/f, v/w, aa/a, doubled-letter variants | 0.8690 | 23.78 | -1.15 pp |
| Phonetic-address token pairs | 0.8563 | 23.53 | -2.42 pp |

No variant met the predeclared +2 point gate. Keep the frozen blocker unchanged. Full results
and examples are in `artifacts/india_retrieval_report.json`; runtime 344 s, peak RAM 11.4 GB.

## 2026-09-27 — Chunked test scorer smoke test (Role C)

`src/score_test.py` was tested end to end with a tiny LightGBM bundle and contested target:
chunked scoring preserved all pair rows and order, global ownership chose the higher score,
atomic score output completed, both TSVs were written, `checks.py` passed, and the organizer
validator passed with `--check-ids`.

The full test feature `rev_*` columns were also independently recomputed for 100 contested
targets (5,263 rows across 3,514 S1s); rank, contender count, and score gap matched exactly,
confirming that competition was computed globally before feature batching.
