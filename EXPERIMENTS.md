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
