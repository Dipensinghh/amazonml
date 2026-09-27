## 5. Results & Error Analysis

All numbers below are **tuned validation scores**: early stopping and threshold selection used the same
validation labels, so they are not untouched test estimates.

**Setup.** Model fitted on `cand_fit200k` (4,732,479 candidate pairs from 200,000 fitting S1s, 13.6%
positive). Evaluated on `cand_val50k` (1,183,283 pairs) over the exact 50,000 validation S1s in
`val50k_s1_ids.parquet`, searched against all train S2/S3 records of the same country (distractors kept).
Macro F0.5 is computed over all 50,000 S1s after global target ownership; the blocking recall ceiling on
this set is 0.9293.

### 5.1 Model lineup

All models are LightGBM binary classifiers on the frozen blocking-v3 candidates (K = 10 per source),
scored on val50k with a single threshold tuned after global target ownership. v1 to v2cq_fix are fitted on
fit200k (200,000 S1s); v3 uses the v2cq_fix recipe on 600,000 fitting S1s.
`rev_*` = the three cross-S1 competition features (`rev_rank`, `rev_count`, `rev_score_gap`).

| model | features | leaves / trees | val F0.5 | t | US | India | pair P / R | singleton acc. | public |
|---|---|---|---|---|---|---|---|---|---|
| rules fallback (A) | 3 lexical scores | – | 0.6561 | 0.85 | | | | | **0.658** |
| v1 | 43 (rev_* per candidate file) | 31 / 800 | 0.9256 | 0.71 | 0.9439 | 0.8984 | 0.9794 / 0.8504 | 0.9175 | |
| v1b | 40 (no rev_*) | 31 / 800 | 0.9262 | 0.68 | 0.9448 | 0.8985 | 0.9813 / 0.8478 | 0.9171 | **0.9164** |
| v2q | 40 (no rev_*) | 255 / 1648 | 0.9326 | 0.70 | 0.9491 | 0.9079 | 0.9833 / 0.8602 | 0.9178 | |
| v2cq | 43 (rev_* over all 2.2M train S1s) | 255 / 1661 | 0.9370 | 0.68 | 0.9529 | 0.9133 | 0.9825 / 0.8763 | 0.9130 | |
| v2cq_fix | 43, leading-zero fix | 255 / 1503 | 0.9394 | 0.73 | 0.9561 | 0.9145 | 0.9859 / 0.8739 | 0.9393 | |
| **v3** | 43, as v2cq_fix; 600k fit S1s | 255 / 2554 | **0.9423** | 0.66 | 0.9578 | 0.9192 | 0.9841 / 0.8844 | – | Submission 3: _pending_ |
| v4r | v3 + reverse-retrieval candidates | | _pending_ | | | | | | |

Predicting no matches scores 0.054 on val50k. The blocking recall ceiling is 0.9293 (US 0.9621, India
0.8806), and every model sits below it.

**What moved the score.**
- **Supervised model over rules: +0.27.** Same candidates, same ownership policy.
- **Capacity (v1b → v2q): +0.0064.** 255 leaves with early stopping at 1,648 trees; at 800 trees and 31
  leaves no run had converged.
- **Competition features at the right scale (v2q → v2cq): +0.0044.** Computed inside one candidate file,
  rev_* describe how many of *that file's* S1s want a target, so their scale changes with file size
  (negative `rev_count` 4.0 on val50k vs 13.2 on fit200k) and they added nothing (v1 0.9256 vs v1b 0.9262).
  Recomputed over all 2.2M train S1s, the scale of the 1.73M-S1 test set, they help; the share of wrong
  pairs with `rev_rank = 0` fell from 68.5% to 10.1%.
- **Leading-zero fix (v2cq → v2cq_fix): +0.0024.** House numbers were compared with their zero padding,
  so `005705` and `5705` counted as a conflict. After stripping leading zeros, true val pairs flagged
  with `num_conflict = 1` fell from 11,907 to 7,305 (−38.6%) against −3.9% for non-matches, and singleton
  accuracy rose from 0.913 to 0.939. The same fix on v2q's 40 features gives 0.9326 → 0.9355.
- **More fitting data (v2cq_fix → v3): +0.0029.** Same recipe on 600,000 instead of 200,000 fitting S1s;
  it gains mostly in recall (0.8739 → 0.8844) and in India (0.9145 → 0.9192), and needs more trees
  (early stop at 2,554 vs 1,503).

**Validation vs public.** The fallback's public score matched validation (0.658 vs 0.6561); v1b lost
about 0.01 (0.9164 vs 0.9262). Likely contributors: France (15% of test S1s, no labels, unseen in
training) and denser competition for targets at test scale.

### 5.2 Other experiments

**Hyperparameter sweep** (D; v1 features, then v1b features, fit200k / val50k):

| config | features | val F0.5 | threshold | avg matches/S1 | best iter |
|---|---|---|---|---|---|
| baseline (lr 0.05, 31 leaves, min_child_samples 50) | 43 | 0.9256 | 0.71 | 3.005 | 800 (cap) |
| num_leaves=63 | 43 | 0.9280 | 0.75 | 2.984 | 800 (cap) |
| num_leaves=127 | 43 | 0.9303 | 0.74 | 3.006 | 800 (cap) |
| min_child_samples=100 | 43 | 0.9257 | 0.69 | 3.023 | 800 (cap) |
| lr 0.03, up to 2000 trees | 43 | 0.9272 | 0.71 | 3.015 | 1999 (cap) |
| drop 6 low-signal features | 37 | 0.9245 | 0.71 | 3.004 | 800 (cap) |
| num_leaves=127, up to 3000 trees | 40 | 0.9323 | 0.71 | 3.017 | 2572 (early stop) |
| num_leaves=255, up to 3000 trees (= v2q) | 40 | 0.9326 | 0.70 | 3.028 | 1648 (early stop) |

**Fewer candidates (K = 5 per source).** Cutting candidates from 23.7 to 9.6 per S1 costs 0.018
(v1: 0.9256 → 0.9074). K = 10 was kept.

**Low-signal features.** Six features that looked uninformative on an early development set
(`name_missing`, `was_non_latin`, `postal_conflict`, `is_s3`, `postal_missing`, `name_alt_token_set`)
cost 0.0011 when removed, so all were kept.

**Global ownership.** It changes the model scores by at most 0.0001 on val50k (e.g. v2cq_fix 0.9394 vs
0.9393 without), because 50,000 validation S1s rarely compete for the same target. On the rules fallback,
whose scores are less confident, it added +0.051 (0.6049 → 0.6561). It stays on for test, where 1.73M
S1s compete.

**Cross-source corroboration (tried, not adopted).** 85% of matched S1s have matches in both S2 and S3,
so a true candidate should resemble one of the S1's candidates from the other source. For each pair
(S1, t) we compared t with the S1's top-5 other-source candidates by final_rank: best name token-set,
best address token-set, best average of the two, and that partner's rank. The signal is real
(`xsrc_best_both` 0.87 on true pairs vs 0.59 on non-matches), but on v2q + leading-zero fix it added only
**+0.0019** (0.9355 → 0.9374): below-threshold misses fell 10.7% (11,153 → 9,959) and pair recall rose
0.0069, while precision moved −0.0006. It cannot reach pairs that blocking never retrieved, which are half
of all misses, so it fell short of the +0.004 adoption bar. Cost would have been small (≈38k pairs/s,
about 18 minutes for the 40.7M test pairs on an 8-core laptop).

### 5.3 Where the errors come from

Error analysis was run on v2q (val F0.5 0.9326, t = 0.70), before the leading-zero fix; its findings led
to that fix. Of 173,059 true pairs, 24,194 are missed; of 151,394 predicted pairs, 2,529 are false merges.

| error type | count | share | notes |
|---|---|---|---|
| Miss: never retrieved by blocking | 12,237 | 50.6% of misses | 68% are India (8,320) |
| Miss: retrieved, scored below t | 11,950 | 49.4% of misses | 41% have a blank candidate address |
| Miss: lost target under ownership | 7 | ~0% | |
| False merge: target belongs to no S1 (distractor) | 1,690 | 66.8% of false merges | |
| False merge: target belongs to another S1 | 839 | 33.2% of false merges | |

- **Retrieval is half the recall loss, and it is mostly India.** India has 40% of true pairs but 68% of
  unretrieved ones; these are typically addresses cut down to common tokens (city, state, a short number).
- **Blank candidate addresses are hard both ways.** Only 22.8% of true pairs with a blank candidate
  address are recalled, and 17.4% of false merges have one: with no address, a same-name S1 elsewhere is
  indistinguishable.
- **False merges are mostly "twins".** 60% of false merges have near-identical names
  (token-set ≥ 0.9), usually at the same street with the house or flat number changed. These are
  deliberately hard distractors. 9.6% are the opposite case: identical address, unrelated name
  (a different business in the same building).
- **Actionable feature bug: zero-padded house numbers.** 38.6% of true pairs flagged with
  `num_conflict = 1` differ only by leading zeros (`5705` vs `005705`), against 3.9% of non-matches.
  Comparing numeric tokens with leading zeros stripped clears the false conflict on 4,602 true pairs
  (2.9% of retrieved true pairs). The blocker already stripped them but `features.py` did not; this was
  fixed in v2cq_fix (section 5.1).

### 5.4 Twenty false merges (random sample, seed 42, t = 0.70)

| # | ctry | p | S1 | wrongly matched record | cause |
|---|---|---|---|---|---|
| 1 | IN | 0.95 | Agile Hospital · Flat 468, Road No 87 Jubilee Hills | Agile Hospital Pvt Ltd · Flat A-472, same road (Telugu state name) | Twin: same name, neighbouring flat |
| 2 | IN | 0.81 | Mhm College Pvt Ltd · Rajender Singh Vill-Hera Heri, Pataudi | Mhm Limited Center · #73, same village | Same address, only the generic first token shared |
| 3 | US | 0.85 | Wilson Express Canada Inc · 26 Smith Terrace, Staten Island | PLABHNG · 26 Smith Ter, Staten Island | Identical address, unrelated name (other S1's record) |
| 4 | US | 0.77 | Rollins, Nerta, Esq. · 341 Townsend St, Fitchburg | Rollins, Nerta, Esq. · (blank address) | Blank address: same name belongs to another S1 |
| 5 | US | 0.76 | Spinosa Select Industrial LP · 200 Market St Unit 202 | Liberty · same address and unit | Shared unit, different business |
| 6 | US | 0.79 | Pediatric Dental Care Associates · 113 Brownlea Dr | Pediatric Dental Care Associaregse · 126 Brownlea Dr | Twin: same street, different number despite conflict flag |
| 7 | IN | 0.82 | Integrated Industries Group Pvt Ltd · Flat 102, Mayekar Park | Integrated Industries Group Holdings · Flat 12, same building | Twin: one flat apart |
| 8 | IN | 0.85 | Vkr (India) Technologies · Plot A-787/2/2, TTC Industrial Area | Sp-Academy · same plot | Shared plot, unrelated name (other S1's record) |
| 9 | US | 0.88 | Grabner and Brumfield House Partners · 4830 137th St | … Partners [LLC] · 4834b 137rd St | Twin: neighbouring house number |
| 10 | US | 0.79 | Womens Health Beacon Physicians · 21808 Colton Point Rd | Womens Beacon Physicians LLC Services · (blank) | Blank address, near-same name, rank 20 |
| 11 | IN | 0.72 | Sree Finance · S.No 238/239, Magarpatta Rd | Sree Finance LLP · S.No 238/240, same road | Twin: survey number off by one |
| 12 | US | 0.94 | Lise Dawson Vanguard Index PLLC · 2200-2300 Beaumont Ln | Phelps Dawson Vanguard Index PLLC · same | Same address, only the first name differs |
| 13 | US | 0.98 | Secure Coastal Pantages Group · 2412 San Augustine Ln | Secure Cóastal Pantages Group · 2423 same lane | Twin with accent noise; high confidence |
| 14 | IN | 0.78 | Hera Chits Ltd · 11/3F Canal Circular Rd | Hera Clsh LLP · Door 11/24F, same road | Twin: similar name, adjacent door |
| 15 | US | 0.92 | Mavue Capital Partners LLC · 1232 Todd Ct | Mavue Pártners Capital LLC · (blank) | Blank address, reordered name of another S1 |
| 16 | US | 0.98 | Juniper L.L.C. · 2998 3430, St George | juniper llc · 3019 3430, Saint George | Common name, nearby number |
| 17 | US | 0.93 | Glyphelle Compania LLC · 20249 Glenbrook Path | Glyfelle Compania LLC · same address | Indistinguishable from a true match (ph/f spelling) |
| 18 | IN | 0.97 | Shiksha & Associates · Bldg D-03, Plot ML-15 | Shiksha & Associates Ltd · Bldg D-16, same plot | Twin: different building in the same estate |
| 19 | IN | 0.99 | Taravi Motels Pvt Ltd · H.No 03, Samera Gate, Bhopal | Taravy Pvt Ltd Center · same | Same house, near-same name; very high confidence |
| 20 | US | 0.98 | Lyrial Prime Partners · 5011 Grande Shops Ave | Lyrix Prime-Partners · 501 same avenue | Twin: truncated number, similar name |

### 5.5 Twenty misses (random sample, seed 42)

| # | ctry | type | p | S1 | missed true record | cause |
|---|---|---|---|---|---|---|
| 1 | US | below t | 0.65 | Selders's Prime Enterprises LLC · 301 3rd St, Peru IN | exact name · (blank address) | Blank address: exact name still scored just under t |
| 2 | IN | below t | 0.64 | Indram Electromech Pvt Ltd · Goregaon, Mumbai | Indram Electromech Private · (blank) | Blank address |
| 3 | US | below t | 0.32 | Dermatology Partners LLC · 5705 Summerwood Ln | … Authority · 005705 Summerwood Ln | Zero-padded number read as a conflict |
| 4 | US | below t | 0.01 | Marget's Couture · 49 Woodrow Ave | Marget's Services · (blank) | Blank address and name suffix replaced |
| 5 | US | not retrieved | – | Cozy Ice Cream L.L.C. · 407 Delaware Ave, Wilmington | COZY ICE CREAM LLC · Delaware Ave, Wilmington | House number dropped; only common tokens left |
| 6 | US | below t | 0.34 | Montes, Arndt & Jurgens Inc · 312 Poplar Rd | Services Montes, Arndt Jurgens · Poplar Rd, Boken Bow | Missing number plus city typo |
| 7 | US | below t | 0.52 | Atlantic Tbk Inc · 1228 Byron Dr | Xylotavo · 228 Byron Dr | Unrelated trade name and truncated number |
| 8 | US | not retrieved | – | Crichton & Delacruz LLP · 18508 16th Ln | Crichton & Deatgcruz LLP · 16th Ln | Number dropped, name typo |
| 9 | US | below t | 0.39 | Georges Crystal Transition · 681 Sitka Deer Ct | same · 68 Sitka Deer Ct | Truncated number read as a conflict |
| 10 | IN | below t | 0.02 | Caravan Brothers Pvt Ltd · Plot R 257, Rabale | Dr Caravan Pvt Ltd Center · (blank) | Blank address, name changed |
| 11 | US | not retrieved | – | Pinnacle Adr, LLC · 1896 Norton St | Adr -- Pinnacle · Norton St | Reordered name, number dropped |
| 12 | IN | below t | 0.07 | Almond Vidyalaya · A-63 Shantam Towers | Smt Almond Services · same address | Name word replaced; exact address not enough |
| 13 | US | below t | 0.28 | Modern Guaranty Corp · 126 Westway | Modern Guaranty · 26 Westway | Truncated number plus city typo |
| 14 | US | below t | 0.28 | Little Tavern P.C. · 3240 Flame Dr | Little (Tavern) · 3191 Flame Dr | Number fully different; looks like a twin |
| 15 | US | below t | 0.13 | Y/D Doubleline-Kokomo · 1213 Monroe St | Y/D (Services) · ##1213 Monroe St | Name mostly replaced; exact address |
| 16 | IN | not retrieved | – | Sky Swastik Properties Pvt Ltd · Survey 338, Rangpar | Sky Swastik Pvt Ltd Services · 38, GJ, Rangpar | Address cut to a partial number and village |
| 17 | US | not retrieved | – | Metro Endocrinology · 13527 Madison Ave | Metro 5ervices · (blank) | Blank address and name replaced |
| 18 | IN | not retrieved | – | Jaipur Services Pvt Ltd · T-23, Ashiana Manglam | Jaipur Services-Private Limited · T-23, Jaipur | Address cut to common tokens |
| 19 | IN | below t | 0.03 | Radha Hospital Limited · Patna | M/s Radha Hospital · (blank) | Blank address: exact core name, very low score |
| 20 | IN | not retrieved | – | Jain International LLP · No. 77, Ramanagara | International Jain Jain · No. 77, Ramanagara | Reordered, duplicated name; partial address |

### 5.6 Follow-up on the error analysis

1. **Zero-padded house numbers — fixed** (v2cq → v2cq_fix, +0.0024; section 5.1).
2. **Retrieval, mostly India — the largest remaining gap.** Half of all misses were never retrieved, and
   68% of those are India. A reverse-retrieval line (v4r: searching from targets back to S1s) raises the
   candidate recall ceiling from 0.929 to 0.958 in validation; its model result is pending.
3. **Blank-address pairs.** Only 22.8% of true pairs with a blank candidate address are recalled.
   Cross-source corroboration was aimed partly at these and gave +0.0019 (section 5.2).
4. **Twins.** Most false merges are same-name records at a neighbouring house number; a numeric-distance
   feature between house numbers is the untested next step.

### 5.7 Limitations

- Threshold and early stopping use the same validation labels, so all validation scores are tuned
  scores; v1b lost about 0.01 from validation to the public leaderboard.
- Validation has 50,000 S1s competing for targets against 1.73M at test, so ownership is tested far less
  than it will be used.
- France is unlabelled; only coverage and score distributions can be checked there.
- The recall ceiling from blocking (0.9293; India 0.8806) bounds every result above.
