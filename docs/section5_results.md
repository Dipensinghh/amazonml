## 5. Results & Error Analysis

All numbers below are **tuned validation scores**: early stopping and threshold selection used the same
validation labels, so they are not untouched test estimates.

**Setup.** Model fitted on `cand_fit200k` (4,732,479 candidate pairs from 200,000 fitting S1s, 13.6%
positive). Evaluated on `cand_val50k` (1,183,283 pairs) over the exact 50,000 validation S1s in
`val50k_s1_ids.parquet`, searched against all train S2/S3 records of the same country (distractors kept).
Macro F0.5 is computed over all 50,000 S1s after global target ownership; the blocking recall ceiling on
this set is 0.9293.

### 5.1 Headline (best config: LightGBM, 40 features, `num_leaves=255`, 1,648 trees, t = 0.70)

| metric | value |
|---|---|
| **Macro F0.5 (validation)** | **0.9326** |
| US / India | 0.9491 (29,914 S1s) / 0.9079 (20,086 S1s) |
| Pair precision / pair recall (after ownership) | 0.9833 / 0.8602 |
| Blocking recall ceiling | 0.9293 |
| Singleton accuracy (2,702 true singletons) | 0.9178 |
| Avg predicted matches per S1 | 3.03 (true: 3.46) |
| Rules fallback, same candidates (A) | 0.6561 val / 0.658 public leaderboard |
| Predict nothing | 0.054 |

The fallback's public score (0.658) matched its validation score (0.6561), which supports using this
validation set to choose the model. France has no labels, so no French accuracy is reported.

### 5.2 Model selection

**Sweep 1** (43 features, up to 800 trees, early stopping 50):

| config | val F0.5 | threshold | avg matches/S1 | best iter |
|---|---|---|---|---|
| baseline (lr 0.05, 31 leaves, min_child_samples 50) | 0.9256 | 0.71 | 3.005 | 800 (cap) |
| num_leaves=63 | 0.9280 | 0.75 | 2.984 | 800 (cap) |
| num_leaves=127 | 0.9303 | 0.74 | 3.006 | 800 (cap) |
| min_child_samples=100 | 0.9257 | 0.69 | 3.023 | 800 (cap) |
| lr 0.03, up to 2000 trees | 0.9272 | 0.71 | 3.015 | 1999 (cap) |
| drop 6 low-signal features (37 left) | 0.9245 | 0.71 | 3.004 | 800 (cap) |
| K=5 candidates (9.6 vs 23.7 per S1) | 0.9074 | 0.71 | 2.822 | 800 (cap) |

**Sweep 2** (40 features, up to 3000 trees, early stopping 100):

| config | val F0.5 | threshold | avg matches/S1 | best iter |
|---|---|---|---|---|
| num_leaves=127 | 0.9323 | 0.71 | 3.017 | 2572 (early stop) |
| **num_leaves=255** | **0.9326** | 0.70 | 3.028 | 1648 (early stop) |

**v1 vs v1b.** v1 uses all 43 features, including three cross-S1 competition features (`rev_rank`,
`rev_count`, `rev_score_gap`); v1b drops them (40 features). These features are computed within one
candidate set, so their scale depends on how many S1s compete: the mean `rev_count` of negatives is 4.0 on
val50k, 13.2 on fit200k, and would be larger again on the 1.73M-S1 test set. A model trained on one scale
meets a shifted distribution at inference.

| | features | best val F0.5 |
|---|---|---|
| v1 (num_leaves=127, 800 trees) | 43 | 0.9303 |
| v1b (num_leaves=127, 2572 trees) | 40 | 0.9323 |
| v1b (num_leaves=255, 1648 trees) | 40 | 0.9326 |

v1b is at least as good on validation. Tree count changed at the same time, so the comparison does not
isolate the effect of the rev_* features; it shows that removing the scale-dependent features costs
nothing measurable, which makes v1b the safer choice for test.

**Other findings.**
- **Capacity helped:** more leaves and more trees gave +0.007 over the baseline; no sweep-1 run reached
  early stopping.
- **Low-signal features are not dead:** the six features that looked uninformative on an early dev set
  (`name_missing`, `was_non_latin`, `postal_conflict`, `is_s3`, `postal_missing`, `name_alt_token_set`)
  cost 0.0011 when removed. All were kept.
- **K=5 vs K=10:** cutting candidates from 23.7 to 9.6 per S1 costs 0.018 F0.5. The team kept K=10.
- **Global ownership:** no measurable effect at 50,000 S1s (identical F0.5 with and without, 7 misses
  caused by it). On A's rules fallback it added +0.051 (0.6049 → 0.6561), because weaker scores collide more.
  Competition is denser at test scale, so it stays on.

### 5.3 Where the errors come from

Of 173,059 true pairs, 24,194 are missed; of 151,394 predicted pairs, 2,529 are false merges.

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
  Comparing numeric tokens with leading zeros stripped would clear the false conflict on 4,602 true
  pairs (2.9% of retrieved true pairs). Blocking v3 already strips them; `features.py` does not.

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

### 5.6 What to fix next, in order of expected gain

1. **Retrieval for India** (half of all misses, two-thirds from India): a channel for short, truncated
   addresses — e.g. name tokens combined with city — would recover pairs the model never sees.
2. **Zero-padded numbers in `features.py`**: strip leading zeros before `num_agree` / `num_conflict`.
   Cheap, and it removes a false conflict from 4,602 true pairs.
3. **Blank-address pairs**: only 22.8% are recalled. A separate threshold, or features comparing the
   candidate's name with the other candidates of the same S1, could help without adding false merges.
4. **Twins**: most false merges are same-name records at a neighbouring number. A feature for the
   numeric distance between house numbers (rather than equal / not equal) targets them directly.

### 5.7 Limitations

- Threshold and early stopping use the same validation labels, so 0.9326 is a tuned score.
- Validation has 50,000 S1s competing for targets against 1.73M at test, so ownership is tested far less
  than it will be used.
- France is unlabelled; only coverage and score distributions can be checked there.
- The recall ceiling from blocking (0.9293; India 0.8806) bounds every result above.
