# Experiments

| Run | Owner | Question | Setup | Result | Decision |
|---|---|---|---|---|---|
| E00 | D | Are scoring/split correct? | toy asserts (0.714 example, singleton rules); split seed 42, val = ceil(0.2*2,206,821) = 441,365 S1s | asserts pass; all-empty baseline macro F0.5 = 0.0561 on val | metric + split frozen |
| E01 | D | Does a rules fallback work? | 2,000 random val S1s (seed 42) vs ALL train S2/S3 of same country+source; word TF-IDF (max_df 0.02) name top-10 + address top-10 union; score = w*name_cos + (1-w)*addr_cos (addr blank -> name only); global ownership; one threshold | pair recall 0.930, 37.9 cands/S1 (75,862 pairs); best w_name 0.2 -> t 0.73, **macro F0.5 0.662** (w 0.5: 0.646, w 0.0: 0.619); US 0.706 / India 0.574; ownership effect +0.001 (collision rate 0.2%, thin competition at 2k S1s); runtime 167 s, peak RSS 2.7 GB | fallback ready as a floor; LightGBM must beat 0.662 on the same candidates |

Limits of E01: 2,000-S1 sample (noisy, thin ownership competition); own light normalisation (not B's); word-level retrieval (not C's trigram channels); no acronym channel; names made only of common tokens get no name candidates.
