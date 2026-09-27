# Section 2 — Normalization and candidate retrieval

## Normalization

All input columns were read as strings from tab-separated files with quoting disabled. Raw text
was preserved in the organizer data; derived Parquets contain the normalized fields. Names and
addresses were repaired with ftfy, Unicode NFKC normalization, AnyAscii transliteration,
lowercasing, conservative punctuation cleanup, and small hand-written legal-suffix/address
abbreviation rules. Name aliases introduced by `dba`, `fka`, and `aka` were retained separately.
The pipeline also derived sorted-name, acronym, numeric, postal-like, script, and phonetic fields.

The training normalization run covered 2,206,821 S1, 5,034,616 S2, and 5,285,603 S3 rows.
The independent test run covered 1,732,544 S1, 4,887,273 S2, and 5,082,316 S3 rows and peaked
at 5.1 GB RAM. A labelled-pair study showed that Indic records contain transliterations rather
than translations: the phonetic name key increased mean non-Latin true-pair similarity from
66.7 to 93.1. No embedding model or external data was used.

## Retrieval method

Retrieval was partitioned by the open-set country value and by target source. No labelled true
pair crosses countries, so this partition was lossless on training. The frozen index hashes
address and name words, number variants, concatenated names, and unordered pairs of non-common
tokens. Target tokens with document frequency above 200 are excluded. Each S1 retrieves a
30-record pool per target source, which is reranked by rare-token evidence plus six times the
normalized name similarity. The final quota is 10 records per source, plus up to two per source
from a separate name-only index for blank-address targets.

On the seeded 50,000-S1 validation scope, the frozen configuration achieved 0.9293 labelled-pair
recall at 23.7 candidates per S1: 0.9621 in the US and 0.8806 in India. The independent full-test
run produced 40,696,839 pairs (23.4896 per S1) in 433 seconds at 10.6 GB peak RAM. Country means
were 23.7616 for the US, 23.5262 for India, and 22.6806 for unseen France. One France S1 had no
candidate. Exact audits found no duplicate pair and verified source prefixes, finite scores,
contiguous ranks, and normalized-query order.

## India miss study

The exact India subset of the seeded validation scope contained 20,086 S1s and 69,685 true
pairs. The frozen blocker missed 8,327 pairs. A mutually exclusive diagnostic attributed 2,049
to strong phonetic evidence lost through transliteration mismatch, 477 to blank target
addresses, 1,074 to shared tokens whose target frequency exceeded the cap, and 4,727 to other
noise or pool/rerank limitations.

All alternatives retained K=10 and therefore stayed near the 24-candidate budget:

| Configuration | India recall | Candidates/S1 | Change |
|---|---:|---:|---:|
| Frozen cap 200 | 0.8805 | 23.51 | — |
| India cap 500 | 0.8803 | 23.77 | -0.02 points |
| India cap 1000 | 0.8799 | 23.88 | -0.06 points |
| Transliteration variants | 0.8690 | 23.78 | -1.15 points |
| Phonetic address pairs | 0.8563 | 23.53 | -2.42 points |

None met the predeclared two-point improvement threshold, so the production blocker was left
unchanged. This avoids replacing a validated cross-country recipe with an India-specific rule
that performed worse locally.

## Downstream features and ownership

Candidate features include normalized and phonetic string similarities, address TF-IDF,
numeric/postal agreement, retrieval score and rank, score gaps, and reverse competition fields.
The 40,696,839-row test feature artifact contains 43 ordered features. Exact audits verified that
every candidate pair appears once in identical order and that every feature is finite. Reverse
competition fields were computed over all 1,732,544 test S1s before feature batching; an
independent full-file check reproduced them for 100 contested targets spanning 3,514 S1s.

The LightGBM threshold is selected on validation after applying global competition. Target
exclusivity is a verified training property and is used as an inference constraint: each S2/S3
target is assigned to its highest-scoring S1, with deterministic lexicographic tie-breaking.
