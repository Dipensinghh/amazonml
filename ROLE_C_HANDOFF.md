# Role C handoff — retrieval and performance

Owner: Aryan (Role C)  
Branch: `c-retrieval`  
Base audited: `origin/main` at `58858c2`

## Scope

Role C owns production retrieval runs, resource measurements, and the handoff of candidate
pairs to features/modeling. Dipen owns `src/blocking.py`, `src/normalize.py`, and
`src/common.py`; this branch does not modify those files.

## Completed

- Created the Python 3.14 virtual environment and installed the pinned requirements.
- Linked the organizer dataset locally without adding it to Git.
- Rebuilt all normalized test parquets with `python -u src/normalize.py --split test`.
- Reproduced the frozen blocker with
  `python -u src/blocking.py --produce test --jobs 8`.
- Audited every candidate group for schema, nulls, finite score ranges, source prefixes,
  contiguous ranks, duplicate candidates, and S1 query order.
- Ran A's integration suite (26 passed), D's model checks (passed), and B's v3 candidate
  collapse against 100,000 real test pairs.
- Ran B's full feature builder on all 40,696,839 test pairs: exact pair/order equality, 43
  finite feature columns, 23,738 pairs/s, 11.6 GB peak RAM.

## Production artifacts

Ignored artifacts stay in `/Users/aryanjha/amazonml/artifacts/`:

| Artifact | Rows | Size |
|---|---:|---:|
| `test_s1_norm.parquet` | 1,732,544 | 144 MiB |
| `test_s2_norm.parquet` | 4,887,273 | 424 MiB |
| `test_s3_norm.parquet` | 5,082,316 | 434 MiB |
| `cand_test.parquet` | 40,696,839 | 395 MiB |
| `cand_test_audit.json` | audit report | 1.1 KiB |
| `cand_test_features.parquet` | 40,696,839 | 2.08 GiB |
| `cand_test_features_audit.json` | audit report | 1.2 KiB |

Candidate schema:
`s1_id, cand_id, source, blank_addr_index, tok_score, name_sim, final_rank`.

The run took 433 seconds and peaked at 10.6 GB RAM. It produced 23.4896 candidates per
test S1; one France S1 has no candidates. Means by country are US 23.7616, India 23.5262,
and France 22.6806. These reproduce Dipen's reported full-test result.

SHA-256 checksums for transfer verification:

```text
5ee5d5922a6e478a1960daaadcf9b6ff75696145e6a766a5e773a61cf0f85795  test_s1_norm.parquet
4f365178a26e3bdf4948bd8bf7cea7a52460f8b19e49670b586c7f864873ffc0  test_s2_norm.parquet
c979cf88e477c5cdfb6638804d0667e93860b1a5dacc91a38ff80ae0cd8e25e6  test_s3_norm.parquet
9f871dcc03a851b48b37197eb2b5b6b446862b091cfa275d492ab02020f5e73a  cand_test.parquet
985fb0f86e699e1cc1d2395b5f75af9ce9722c8466eaf517d20ff0cac9c149c2  cand_test_features.parquet
```

## Integration status

- `b-features` already contains current `main`. Its frozen-v3 contract consumes this artifact
  correctly and preserves all sampled `(s1_id, cand_id)` pairs.
- `a-integration` is two main commits behind. A test merge with current `main` is clean; keep
  main's frozen blocker when integrating.
- `d-model` is four main commits behind. A test merge has one add/add conflict in
  `EXPERIMENTS.md`; keep both experiment sections. Use B's latest `features.py`, not D's older
  copy.
- B's feature CLI retains the full candidate frame, referenced records, and completed feature
  chunks until writing. The measured full run is safe on this 36 GiB Mac: about 31 minutes
  end-to-end and 11.6 GB peak RAM. A streamed writer is unnecessary for today's run.

## Downstream handoff

After Dipen merges D, D can score `cand_test_features.parquet`; A can then enforce ownership,
write the two TSVs, and run both strict and organizer validators. Role C's normalized,
retrieval, performance, and feature-handoff work is complete. If the final scoring run is moved
to this Mac, Role C should verify exact scored/exported pair equality before giving A the paths.

### A — integration and submission

- Merge current `main`, `b-features`, `d-model`, and this branch; preserve main's frozen
  `src/blocking.py`. The current test merge of `a-integration` with main is clean.
- Copy the five production Parquets from this Mac and verify the SHA-256 values above; they are
  intentionally Git-ignored.
- After D returns scored pairs and the selected threshold, apply global target ownership, write
  both TSVs, run `src/checks.py`, then run the organizer validator with `--check-ids`.
- Complete the README, pinned requirements, methodology document, clean-room run, and ZIP.

### B — feature contract

- Get `b-features` (`b7cb2e9`) merged before D; it is the only team branch already based on the
  current main blocker and it defines the 43-feature frozen-v3 contract used by the full artifact.
- Give D the exact ordered feature list from `cand_test_features_audit.json` and keep that order
  in the model bundle. No full test feature rerun is needed unless B changes feature logic.
- If feature logic changes, version the output separately and rerun pair/order and finite-value
  checks before replacing the audited artifact.

### D — model, threshold, and scoring

- Merge current main and B's latest `features.py`; resolve the expected add/add conflict in
  `EXPERIMENTS.md` by retaining both experiment histories.
- Train on `cand_fit200k` features, evaluate and tune the threshold on the exact `cand_val50k`
  S1 scope with ownership enabled, then refit the selected LightGBM configuration on all allowed
  labelled fitting data.
- Score `cand_test_features.parquet` without changing row order or pair coverage. Return a score
  artifact containing exactly `s1_id`, `cand_id`, and the model probability, plus the model,
  bundle, threshold, and validation report for A.

## Additional handoff completed after B became unavailable

- `src/score_test.py` is ready for the LightGBM bundle. It scores in chunks, writes every score
  atomically in candidate order, resolves global ownership, writes both TSVs, and runs both
  validators. A contested-target dummy model passed the entire path.
- The test feature `rev_*` fields are global. An independent full-file check reproduced all
  three fields for 100 contested targets spanning 3,514 S1s.
- The India study found no improvement worth a production rerun. Baseline was 0.8805 recall at
  23.51 candidates/S1; cap 500, cap 1000, transliteration variants, and phonetic address pairs
  all reduced recall. Keep the frozen blocker.
- The package skeleton now contains `code/business_entity_resolution/{README.md,requirements.txt,run.py}`
  and `docs/section2_retrieval.md`. A must copy the final integrated `src/` into the package.
