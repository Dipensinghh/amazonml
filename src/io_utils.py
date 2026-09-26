"""Writers for the two submission files (matching_results.tsv, candidate_pairs.tsv).

Format (CLAUDE_v2.md "Outputs"): tab-separated, header row, one row per test S1
(empty lists allowed), IDs comma-joined and sorted, UTF-8 without BOM, LF line
endings, no quoting. Rows are ordered by S1 id so output is deterministic.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

MATCHING_FILE = "matching_results.tsv"
CANDIDATE_FILE = "candidate_pairs.tsv"
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")
_BAD_CHARS = ("\t", "\n", "\r", ",", '"')


def _pairs(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return deduplicated (s1_id, cand_id) string pairs sorted by s1_id then cand_id."""
    missing = {"s1_id", "cand_id"} - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    out = df[["s1_id", "cand_id"]].astype(str).drop_duplicates()
    bad = ~out["cand_id"].str.startswith(("S2-", "S3-"))
    if bad.any():
        raise ValueError(f"{name}: non S2-/S3- ids, e.g. {out.loc[bad, 'cand_id'].head(3).tolist()}")
    return out.sort_values(["s1_id", "cand_id"], ignore_index=True)


def _check_ids_clean(ids, name: str) -> None:
    """Refuse ids containing separators/quotes, which would corrupt the TSV."""
    for ch in _BAD_CHARS:
        hit = ids[ids.str.contains(ch, regex=False)]
        if len(hit):
            raise ValueError(f"{name}: id contains {ch!r}: {hit.iloc[0]!r}")


def _write_lists(path: Path, header: tuple, s1_sorted: np.ndarray, pairs: pd.DataFrame) -> None:
    """Stream one row per S1 in s1_sorted, joining that S1's (already sorted) ids.

    Writes to a temp file and renames, so a crash never leaves a half-written file.
    """
    ps1 = pairs["s1_id"].to_numpy(dtype=object)
    pcand = pairs["cand_id"].to_numpy(dtype=object)
    # Row slice [lo, hi) of pairs belonging to each S1 (pairs are sorted by s1_id).
    lo = np.searchsorted(ps1, s1_sorted, side="left")
    hi = np.searchsorted(ps1, s1_sorted, side="right")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        buf = []
        for s1, a, b in zip(s1_sorted, lo, hi):
            buf.append(f"{s1}\t{','.join(pcand[a:b])}\n")
            if len(buf) >= 100_000:
                f.write("".join(buf))
                buf.clear()
        f.write("".join(buf))
    os.replace(tmp, path)


def write_outputs(test_s1_ids, candidates_df: pd.DataFrame, matches_df: pd.DataFrame,
                  out_dir) -> tuple[Path, Path]:
    """Write matching_results.tsv and candidate_pairs.tsv into out_dir.

    test_s1_ids: every test S1 id (each gets exactly one row, possibly empty).
    candidates_df / matches_df: long format with columns s1_id, cand_id
    (extra columns ignored, duplicate pairs collapsed).
    Raises ValueError if a pair references an S1 outside test_s1_ids, a target
    id is not S2-/S3-, or a match is not among the candidates.
    Returns (matching_path, candidate_path).
    """
    s1 = pd.Series(pd.unique(pd.Series(list(test_s1_ids), dtype=object).astype(str)))
    if not s1.str.startswith("S1-").all():
        raise ValueError("test_s1_ids must all start with 'S1-'")
    _check_ids_clean(s1, "test_s1_ids")
    s1_sorted = np.sort(s1.to_numpy(dtype=object))

    cands = _pairs(candidates_df, "candidates_df")
    matches = _pairs(matches_df, "matches_df")
    _check_ids_clean(cands["cand_id"], "candidates_df")
    s1_set = set(s1_sorted)
    for name, df in (("candidates_df", cands), ("matches_df", matches)):
        unknown = set(df["s1_id"].unique()) - s1_set
        if unknown:
            raise ValueError(f"{name}: {len(unknown)} s1_id not in test S1, e.g. {sorted(unknown)[:3]}")

    # matches ⊆ candidates: an inner merge keeps only matched pairs that were candidates.
    kept = matches.merge(cands, on=["s1_id", "cand_id"], how="inner")
    if len(kept) != len(matches):
        raise ValueError(f"matches_df: {len(matches) - len(kept)} match pair(s) not in candidates_df")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mpath, cpath = out / MATCHING_FILE, out / CANDIDATE_FILE
    _write_lists(cpath, CANDIDATE_HEADER, s1_sorted, cands)
    _write_lists(mpath, MATCHING_HEADER, s1_sorted, matches)
    return mpath, cpath
