"""Writers for the two submission files (matching_results.tsv, candidate_pairs.tsv).

Format (CLAUDE.md "Outputs"): tab-separated, header row, one row per test S1
(empty lists allowed), IDs comma-joined and sorted, UTF-8 without BOM, LF line
endings, no quoting. Rows are ordered by S1 id so output is deterministic.

Implemented with pyarrow compute so ~50M candidate pairs never become Python
string objects (which would cost several GB).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

MATCHING_FILE = "matching_results.tsv"
CANDIDATE_FILE = "candidate_pairs.tsv"
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")
S1_RE = r"^S1-[0-9A-Za-z_-]+$"   # no tab / comma / quote / newline can slip through
TGT_RE = r"^S[23]-[0-9A-Za-z_-]+$"


def _to_table(df, name: str) -> pa.Table:
    """(s1_id, cand_id) as a string Arrow table from a pandas DataFrame or Arrow table."""
    cols = df.column_names if isinstance(df, pa.Table) else list(df.columns)
    missing = {"s1_id", "cand_id"} - set(cols)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if not isinstance(df, pa.Table):
        df = pa.Table.from_pandas(df[["s1_id", "cand_id"]], preserve_index=False)
    return pa.table({c: pc.cast(df[c], pa.string()).combine_chunks() for c in ("s1_id", "cand_id")})


def _check_re(arr: pa.Array, regex: str, name: str) -> None:
    """Raise ValueError when any id does not match regex."""
    ok = pc.match_substring_regex(arr, regex)
    if not pc.all(ok).as_py():
        raise ValueError(f"{name}: ids must match {regex}, e.g. {arr.filter(pc.invert(ok))[:3].to_pylist()}")


def _pair_keys(df, name: str, s1_sorted: pa.Array, tgt_sorted: pa.Array | None) -> tuple:
    """Validate pairs and return (sorted unique int64 keys, tgt_sorted).

    key = s1_rank * len(tgt_sorted) + tgt_rank, where ranks index the sorted id
    arrays, so ascending keys == (s1_id, cand_id) lexicographic order and
    np.unique both dedupes and sorts without touching Python strings.
    tgt_sorted=None builds it from this frame's own targets. A target missing
    from a given tgt_sorted raises (used for matches ⊆ candidates).
    """
    t = _to_table(df, name)
    ok = pc.match_substring_regex(t["cand_id"], TGT_RE)
    if not pc.all(ok).as_py():
        bad = t["cand_id"].filter(pc.invert(ok))
        raise ValueError(f"{name}: non S2-/S3- (or malformed) ids, e.g. {bad[:3].to_pylist()}")
    s1c = pc.index_in(t["s1_id"], value_set=s1_sorted)
    if s1c.null_count:
        ex = pc.unique(t["s1_id"].filter(pc.is_null(s1c)))
        raise ValueError(f"{name}: {len(ex)} s1_id not in test S1, e.g. {ex[:3].to_pylist()}")
    if tgt_sorted is None:
        u = pc.unique(t["cand_id"])
        tgt_sorted = u.take(pc.sort_indices(u))
    tc = pc.index_in(t["cand_id"], value_set=tgt_sorted)
    if tc.null_count:
        raise ValueError(f"{name}: {tc.null_count} match pair(s) not in candidates_df")
    n_t = max(len(tgt_sorted), 1)
    if len(s1_sorted) * n_t >= 2**62:
        raise OverflowError("pair keys would overflow int64")
    keys = s1c.to_numpy().astype(np.int64) * n_t + tc.to_numpy().astype(np.int64)
    return np.unique(keys), tgt_sorted


def _row_lists(keys: np.ndarray, tgt_sorted: pa.Array, n_s1: int) -> pa.Array:
    """Comma-joined id list for each of the n_s1 sorted S1s ('' when it has none)."""
    n_t = max(len(tgt_sorted), 1)
    counts = np.bincount(keys // n_t, minlength=n_s1)
    offsets = pa.array(np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
    values = tgt_sorted.take(pa.array(keys % n_t))
    return pc.binary_join(pa.LargeListArray.from_arrays(offsets, values), ",")


def _write(path: Path, header: tuple, s1_sorted: pa.Array, lists: pa.Array) -> None:
    """Write header + one line per S1 via a temp file and rename (never half-written)."""
    lines = pc.binary_join_element_wise(s1_sorted, lists, "\t")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        step = 200_000
        for lo in range(0, len(lines), step):
            f.write("\n".join(lines.slice(lo, step).to_pylist()) + "\n")
    os.replace(tmp, path)


def write_outputs(test_s1_ids, candidates_df, matches_df, out_dir) -> tuple[Path, Path]:
    """Write matching_results.tsv and candidate_pairs.tsv into out_dir.

    test_s1_ids: every test S1 id (each gets exactly one row, possibly empty).
    candidates_df / matches_df: pandas DataFrame or pyarrow Table with columns
    s1_id, cand_id (extra columns ignored, duplicate pairs collapsed).
    Raises ValueError if a pair references an S1 outside test_s1_ids, a target
    id is not S2-/S3-, or a match is not among the candidates.
    Returns (matching_path, candidate_path).
    """
    if isinstance(test_s1_ids, (pa.Array, pa.ChunkedArray)):
        s1 = pc.cast(test_s1_ids, pa.string())
    else:
        s1 = pa.array(pd.Series(list(test_s1_ids), dtype=object).astype(str), type=pa.string())
    s1 = pc.unique(s1)
    if len(s1) and not pc.all(pc.starts_with(s1, "S1-")).as_py():
        raise ValueError("test_s1_ids must all start with 'S1-'")
    _check_re(s1, S1_RE, "test_s1_ids")
    s1_sorted = pc.take(s1, pc.sort_indices(s1))
    if isinstance(s1_sorted, pa.ChunkedArray):
        s1_sorted = s1_sorted.combine_chunks()

    ckeys, tgt_sorted = _pair_keys(candidates_df, "candidates_df", s1_sorted, None)
    # matches are keyed on the candidates' target dictionary, so keys are comparable
    mkeys, _ = _pair_keys(matches_df, "matches_df", s1_sorted, tgt_sorted)
    pos = np.minimum(np.searchsorted(ckeys, mkeys), max(len(ckeys) - 1, 0))
    missing = int((ckeys[pos] != mkeys).sum()) if len(mkeys) else 0
    if missing:
        raise ValueError(f"matches_df: {missing} match pair(s) not in candidates_df")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mpath, cpath = out / MATCHING_FILE, out / CANDIDATE_FILE
    _write(cpath, CANDIDATE_HEADER, s1_sorted, _row_lists(ckeys, tgt_sorted, len(s1_sorted)))
    del ckeys
    _write(mpath, MATCHING_HEADER, s1_sorted, _row_lists(mkeys, tgt_sorted, len(s1_sorted)))
    return mpath, cpath


def dedupe_pairs(t: pa.Table) -> pa.Table:
    """Drop duplicate (s1_id, cand_id) rows via int64 keys (far less RAM than a string group_by)."""
    s1 = pc.unique(t["s1_id"])
    tg = pc.unique(t["cand_id"])
    key = (pc.index_in(t["s1_id"], value_set=s1).to_numpy().astype(np.int64) * max(len(tg), 1)
           + pc.index_in(t["cand_id"], value_set=tg).to_numpy().astype(np.int64))
    _, first = np.unique(key, return_index=True)
    return t.take(pa.array(np.sort(first)))
