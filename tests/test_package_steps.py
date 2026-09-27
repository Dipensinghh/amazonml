"""Tests for the v3 reproduction steps: fit_extra sampling and apply_competition (synthetic, fast)."""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apply_competition import REV, apply  # noqa: E402
from blocking import sample_ids  # noqa: E402
from fit_extra import sample_sorted  # noqa: E402


@pytest.mark.parametrize("n", [0, 7, 500, 999, 1000, 5000])
def test_sample_sorted_equals_blocking_sample_ids(n):
    """fit_extra's Arrow sampler draws exactly blocking.sample_ids' ids (seed 42)."""
    rng = np.random.default_rng(1)
    ids = np.array([f"S1-{v}" for v in rng.choice(10**7, 1000, replace=False)], dtype=object)
    got = sample_sorted(pa.array(ids), n).to_pylist()
    assert got == list(sample_ids(ids, n))


def _write(path, table):
    """Write a small parquet with several row groups so streaming is exercised."""
    pq.write_table(table, path, row_group_size=3)
    return path


def _features():
    """Feature table: keys, a label, two ordinary features and stale rev_* values."""
    return pa.table({
        "s1_id": ["S1-2", "S1-1", "S1-2", "S1-3", "S1-1", "S1-3", "S1-2"],
        "cand_id": ["S2-9", "S2-9", "S3-4", "S3-4", "S3-7", "S2-1", "S2-1"],
        "label": pa.array([0, 1, 1, 0, 1, 1, 0], pa.int8()),
        "retr_score": pa.array(np.arange(7, dtype=np.float32)),
        "rev_rank": pa.array(np.zeros(7, np.float32)),
        "rev_count": pa.array(np.zeros(7, np.float32)),
        "rev_score_gap": pa.array(np.zeros(7, np.float32)),
        "addr_blank": pa.array(np.ones(7, np.float32)),
    })


def test_apply_competition_replaces_only_rev(tmp_path):
    """rev_* come from the competition file; row order, schema and other columns are kept."""
    feats = _features()
    # competition covers these pairs (+ an extra S1 not in the feature file), in a different order
    comp_pairs = [("S1-3", "S2-1"), ("S1-1", "S3-7"), ("S1-2", "S2-9"), ("S1-9", "S2-5"),
                  ("S1-1", "S2-9"), ("S1-3", "S3-4"), ("S1-2", "S2-1"), ("S1-2", "S3-4")]
    val = {p: (i + 1.0, 10.0 * (i + 1), -0.5 * i) for i, p in enumerate(comp_pairs)}
    comp = pa.table({"s1_id": [p[0] for p in comp_pairs], "cand_id": [p[1] for p in comp_pairs],
                     **{c: pa.array([val[p][k] for p in comp_pairs], pa.float32()) for k, c in enumerate(REV)}})
    f = _write(tmp_path / "f.parquet", feats)
    c = _write(tmp_path / "c.parquet", comp)
    out = tmp_path / "o.parquet"
    assert apply(f, c, out) == feats.num_rows
    got = pq.read_table(out)
    assert got.schema.equals(feats.schema, check_metadata=False)
    for col in feats.column_names:
        if col not in REV:
            assert got[col].equals(feats[col]), col
    pairs = list(zip(feats["s1_id"].to_pylist(), feats["cand_id"].to_pylist()))
    for k, col in enumerate(REV):
        assert got[col].to_pylist() == [val[p][k] for p in pairs], col


def test_apply_competition_missing_pair_fails(tmp_path):
    """A feature pair absent from the competition file is an error and leaves no output."""
    feats = _features()
    comp = pa.table({"s1_id": ["S1-2"], "cand_id": ["S2-9"],
                     **{c: pa.array([1.0], pa.float32()) for c in REV}})
    out = tmp_path / "o.parquet"
    with pytest.raises(ValueError, match="not in"):
        apply(_write(tmp_path / "f.parquet", feats), _write(tmp_path / "c.parquet", comp), out)
    assert not out.exists() and not out.with_suffix(".parquet.tmp").exists()
