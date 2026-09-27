"""Stage 2b — extra fitting S1s for v3: fit600k = cand_fit200k + cand_fit{n}_extra.

Selects n more non-validation train S1s that are not already in cand_fit200k, with the same
recipe as blocking.sample_ids (seed 42 numpy default_rng choice over the sorted id list), and
takes their candidate rows from cand_trainall. Blocking runs every S1 query independently
against the same target index, so those rows are what `blocking.py --produce` would return
for these S1s.

Reproduces the exact 400,000 ids used to train v3 (checked against the original
fit400k_extra_s1_ids.parquet). Runs in about 11 s.

Outputs: artifacts/fit{n//1000}k_extra_s1_ids.parquet, artifacts/cand_fit{n//1000}k_extra.parquet

Usage:
  python -u src/fit_extra.py --n 400000
"""
import argparse
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import ART, SEED, peak_ram_gb


def sorted_ids(arr) -> pa.Array:
    """Arrow string array sorted ascending (C++ sort; no Python string objects)."""
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    return arr.take(pc.sort_indices(arr))


def sample_sorted(ids: pa.Array, n: int) -> pa.Array:
    """blocking.sample_ids on Arrow: seed-42 choice of n from the sorted ids, returned sorted.

    Generator.choice(a, n, replace=False) draws the same positions as choice(len(a), n, ...),
    so this equals np.sort(default_rng(42).choice(np.sort(ids), n, replace=False)).
    """
    ids = sorted_ids(ids)
    if n <= 0 or n >= len(ids):
        return ids
    pos = np.random.default_rng(SEED).choice(len(ids), size=n, replace=False)
    return sorted_ids(ids.take(pa.array(pos)))


def extra_ids(n: int) -> pa.Array:
    """n non-validation train S1s outside cand_fit200k (seed 42), sorted."""
    split = pq.read_table(ART / "split_train.parquet")
    nonval = split.filter(pc.invert(split["is_val"]))["entity_id"]
    fit200 = pc.unique(pq.read_table(ART / "cand_fit200k.parquet", columns=["s1_id"])["s1_id"])
    eligible = nonval.filter(pc.invert(pc.is_in(nonval, value_set=fit200)))
    print(f"non-val {len(nonval):,}; fit200k {len(fit200):,}; eligible {len(eligible):,}", flush=True)
    return sample_sorted(eligible, n)


def main() -> int:
    """Write the extra id list and its candidate rows filtered from cand_trainall."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400_000, help="number of extra fitting S1s")
    a = ap.parse_args()
    t0 = time.perf_counter()
    tag = f"fit{a.n // 1000}k_extra"
    ids = extra_ids(a.n)
    pq.write_table(pa.table({"s1_id": ids}), ART / f"{tag}_s1_ids.parquet")
    pf = pq.ParquetFile(ART / "cand_trainall.parquet")
    out = ART / f"cand_{tag}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    kept = 0
    with pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd") as w:
        for i in range(pf.num_row_groups):
            rg = pf.read_row_group(i)
            rg = rg.filter(pc.is_in(rg["s1_id"], value_set=ids))
            w.write_table(rg)
            kept += rg.num_rows
    tmp.replace(out)
    n_s1 = len(pc.unique(pq.read_table(out, columns=["s1_id"])["s1_id"]))
    print(f"wrote {out}: {kept:,} pairs, {n_s1:,} S1 with candidates; "
          f"{time.perf_counter() - t0:.0f}s, peak RAM {peak_ram_gb():.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
