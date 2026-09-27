"""Recompute only num_agree / num_conflict / num_missing in an existing feature parquet, using
features.num_features (leading zeros stripped), instead of rebuilding all 43 features.

Streams the file in record batches and writes a new parquet with the same columns, column order and row
order; every other column is copied unchanged. Only entity_id + addr_nums of the referenced normalized
records are loaded. Writes to a temporary file and renames it over the output when finished.

Usage:
  python -u src/patch_num_features.py --features artifacts/cand_val50k_features.parquet --split train
  python -u src/patch_num_features.py --features artifacts/cand_test_features.parquet --split test
"""
import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import peak_ram_gb
from features import load_records, num_features, positions

NUM_COLS = ("num_agree", "num_conflict", "num_missing")


def unique_ids(pf: pq.ParquetFile, col: str) -> pa.Array:
    """Distinct values of one id column, read column-only."""
    return pc.unique(pf.read(columns=[col]).column(col).combine_chunks())


def patch(src: str, dst: str, split: str, batch: int = 1_000_000) -> dict:
    """Write dst = src with the three num_* columns recomputed; returns before/after flag counts."""
    pf = pq.ParquetFile(src)
    names = pf.schema_arrow.names
    missing = [c for c in ("s1_id", "cand_id", *NUM_COLS) if c not in names]
    if missing:
        raise ValueError(f"{src} lacks columns {missing}")
    t0 = time.perf_counter()
    cols = ["entity_id", "addr_nums"]
    s1 = load_records(unique_ids(pf, "s1_id"), split, ("S1",), cols)
    cd = load_records(unique_ids(pf, "cand_id"), split, ("S2", "S3"), cols)
    print(f"  loaded {s1.num_rows:,} S1 + {cd.num_rows:,} candidate addr_nums "
          f"({time.perf_counter() - t0:.0f}s, peak RAM {peak_ram_gb():.1f} GB)", file=sys.stderr, flush=True)

    stats = {"rows": 0, "conflict_before": 0, "conflict_after": 0, "agree_changed": 0}
    has_label = "label" in names
    if has_label:
        stats.update(pos_conflict_before=0, pos_conflict_after=0, neg_conflict_before=0, neg_conflict_after=0)
    tmp = dst + ".tmp"
    with pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd") as w:
        for rb in pf.iter_batches(batch_size=batch):
            a = s1.column("addr_nums").take(positions(s1, rb.column("s1_id"))).to_pylist()
            b = cd.column("addr_nums").take(positions(cd, rb.column("cand_id"))).to_pylist()
            new = num_features(a, b)
            old_conf = np.asarray(rb.column("num_conflict"))
            stats["rows"] += rb.num_rows
            stats["conflict_before"] += int(old_conf.sum())
            stats["conflict_after"] += int(new["num_conflict"].sum())
            stats["agree_changed"] += int((np.asarray(rb.column("num_agree")) != new["num_agree"]).sum())
            if has_label:
                lab = np.asarray(rb.column("label")) == 1
                stats["pos_conflict_before"] += int(old_conf[lab].sum())
                stats["pos_conflict_after"] += int(new["num_conflict"][lab].sum())
                stats["neg_conflict_before"] += int(old_conf[~lab].sum())
                stats["neg_conflict_after"] += int(new["num_conflict"][~lab].sum())
            arrays = [pa.array(new[c]).cast(rb.schema.field(c).type) if c in NUM_COLS else rb.column(c)
                      for c in names]
            w.write_batch(pa.RecordBatch.from_arrays(arrays, schema=pf.schema_arrow))
            print(f"  patched {stats['rows']:,}/{pf.metadata.num_rows:,}", file=sys.stderr, flush=True)
    os.replace(tmp, dst)
    stats["seconds"] = round(time.perf_counter() - t0)
    stats["peak_ram_gb"] = round(peak_ram_gb(), 1)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="feature parquet to patch")
    ap.add_argument("--split", required=True, choices=["train", "test"], help="which normalized records to join")
    ap.add_argument("--out", default=None, help="output path (default: overwrite --features in place)")
    ap.add_argument("--batch", type=int, default=1_000_000)
    args = ap.parse_args()
    stats = patch(args.features, args.out or args.features, args.split, args.batch)
    for k, v in stats.items():
        print(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
