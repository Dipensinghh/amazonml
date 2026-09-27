"""Stage 3b — the competition block (rev_rank, rev_count, rev_score_gap) over a whole
candidate parquet.

features.collapse() derives these from whatever candidate file it is handed, so a sampled
file (e.g. cand_fit200k, 200k S1s) only sees the S1s *in that sample* competing for a target.
At test time every S1 competes, so the sampled values understate rev_count / rev_rank. This
script computes them once over the full population (cand_trainall: all 2.2M train S1s) so the
fitting rows can carry the realistic values.

The math is not reimplemented: each piece is passed through features.collapse(), which rebuilds
the composite score with blocking.FROZEN's alpha. Two things keep peak RAM bounded:

  * ids are dictionary-encoded to int32 codes before pandas sees them (strings dominate RAM);
  * rows are partitioned by cand_id code into --buckets pieces. Every rev_* value depends only
    on the rows sharing its cand_id (and a pair's duplicate rows share it too), so each bucket
    is computed exactly; only the within-S1 columns collapse() also emits would be wrong, and
    they are discarded.

Output: artifacts/competition_{name}.parquet (s1_id, cand_id, rev_rank, rev_count, rev_score_gap),
one row per distinct pair. Ids are decoded back to strings one row group at a time (writing the
int codes as a DictionaryArray would copy the 9.7M-entry cand dictionary into every row group).

With --compare, also reports mean rev_count / rev_rank by label on the rows of a feature parquet
(default cand_fit200k_features) next to the sample-level values stored there.

Usage:
  python -u src/competition.py --name trainall --compare
"""
import argparse
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import ART, peak_ram_gb, timed
from features import collapse  # the single definition of the competition block

REV = ["rev_rank", "rev_count", "rev_score_gap"]
NEEDED = ["tok_score", "name_sim", "final_rank"]   # what collapse()'s v3 path reads
WRITE_ROWS = 1 << 20   # decode ids to strings per row group, never the whole column at once


def encode_ids(path, col: str) -> tuple:
    """Read one id column and return (int32 codes, string dictionary)."""
    arr = pq.read_table(path, columns=[col])[col].combine_chunks()
    enc = pc.dictionary_encode(arr)
    del arr
    return enc.indices.to_numpy().astype(np.int32, copy=False), enc.dictionary


def fit_keys(feat_path, s1_dict: pa.Array, cd_dict: pa.Array, n_cd: int) -> tuple:
    """The comparison rows as sorted int64 pair keys, plus their labels and stored rev_* values."""
    t = pq.read_table(feat_path, columns=["s1_id", "cand_id", "label", "rev_rank", "rev_count"])
    s = pc.fill_null(pc.index_in(t["s1_id"], value_set=s1_dict), -1).to_numpy().astype(np.int64)
    c = pc.fill_null(pc.index_in(t["cand_id"], value_set=cd_dict), -1).to_numpy().astype(np.int64)
    ok = (s >= 0) & (c >= 0)
    keys = np.where(ok, s * n_cd + c, -1)
    order = np.argsort(keys, kind="stable")
    stored = {k: t[k].to_numpy() for k in ("label", "rev_rank", "rev_count")}
    return keys[order], {k: v[order] for k, v in stored.items()}, int((~ok).sum())


def run(cand_path, out_path, buckets: int, feat_path=None) -> dict:
    """Compute the competition block bucket by bucket and stream it to out_path."""
    t0 = time.perf_counter()
    with timed("encode ids"):
        s1_codes, s1_dict = encode_ids(cand_path, "s1_id")
        cd_codes, cd_dict = encode_ids(cand_path, "cand_id")
        num = pq.read_table(cand_path, columns=NEEDED)
    n = len(s1_codes)
    n_cd = len(cd_dict)
    print(f"  {n:,} rows, {len(s1_dict):,} S1s, {n_cd:,} candidates", file=sys.stderr, flush=True)

    cmp = None
    if feat_path:
        keys, stored, missing = fit_keys(feat_path, s1_dict, cd_dict, n_cd)
        full = {k: np.full(len(keys), np.nan, np.float32) for k in REV}
        cmp = (keys, stored, missing, full)

    schema = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()),
                        *[(k, pa.float32()) for k in REV]])
    written = 0
    with pq.ParquetWriter(out_path, schema, compression="zstd") as w:
        bucket_of = cd_codes % buckets
        for b in range(buckets):
            with timed(f"bucket {b + 1}/{buckets}"):
                idx = np.flatnonzero(bucket_of == b)
                piece = pa.table({"s1_id": s1_codes[idx], "cand_id": cd_codes[idx],
                                  **{c: num[c].take(pa.array(idx)) for c in NEEDED}})
                del idx
                agg = collapse(piece)
                del piece
                s = agg["s1_id"].to_numpy(np.int32)
                c = agg["cand_id"].to_numpy(np.int32)
                rev = {k: agg[k].to_numpy(np.float32) for k in REV}
                del agg
                for lo in range(0, len(s), WRITE_ROWS):
                    sl = slice(lo, lo + WRITE_ROWS)
                    w.write_table(pa.table({
                        "s1_id": s1_dict.take(pa.array(s[sl])),
                        "cand_id": cd_dict.take(pa.array(c[sl])),
                        **{k: pa.array(v[sl]) for k, v in rev.items()}}, schema=schema))
                written += len(s)
                if cmp is not None:
                    k = s.astype(np.int64) * n_cd + c
                    pos = np.searchsorted(cmp[0], k)
                    pos_c = np.minimum(pos, len(cmp[0]) - 1)
                    hit = cmp[0][pos_c] == k
                    for name in REV:
                        cmp[3][name][pos_c[hit]] = rev[name][hit]
    elapsed = time.perf_counter() - t0
    print(f"wrote {out_path}: {written:,} distinct pairs in {elapsed:,.1f}s "
          f"(peak RAM {peak_ram_gb():.2f} GB)", flush=True)
    if cmp is not None:
        compare(*cmp)
    return dict(rows=written, seconds=elapsed, peak_gb=peak_ram_gb())


def compare(keys, stored, missing, full) -> None:
    """Mean rev_count / rev_rank by label: sample-level (stored) vs whole-population."""
    found = ~np.isnan(full["rev_count"])
    print(f"\ncomparison rows: {len(keys):,}  matched in full file: {int(found.sum()):,}  "
          f"(ids absent from full file: {missing:,})")
    lab = stored["label"]
    print(f"{'label':>5} {'n':>10} | {'rev_count':>21} | {'rev_rank':>21}")
    print(f"{'':>5} {'':>10} | {'sample':>10} {'full':>10} | {'sample':>10} {'full':>10}")
    for L in (0, 1):
        m = (lab == L) & found
        print(f"{L:>5} {int(m.sum()):>10,} | {stored['rev_count'][m].mean():>10.3f} "
              f"{full['rev_count'][m].mean():>10.3f} | {stored['rev_rank'][m].mean():>10.3f} "
              f"{full['rev_rank'][m].mean():>10.3f}")
    for L in (0, 1):
        m = (lab == L) & found
        print(f"  label {L}: full rev_rank==0 share {np.mean(full['rev_rank'][m] == 0):.3f} "
              f"(sample {np.mean(stored['rev_rank'][m] == 0):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="trainall", help="reads artifacts/cand_{name}.parquet")
    ap.add_argument("--buckets", type=int, default=4, help="cand_id partitions (RAM bound)")
    ap.add_argument("--compare", nargs="?", const=str(ART / "cand_fit200k_features.parquet"),
                    default=None, help="feature parquet whose rows to compare (label + rev_*)")
    args = ap.parse_args()
    run(ART / f"cand_{args.name}.parquet", ART / f"competition_{args.name}.parquet",
        args.buckets, args.compare)


if __name__ == "__main__":
    sys.exit(main())
