"""Build production forward-union-reverse(R=1) candidate parquets.

The frozen forward blocker remains untouched. Every target queries the country-specific S1
token index; its top S1 is the reverse candidate. Forward evidence wins for pairs found in both
directions, while ``from_reverse`` records reverse discovery. Final ranks are recomputed over
the complete union. Hash buckets keep the 60M-row train union within RAM.

Usage:
  python -u src/reverse_union.py --name trainall --subsets
  python -u src/reverse_union.py --name test
"""
import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from joblib import Parallel

import blocking as bl


BASE = ["s1_id", "cand_id", "source", "blank_addr_index", "tok_score", "name_sim"]
OUT_SCHEMA = pa.schema([
    ("s1_id", pa.string()), ("cand_id", pa.string()), ("source", pa.string()),
    ("blank_addr_index", pa.bool_()), ("tok_score", pa.float32()),
    ("name_sim", pa.float32()), ("final_rank", pa.int32()), ("from_reverse", pa.bool_()),
])


def top_one(found: pd.DataFrame, targets: pd.DataFrame, s1: pd.DataFrame) -> pd.DataFrame:
    """Rerank the frozen top-30 pool and retain one S1 per target."""
    if found.empty:
        return found
    sim = bl.name_sim(targets, s1, found["q_row"].to_numpy(), found["t_row"].to_numpy())
    key = found["score"].to_numpy() + bl.FROZEN["alpha"] * sim / 100.0
    q = found["q_row"].to_numpy()
    if np.any(q[1:] < q[:-1]):
        raise ValueError("blocking.search must return rows grouped by q_row")
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(q)])
    best_key = np.repeat(np.maximum.reduceat(key, starts), lengths)
    tied = key == best_key
    t = found["t_row"].to_numpy()
    best_t = np.repeat(np.minimum.reduceat(np.where(tied, t, np.iinfo(t.dtype).max), starts), lengths)
    take = np.flatnonzero(tied & (t == best_t))
    return found.iloc[take].assign(name_sim=sim[take], key=key[take])


def build_reverse(split: str, out: Path, jobs: int, chunk: int) -> int:
    """Stream every target's top-1 S1 to a reverse candidate parquet."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    s1_all = bl.load("S1", split)
    schema = pa.schema([(c, OUT_SCHEMA.field(c).type) for c in BASE])
    rows = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer, Parallel(n_jobs=jobs) as pool:
        for country in sorted(s1_all["country"].unique()):
            s1 = s1_all[s1_all["country"].eq(country)].reset_index(drop=True)
            t0 = time.perf_counter()
            index = bl.TokenIndex(s1, pool)
            idf = index.for_cap(bl.FROZEN["cap"])
            s1_ids = s1["entity_id"].to_numpy()
            print(f"{split}/{country}: indexed {len(s1):,} S1 in {time.perf_counter()-t0:.0f}s",
                  flush=True)
            for source in bl.TARGETS:
                all_targets = bl.load(source, split)
                all_targets = all_targets[all_targets["country"].eq(country)].reset_index(drop=True)
                for start in range(0, len(all_targets), chunk):
                    q = all_targets.iloc[start:start + chunk].reset_index(drop=True)
                    qa, qn = index.queries(q, pool)
                    found = bl.search(bl.weight_queries(qa, qn, idf), index.tt, bl.POOL)
                    selected = top_one(found, q, s1)
                    qi = selected["q_row"].to_numpy()
                    table = pa.table({
                        "s1_id": s1_ids[selected["t_row"].to_numpy()],
                        "cand_id": q["entity_id"].to_numpy()[qi],
                        "source": np.full(len(selected), source),
                        "blank_addr_index": np.zeros(len(selected), dtype=bool),
                        "tok_score": selected["score"].to_numpy(np.float32),
                        "name_sim": selected["name_sim"].to_numpy(np.float32),
                    }, schema=schema)
                    writer.write_table(table)
                    rows += len(selected)
                    print(f"  {country}/{source}: {min(start+chunk, len(all_targets)):,}/"
                          f"{len(all_targets):,}; reverse rows {rows:,}", flush=True)
                del all_targets
            del index, idf
    os.replace(tmp, out)
    return rows


def bucket_of(ids: pa.Array, buckets: int) -> np.ndarray:
    """Stable bucket from the numeric S1 suffix."""
    return np.fromiter((int(x[3:]) % buckets for x in ids.to_pylist()), np.int16, len(ids))


def partition(path: Path, folder: Path, kind: str, buckets: int, batch: int) -> None:
    """Partition a parquet by S1 so every pair for one S1 is processed together."""
    pf = pq.ParquetFile(path)
    writers = {}
    try:
        for rb in pf.iter_batches(batch_size=batch, columns=BASE):
            table = pa.Table.from_batches([rb])
            b = bucket_of(table["s1_id"], buckets)
            for value in np.unique(b):
                piece = table.take(pa.array(np.flatnonzero(b == value)))
                writer = writers.get(int(value))
                if writer is None:
                    writer = pq.ParquetWriter(folder / f"{kind}_{int(value):03d}.parquet",
                                              piece.schema, compression="zstd")
                    writers[int(value)] = writer
                writer.write_table(piece)
    finally:
        for writer in writers.values():
            writer.close()


def union_bucket(forward: pd.DataFrame, reverse: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Prefer forward evidence, add reverse-only pairs, and recompute deterministic S1 ranks."""
    keys = ["s1_id", "cand_id"]
    fidx = pd.MultiIndex.from_frame(forward[keys])
    ridx = pd.MultiIndex.from_frame(reverse[keys])
    forward = forward.copy()
    forward["from_reverse"] = fidx.isin(ridx)
    only = reverse.loc[~ridx.isin(fidx)].copy()
    only["from_reverse"] = True
    overlap = len(reverse) - len(only)
    out = pd.concat([forward, only], ignore_index=True)
    score = out["tok_score"].to_numpy(np.float32)
    score += bl.FROZEN["alpha"] * out["name_sim"].to_numpy(np.float32) / 100.0
    out["_score"] = score
    out = out.sort_values(["s1_id", "_score", "cand_id"],
                          ascending=[True, False, True], kind="stable")
    out["final_rank"] = out.groupby("s1_id", sort=False).cumcount().astype(np.int32)
    return out[[*BASE, "final_rank", "from_reverse"]], overlap


def build_union(forward: Path, reverse: Path, out: Path, buckets: int, batch: int) -> dict:
    """External-memory union via S1 hash buckets."""
    folder = out.parent / f"_{out.stem}_buckets"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    print(f"partitioning {forward}", flush=True)
    partition(forward, folder, "f", buckets, batch)
    print(f"partitioning {reverse}", flush=True)
    partition(reverse, folder, "r", buckets, batch)
    tmp = out.with_suffix(out.suffix + ".tmp")
    rows = overlap = reverse_only = 0
    with pq.ParquetWriter(tmp, OUT_SCHEMA, compression="zstd") as writer:
        for b in range(buckets):
            fp, rp = folder / f"f_{b:03d}.parquet", folder / f"r_{b:03d}.parquet"
            f = pq.read_table(fp).to_pandas() if fp.exists() else pd.DataFrame(columns=BASE)
            r = pq.read_table(rp).to_pandas() if rp.exists() else pd.DataFrame(columns=BASE)
            piece, hit = union_bucket(f, r)
            writer.write_table(pa.Table.from_pandas(piece, schema=OUT_SCHEMA, preserve_index=False))
            rows += len(piece); overlap += hit; reverse_only += len(r) - hit
            print(f"  union bucket {b+1}/{buckets}: rows {rows:,}, reverse-only {reverse_only:,}",
                  flush=True)
    os.replace(tmp, out)
    shutil.rmtree(folder)
    return {"rows": rows, "reverse_only": reverse_only, "reverse_overlap": overlap}


def extract_subsets(src: Path, batch: int) -> None:
    """Extract the exact seeded fit200k and val50k S1 populations from trainall."""
    split = pq.read_table(bl.ART / "split_train.parquet").to_pandas()
    fit = set(bl.sample_ids(split.loc[~split["is_val"], "entity_id"].to_numpy(), 200_000))
    val = set(bl.sample_ids(split.loc[split["is_val"], "entity_id"].to_numpy(), 50_000))
    schema = pq.ParquetFile(src).schema_arrow
    outputs = {"fit200k_r1": (fit, 0), "val50k_r1": (val, 0)}
    writers = {name: pq.ParquetWriter(bl.ART / f"cand_{name}.parquet", schema, compression="zstd")
               for name in outputs}
    try:
        for rb in pq.ParquetFile(src).iter_batches(batch_size=batch):
            for name, (ids, count) in list(outputs.items()):
                selected = rb.filter(pc.is_in(rb.column("s1_id"), value_set=pa.array(ids)))
                if selected.num_rows:
                    writers[name].write_batch(selected)
                    outputs[name] = (ids, count + selected.num_rows)
    finally:
        for writer in writers.values():
            writer.close()
    for name, (ids, count) in outputs.items():
        print(f"wrote cand_{name}.parquet: {count:,} pairs / {len(ids):,} S1", flush=True)


def self_check() -> None:
    f = pd.DataFrame([("S1-1", "S2-1", "S2", False, 5.0, 10.0)], columns=BASE)
    r = pd.DataFrame([("S1-1", "S2-1", "S2", False, 9.0, 90.0),
                      ("S1-1", "S3-1", "S3", False, 4.0, 10.0)], columns=BASE)
    out, overlap = union_bucket(f, r)
    assert overlap == 1 and len(out) == 2
    assert out.loc[out["cand_id"].eq("S2-1"), "tok_score"].iat[0] == 5.0
    assert out["from_reverse"].all() and out["final_rank"].tolist() == [0, 1]
    print("PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", choices=["trainall", "test"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=500_000)
    ap.add_argument("--buckets", type=int, default=64)
    ap.add_argument("--reuse-reverse", action="store_true")
    ap.add_argument("--subsets", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check(); return
    if not args.name:
        ap.error("--name is required unless --self-check is used")
    split = "train" if args.name == "trainall" else "test"
    forward = bl.ART / f"cand_{args.name}.parquet"
    reverse = bl.ART / f"reverse_{args.name}_r1.parquet"
    out = bl.ART / f"cand_{args.name}_r1.parquet"
    t0 = time.perf_counter()
    if not args.reuse_reverse or not reverse.exists():
        rows = build_reverse(split, reverse, args.jobs, args.chunk)
        print(f"wrote {reverse}: {rows:,} rows", flush=True)
    stats = build_union(forward, reverse, out, args.buckets, args.batch)
    n_s1 = pq.ParquetFile(bl.norm_path(split, "S1")).metadata.num_rows
    print(f"wrote {out}: {stats['rows']:,} pairs ({stats['rows']/n_s1:.2f}/S1), "
          f"reverse-only {stats['reverse_only']:,}, overlap {stats['reverse_overlap']:,}; "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    if args.subsets:
        extract_subsets(out, args.batch)


if __name__ == "__main__":
    main()
