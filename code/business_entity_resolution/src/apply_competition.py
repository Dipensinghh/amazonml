"""Stage 3c — overwrite the rev_* columns of a feature parquet with whole-population values.

features.py derives rev_rank / rev_count / rev_score_gap inside the candidate file it is given,
so a sampled file (fit200k, fit400k_extra, val50k) only sees competition from its own S1s.
competition.py computes the same three columns over a whole population (cand_trainall for
train, cand_test for test). This script joins those values back onto a feature parquet by
(s1_id, cand_id), streaming one row group at a time. Row order, schema and every other column
are unchanged; a pair missing from the competition file is an error.

Memory: the competition file is streamed by row group, keeping only the feature file's S1s, held as
sorted int64 pair keys plus three float32 columns
(~20 bytes/pair plus the two id dictionaries); the feature file is never fully loaded.

Usage:
  python -u src/apply_competition.py --features artifacts/cand_val50k_features.parquet \
      --competition artifacts/competition_trainall.parquet \
      --out artifacts/cand_val50k_features_fullrev.parquet
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import peak_ram_gb

REV = ["rev_rank", "rev_count", "rev_score_gap"]


def load_competition(path: Path, s1_filter: pa.Array | None = None) -> dict:
    """Competition rows as sorted int64 keys over (s1, cand) dictionaries, plus the rev_* values.

    s1_filter: keep only these S1 ids (bounds RAM when the feature file covers a subset).
    """
    pf = pq.ParquetFile(path)
    parts = []
    for i in range(pf.num_row_groups):  # stream: only the wanted S1s are ever held
        rg = pf.read_row_group(i, columns=["s1_id", "cand_id", *REV])
        parts.append(rg if s1_filter is None else rg.filter(pc.is_in(rg["s1_id"], value_set=s1_filter)))
    t = pa.concat_tables(parts)
    del parts
    s1 = pc.dictionary_encode(t["s1_id"].combine_chunks())
    cd = pc.dictionary_encode(t["cand_id"].combine_chunks())
    n_cd = max(len(cd.dictionary), 1)
    keys = s1.indices.to_numpy().astype(np.int64) * n_cd + cd.indices.to_numpy().astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    if len(keys) > 1 and (keys[1:] == keys[:-1]).any():
        raise ValueError(f"{path}: duplicate (s1_id, cand_id) rows")
    vals = {c: t[c].to_numpy().astype(np.float32)[order] for c in REV}
    return {"keys": keys, "vals": vals, "s1": s1.dictionary, "cd": cd.dictionary, "n_cd": n_cd}


def lookup(comp: dict, s1_ids, cand_ids) -> tuple:
    """Positions of the given pairs in comp, and a mask of pairs that were found."""
    s = pc.index_in(s1_ids, value_set=comp["s1"])
    c = pc.index_in(cand_ids, value_set=comp["cd"])
    ok = ~(pc.is_null(s).to_numpy(zero_copy_only=False) | pc.is_null(c).to_numpy(zero_copy_only=False))
    key = (pc.fill_null(s, 0).to_numpy().astype(np.int64) * comp["n_cd"]
           + pc.fill_null(c, 0).to_numpy().astype(np.int64))
    pos = np.searchsorted(comp["keys"], key)
    pos = np.minimum(pos, max(len(comp["keys"]) - 1, 0))
    ok &= comp["keys"][pos] == key if len(comp["keys"]) else False
    return pos, ok


def apply(features: Path, competition: Path, out: Path) -> int:
    """Rewrite features -> out with rev_* taken from competition; return the row count."""
    pf = pq.ParquetFile(features)
    names = pf.schema_arrow.names
    missing = [c for c in REV if c not in names]
    if missing:
        raise ValueError(f"{features}: no {missing} columns to replace")
    s1_ids = pc.unique(pq.read_table(features, columns=["s1_id"])["s1_id"])
    comp = load_competition(competition, s1_ids)
    print(f"competition: {len(comp['keys']):,} pairs for {len(s1_ids):,} S1s "
          f"(peak RAM {peak_ram_gb():.1f} GB)", flush=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    writer = pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd")
    rows = 0
    try:
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            pos, ok = lookup(comp, t["s1_id"], t["cand_id"])
            if not ok.all():
                bad = np.flatnonzero(~ok)[0]
                raise ValueError(f"{int((~ok).sum()):,} pair(s) in row group {i} are not in {competition}, "
                                 f"e.g. {t['s1_id'][bad]} / {t['cand_id'][bad]}")
            for c in REV:
                j = names.index(c)
                t = t.set_column(j, pf.schema_arrow.field(c),
                                 pa.array(comp["vals"][c][pos]).cast(pf.schema_arrow.field(c).type))
            writer.write_table(t)
            rows += t.num_rows
    except BaseException:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(tmp, out)
    return rows


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="feature parquet (features.py output)")
    ap.add_argument("--competition", required=True, help="competition.py output for the whole population")
    ap.add_argument("--out", required=True, help="output parquet (must differ from --features)")
    a = ap.parse_args()
    if Path(a.out).resolve() == Path(a.features).resolve():
        ap.error("--out must differ from --features")
    t0 = time.perf_counter()
    n = apply(Path(a.features), Path(a.competition), Path(a.out))
    print(f"{a.out}: {n:,} rows, all matched; {time.perf_counter() - t0:.0f}s, "
          f"peak RAM {peak_ram_gb():.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
