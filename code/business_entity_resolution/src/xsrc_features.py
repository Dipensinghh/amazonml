"""Cross-source corroboration features for candidate pairs.

A true S2 match for an S1 usually has a true S3 counterpart among the same S1's candidates (85% of matched
S1s have matches in both sources), and the two records describe the same business. For each candidate pair
(S1, t), compare t with the S1's top-5 candidates from the OTHER source (by final_rank):

  xsrc_best_name     max token_set_ratio(name_core_t, name_core_t')
  xsrc_best_addr     max token_set_ratio(addr_norm_t, addr_norm_t')
  xsrc_best_both     max over partners of the mean of the two
  xsrc_partner_rank  final_rank of the partner achieving xsrc_best_both (-1 if no partner)

Similarities are in [0, 1]; an empty string on either side scores 0. Computed with rapidfuzz cpdist over
the generated (t, t') pairs, in chunks of S1s to bound memory. Output is keyed by (s1_id, cand_id).

Usage: python -u src/xsrc_features.py --candidates artifacts/cand_val50k.parquet --split train
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz

from common import peak_ram_gb
from features import load_records, positions, sim

XSRC = ["xsrc_best_name", "xsrc_best_addr", "xsrc_best_both", "xsrc_partner_rank"]
TOP = 5


def xsrc(cand: pd.DataFrame, split: str, chunk_s1: int = 25_000) -> pd.DataFrame:
    """The four xsrc features for every (s1_id, cand_id) in cand (needs s1_id, cand_id, source, final_rank)."""
    cand = cand[["s1_id", "cand_id", "source", "final_rank"]].drop_duplicates(["s1_id", "cand_id"])
    cand = cand.assign(src_rank=cand.groupby(["s1_id", "source"])["final_rank"].rank(method="first"))
    top = cand[cand["src_rank"] <= TOP][["s1_id", "cand_id", "source", "final_rank"]]
    top = top.rename(columns={"cand_id": "partner", "source": "psource", "final_rank": "prank"})

    recs = load_records(pa.array(cand["cand_id"].unique(), type=pa.string()), split, ("S2", "S3"),
                        ["entity_id", "name_core", "addr_norm"])
    s1s = cand["s1_id"].unique()
    out = []
    t0 = time.perf_counter()
    for i in range(0, len(s1s), chunk_s1):
        keep = set(s1s[i:i + chunk_s1])
        c = cand[cand["s1_id"].isin(keep)]
        p = c.merge(top[top["s1_id"].isin(keep)], on="s1_id")
        p = p[p["source"] != p["psource"]]
        pos_t = positions(recs, pa.array(p["cand_id"].to_numpy(), type=pa.string()))
        pos_p = positions(recs, pa.array(p["partner"].to_numpy(), type=pa.string()))
        n_t = recs.column("name_core").take(pos_t).to_pylist()
        n_p = recs.column("name_core").take(pos_p).to_pylist()
        a_t = recs.column("addr_norm").take(pos_t).to_pylist()
        a_p = recs.column("addr_norm").take(pos_p).to_pylist()
        p = p.assign(n=sim(n_t, n_p, fuzz.token_set_ratio), a=sim(a_t, a_p, fuzz.token_set_ratio))
        p["both"] = (p["n"] + p["a"]) / 2
        best = p.sort_values(["s1_id", "cand_id", "both", "prank"], ascending=[True, True, False, True])
        best = best.drop_duplicates(["s1_id", "cand_id"])[["s1_id", "cand_id", "both", "prank"]]
        agg = p.groupby(["s1_id", "cand_id"], sort=False).agg(xsrc_best_name=("n", "max"),
                                                               xsrc_best_addr=("a", "max")).reset_index()
        agg = agg.merge(best, on=["s1_id", "cand_id"]).rename(
            columns={"both": "xsrc_best_both", "prank": "xsrc_partner_rank"})
        out.append(agg)
        print(f"  xsrc {min(i + chunk_s1, len(s1s)):,}/{len(s1s):,} S1s, {len(p):,} partner pairs "
              f"({time.perf_counter() - t0:.0f}s, peak RAM {peak_ram_gb():.1f} GB)", file=sys.stderr, flush=True)
    res = cand[["s1_id", "cand_id"]].merge(pd.concat(out, ignore_index=True), on=["s1_id", "cand_id"], how="left")
    res[XSRC[:3]] = res[XSRC[:3]].fillna(0.0)
    res["xsrc_partner_rank"] = res["xsrc_partner_rank"].fillna(-1)
    return res.astype({c: np.float32 for c in XSRC})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = xsrc(pd.read_parquet(a.candidates), a.split)
    out = a.out or a.candidates.replace(".parquet", "_xsrc.parquet")
    res.to_parquet(out)
    print(f"wrote {out}: {len(res):,} rows")


if __name__ == "__main__":
    sys.exit(main())
