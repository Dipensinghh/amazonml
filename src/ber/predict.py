"""Decision policy: threshold + global target ownership, and vectorized macro-F0.5 threshold sweep.

A scored table has columns s1_id, cand_id, p (match probability). Ownership is resolved over the
WHOLE table at once (never per S1 chunk): each S2/S3 target goes to the single S1 that scored it
highest, ties broken by lexicographically smallest s1_id.
"""
import argparse
import json
from typing import Dict, Iterable, Set

import numpy as np
import pandas as pd

META = {"s1_id", "cand_id", "source", "y", "p"}


def assign_owners(scored: pd.DataFrame, t: float) -> pd.DataFrame:
    """Keep pairs with p >= t, then give every cand_id to its highest-p S1 (ties: smallest s1_id)."""
    kept = scored.loc[scored["p"] >= t, ["s1_id", "cand_id", "p"]]
    kept = kept.sort_values(["cand_id", "p", "s1_id"], ascending=[True, False, True], kind="mergesort")
    return kept.drop_duplicates("cand_id", keep="first")


def to_lists(assigned: pd.DataFrame, all_s1_ids: Iterable[str]) -> Dict[str, Set[str]]:
    """Group assigned pairs into {s1_id: set(cand_id)}; every S1 in all_s1_ids gets an entry (possibly empty)."""
    out: Dict[str, Set[str]] = {k: set() for k in all_s1_ids}
    for s, c in zip(assigned["s1_id"].tolist(), assigned["cand_id"].tolist()):
        out[s].add(c)
    return out


def macro_f05_scored(scored: pd.DataFrame, gold_sizes: pd.Series, t: float, owners: bool = True) -> float:
    """Vectorized macro F0.5 on a labelled scored table (column y = pair is a true match).

    gold_sizes: number of true matches per S1 id, indexed by s1_id, covering every S1 being scored
    (includes S1s with no candidates and singletons).
    """
    kept = assign_owners(scored, t) if owners else scored.loc[scored["p"] >= t, ["s1_id", "cand_id", "p"]]
    if "y" in scored:
        kept = kept.merge(scored[["s1_id", "cand_id", "y"]], on=["s1_id", "cand_id"], how="left")
    else:
        raise ValueError("scored table needs a y column for evaluation")
    g = kept.groupby("s1_id").agg(npred=("y", "size"), tp=("y", "sum"))
    df = pd.DataFrame({"ngold": gold_sizes}).join(g).fillna({"npred": 0, "tp": 0})
    p = df["tp"] / df["npred"].where(df["npred"] > 0, 1)
    r = df["tp"] / df["ngold"].where(df["ngold"] > 0, 1)
    f = (1.25 * p * r / (0.25 * p + r).where((0.25 * p + r) > 0, 1)).where(df["tp"] > 0, 0.0)
    both_empty = (df["ngold"] == 0) & (df["npred"] == 0)
    f = f.where(~both_empty, 1.0)
    return float(f.mean())


def sweep_threshold(scored: pd.DataFrame, gold_sizes: pd.Series, grid=None) -> Dict[str, float]:
    """Pick the threshold maximising macro F0.5 after ownership; returns best t, best score and the full curve."""
    grid = np.round(np.arange(0.30, 0.96, 0.02), 2) if grid is None else grid
    curve = {float(t): macro_f05_scored(scored, gold_sizes, float(t)) for t in grid}
    best = max(curve, key=curve.get)
    return {"t": best, "macro_f05": curve[best], "curve": curve}


def write_matches(lists: Dict[str, Set[str]], path: str, header: str = "source1_entity_id\tmatched_entity_ids") -> None:
    """Write id lists as TSV: UTF-8, LF, no BOM, sorted ids, one row per S1 (sorted), empty list allowed."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for k in sorted(lists):
            f.write(f"{k}\t{','.join(sorted(lists[k]))}\n")


if __name__ == "__main__":
    import lightgbm as lgb

    ap = argparse.ArgumentParser(description="Score a feature table with a trained model and apply the threshold + ownership policy.")
    ap.add_argument("--features", required=True, help="parquet with s1_id, cand_id and feature columns")
    ap.add_argument("--model", default="artifacts/lgbm.txt")
    ap.add_argument("--threshold", type=float, default=None, help="defaults to artifacts/threshold.json")
    ap.add_argument("--s1-ids", required=True, help="text file: every S1 id to output, one per line")
    ap.add_argument("--out", default="output/matching_results.tsv")
    a = ap.parse_args()
    t = a.threshold if a.threshold is not None else json.load(open("artifacts/threshold.json"))["t"]
    booster = lgb.Booster(model_file=a.model)
    df = pd.read_parquet(a.features)
    df["p"] = booster.predict(df[booster.feature_name()])
    with open(a.s1_ids) as f:
        ids = [l.strip() for l in f if l.strip()]
    write_matches(to_lists(assign_owners(df, t), ids), a.out)
    print(f"wrote {a.out} (t={t})")
