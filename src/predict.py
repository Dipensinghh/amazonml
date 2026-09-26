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


def sweep_threshold(scored: pd.DataFrame, gold_sizes: pd.Series, grid=None, owners: bool = True) -> Dict[str, float]:
    """Pick t maximising macro F0.5 after ownership.

    Coarse pass: 0.10-0.95 step 0.05 plus the all (t=0) and none (t=1.01) endpoints; then a 0.01-step
    refinement within +-0.05 of the coarse best. Returns best t, best score and the full curve.
    """
    if grid is None:
        coarse = [0.0] + [round(x, 2) for x in np.arange(0.10, 0.951, 0.05)] + [1.01]
        curve = {t: macro_f05_scored(scored, gold_sizes, t, owners) for t in coarse}
        b = max(curve, key=curve.get)
        for t in np.round(np.arange(max(b - 0.05, 0.0), min(b + 0.05, 1.0) + 1e-9, 0.01), 2):
            curve.setdefault(float(t), macro_f05_scored(scored, gold_sizes, float(t), owners))
    else:
        curve = {float(t): macro_f05_scored(scored, gold_sizes, float(t), owners) for t in grid}
    best = max(curve, key=curve.get)
    return {"t": best, "macro_f05": curve[best], "curve": curve}


def reduce_owners_streaming(batches: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Global target ownership over scored batches without holding every pair in memory.

    Keeps only the running best (p, then smallest s1_id) per cand_id, so peak memory is one row per
    distinct target plus one batch. Threshold afterwards with `winners[winners.p >= t]`; that is
    equivalent to thresholding first because t is global.
    """
    winners = None
    for b in batches:
        cur = b[["s1_id", "cand_id", "p"]]
        if winners is not None:
            cur = pd.concat([winners, cur], ignore_index=True)
        cur = cur.sort_values(["cand_id", "p", "s1_id"], ascending=[True, False, True], kind="mergesort")
        winners = cur.drop_duplicates("cand_id", keep="first")
    return winners if winners is not None else pd.DataFrame(columns=["s1_id", "cand_id", "p"])


def ownership_report(scored: pd.DataFrame, gold_sizes: pd.Series, t: float) -> Dict[str, float]:
    """Constrained vs unconstrained macro F0.5 and the share of above-threshold targets claimed by 2+ S1s."""
    above = scored[scored["p"] >= t]
    coll = float((above.groupby("cand_id")["s1_id"].nunique() > 1).mean()) if len(above) else 0.0
    return {"with_ownership": macro_f05_scored(scored, gold_sizes, t, True),
            "without_ownership": macro_f05_scored(scored, gold_sizes, t, False),
            "collision_rate": coll}


def validation_report(scored: pd.DataFrame, gold_sizes: pd.Series) -> Dict[str, dict]:
    """Best macro F0.5 with and without target ownership, each with its own tuned threshold.

    gold_sizes must cover EVERY validation S1 (index = s1_id), so S1s with zero candidates score as
    empty predictions (1.0 on true singletons, 0.0 otherwise).
    """
    w = sweep_threshold(scored, gold_sizes, owners=True)
    wo = sweep_threshold(scored, gold_sizes, owners=False)
    return {"with_ownership": {"t": w["t"], "macro_f05": w["macro_f05"]},
            "without_ownership": {"t": wo["t"], "macro_f05": wo["macro_f05"]},
            "ownership_at_best_t": ownership_report(scored, gold_sizes, w["t"]),
            "n_val_s1": int(len(gold_sizes)), "n_pairs": int(len(scored)),
            "s1_with_no_candidates": int((~gold_sizes.index.isin(scored["s1_id"].unique())).sum())}


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
    ap.add_argument("--model", default="artifacts/lgbm.txt", help="use artifacts/lgbm_full.txt for the final refit")
    ap.add_argument("--threshold", type=float, default=None, help="defaults to artifacts/threshold.json")
    ap.add_argument("--s1-ids", required=True, help="text file: every S1 id to output, one per line")
    ap.add_argument("--out", default="output/matching_results.tsv")
    a = ap.parse_args()
    t = a.threshold if a.threshold is not None else json.load(open("artifacts/threshold.json"))["t"]
    booster = lgb.Booster(model_file=a.model)
    df = pd.read_parquet(a.features)
    feats = json.load(open("artifacts/bundle.json"))["features"]
    df["p"] = booster.predict(df[feats])
    with open(a.s1_ids) as f:
        ids = [l.strip() for l in f if l.strip()]
    write_matches(to_lists(assign_owners(df, t), ids), a.out)
    print(f"wrote {a.out} (t={t})")
