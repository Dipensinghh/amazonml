"""Validation report for a trained model bundle: macro F0.5 (overall / per country, with and without
global ownership), chosen threshold, pair precision/recall, singleton accuracy, and K sensitivity.

Scope is the exact validation sample (default: blocking's 50,000 S1s), so S1s with zero candidates count as
empty predictions. K sensitivity re-scores the SAME model on val candidates filtered to final_rank <= K
(rows from the blank-address index are always kept) and re-tunes the threshold; no new blocking is run.

Usage: python src/val_report.py --model models --features-val artifacts/cand_val50k_features.parquet \
           --cand artifacts/cand_val50k.parquet --scope-sample 50000 --k 5
"""
import argparse
import json

import lightgbm as lgb
import pandas as pd

from common import ART
from predict import assign_owners, macro_f05_scored, sweep_threshold
from train import attach_labels, gold_sizes
from valsplit import GT_PATH, load_val_ids, sample_val_scope


def country_map(ids) -> pd.Series:
    """S1 id -> country from the normalized train S1 parquet."""
    t = pd.read_parquet(ART / "train_s1_norm.parquet", columns=["entity_id", "country"])
    return t[t["entity_id"].isin(set(ids))].set_index("entity_id")["country"]


def pair_stats(scored: pd.DataFrame, sizes: pd.Series, t: float) -> dict:
    """Post-ownership pair precision/recall and singleton accuracy at threshold t."""
    w = assign_owners(scored, t).merge(scored[["s1_id", "cand_id", "y"]], on=["s1_id", "cand_id"], how="left")
    tp, npred = int(w["y"].sum()), len(w)
    pred_by_s1 = w.groupby("s1_id").size()
    singles = sizes.index[sizes == 0]
    false_merge = int(pred_by_s1.reindex(singles).fillna(0).gt(0).sum())
    return {"pairs_predicted": npred, "pair_precision": tp / npred if npred else 0.0,
            "pair_recall": tp / int(sizes.sum()), "singletons": len(singles),
            "singleton_accuracy": 1 - false_merge / max(len(singles), 1), "singleton_false_merges": false_merge}


def report(df: pd.DataFrame, sizes: pd.Series, n_scope: int) -> dict:
    """Threshold sweep with/without ownership, per-country scores at the chosen threshold, pair stats."""
    w, wo = sweep_threshold(df, sizes, owners=True), sweep_threshold(df, sizes, owners=False)
    t = w["t"]
    cm = country_map(sizes.index)
    per = {}
    for c in sorted(cm.unique()):
        ids = cm.index[cm == c]
        per[c] = {"n_s1": len(ids), "macro_f05": macro_f05_scored(df[df["s1_id"].isin(set(ids))], sizes[sizes.index.isin(ids)], t)}
    return {"t": t, "macro_f05_with_ownership": w["macro_f05"], "macro_f05_without_ownership": wo["macro_f05"],
            "t_without_ownership": wo["t"], "per_country_at_t": per, **pair_stats(df, sizes, t),
            "avg_cands_per_s1": len(df) / n_scope, "n_scope_s1": n_scope, "n_pairs": len(df)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models", help="dir with lgbm.txt and bundle.json")
    ap.add_argument("--features-val", required=True)
    ap.add_argument("--cand", help="candidate parquet with final_rank / blank_addr_index (needed for K sensitivity)")
    ap.add_argument("--scope-sample", type=int, default=50_000)
    ap.add_argument("--scope-ids")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="artifacts/val_report.json")
    a = ap.parse_args()
    bundle = json.load(open(f"{a.model}/bundle.json"))
    booster = lgb.Booster(model_file=f"{a.model}/lgbm.txt")
    df = attach_labels(pd.read_parquet(a.features_val), str(GT_PATH))
    df["p"] = booster.predict(df[bundle["features"]])
    scope = {l.strip() for l in open(a.scope_ids) if l.strip()} if a.scope_ids else sample_val_scope(a.scope_sample)
    scope &= load_val_ids()
    sizes = gold_sizes(str(GT_PATH))
    sizes = sizes[sizes.index.isin(scope)]
    df = df[df["s1_id"].isin(scope)]
    out = {"features_in_bundle": len(bundle["features"]), "best_iteration": bundle["best_iteration"],
           "full": report(df, sizes, len(scope))}
    if a.cand:
        cand = pd.read_parquet(a.cand, columns=["s1_id", "cand_id", "final_rank", "blank_addr_index"])
        m = df.merge(cand, on=["s1_id", "cand_id"], how="left")
        keep = (m["final_rank"] <= a.k) | m["blank_addr_index"].fillna(0).astype(bool)
        out[f"k{a.k}"] = report(m[keep].drop(columns=["final_rank", "blank_addr_index"]), sizes, len(scope))
        out["k_delta_macro_f05"] = out[f"k{a.k}"]["macro_f05_with_ownership"] - out["full"]["macro_f05_with_ownership"]
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
