"""Experiment: does cross-source corroboration (xsrc_*) help? Same model, with and without the 4 xsrc features.

Fit cand_fit200k, validate on the exact val50k scope, 40 base features (no rev_*), leading-zero-fixed num_*,
num_leaves 255, up to 3000 trees, early stopping 100. Reports macro F0.5 after ownership, threshold, and
the below-threshold misses (retrieved true pairs scored under t), overall and per country.

Usage: python -u src/exp_xsrc.py
"""
import json
import time

import lightgbm as lgb
import pandas as pd

from predict import assign_owners, macro_f05_scored, sweep_threshold
from train import PARAMS, attach_labels, feature_columns, gold_sizes
from valsplit import GT_PATH, sample_val_scope
from xsrc_features import XSRC

DROP = ["rev_rank", "rev_count", "rev_score_gap"]
PARAMS_BEST = {"num_leaves": 255, "n_estimators": 3000}


def load(feat: str, xs: str) -> pd.DataFrame:
    """Feature parquet joined with its xsrc parquet on the pair key, labels attached."""
    df = pd.read_parquet(feat).merge(pd.read_parquet(xs), on=["s1_id", "cand_id"], how="left")
    assert df[XSRC].notna().all().all()
    return attach_labels(df, str(GT_PATH))


def run(fit, val, sizes, cols, label, country):
    t0 = time.time()
    m = lgb.LGBMClassifier(**{**PARAMS, **PARAMS_BEST})
    m.fit(fit[cols], fit["y"], eval_set=[(val[cols], val["y"])], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    v = val[["s1_id", "cand_id", "y"]].copy()
    v["p"] = m.predict_proba(val[cols])[:, 1]
    r = sweep_threshold(v, sizes)
    t = r["t"]
    pred = assign_owners(v, t)
    below = v[(v["y"] == 1) & (v["p"] < t)]
    per = {}
    for c in ("US", "India"):
        ids = set(country.index[country == c])
        per[c] = round(macro_f05_scored(v[v["s1_id"].isin(ids)], sizes[sizes.index.isin(ids)], t), 4)
    tp = int(pred.merge(v[["s1_id", "cand_id", "y"]], on=["s1_id", "cand_id"])["y"].sum())
    res = {"label": label, "n_features": len(cols), "macro_f05": round(r["macro_f05"], 4), "t": t,
           "best_iter": int(m.best_iteration_), "below_t_misses": len(below),
           "below_t_by_country": below["s1_id"].map(country).value_counts().to_dict(),
           "per_country": per, "pair_precision": round(tp / len(pred), 4),
           "pair_recall": round(tp / int(sizes.sum()), 4), "avg_matches_per_s1": round(len(pred) / len(sizes), 3),
           "seconds": round(time.time() - t0)}
    if label.startswith("+"):
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
        res["xsrc_gain_rank"] = {c: int((imp > imp[c]).sum()) + 1 for c in XSRC}
    print(json.dumps(res), flush=True)
    return res


def main():
    fit = load("artifacts/cand_fit200k_features_numfix.parquet", "artifacts/cand_fit200k_xsrc.parquet")
    val = load("artifacts/cand_val50k_features_numfix.parquet", "artifacts/cand_val50k_xsrc.parquet")
    scope = sample_val_scope(50_000)
    sizes = gold_sizes(str(GT_PATH))
    sizes = sizes[sizes.index.isin(scope)]
    s1 = pd.read_parquet("artifacts/train_s1_norm.parquet", columns=["entity_id", "country"])
    country = s1[s1["entity_id"].isin(scope)].set_index("entity_id")["country"]
    base = [c for c in feature_columns(fit) if c not in DROP and c not in XSRC]
    res = [run(fit, val, sizes, base, "baseline 40", country),
           run(fit, val, sizes, base + XSRC, "+ xsrc (44)", country)]
    json.dump(res, open("artifacts/exp_xsrc_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
