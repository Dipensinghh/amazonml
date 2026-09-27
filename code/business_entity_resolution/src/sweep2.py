"""A's second sweep request: does stacking num_leaves with more trees help, without the
scale-fragile rev_rank/rev_count/rev_score_gap features (B flagged these as shifting 3x
between val50k and fit200k, and again at test scale). 200k fit S1s, same val50k scope.

Usage: python -u src/sweep2.py
"""
import json
import time

import lightgbm as lgb
import pandas as pd

from train import PARAMS, attach_labels, feature_columns, gold_sizes
from predict import assign_owners, sweep_threshold, ownership_report
from valsplit import GT_PATH, sample_val_scope

DROP = ["rev_rank", "rev_count", "rev_score_gap"]


def fit_and_eval(tr, va, sizes, params, cols, label):
    t0 = time.time()
    model = lgb.LGBMClassifier(**{**PARAMS, **params})
    model.fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    va = va.copy()
    va["p"] = model.predict_proba(va[cols])[:, 1]
    r = sweep_threshold(va, sizes)
    own = ownership_report(va, sizes, r["t"])
    kept = assign_owners(va, r["t"])
    avg_matches = len(kept) / len(sizes)
    stopped_early = model.best_iteration_ < params.get("n_estimators", PARAMS["n_estimators"])
    print(f"{label:35s} F0.5={r['macro_f05']:.4f}  t={r['t']:.2f}  "
          f"avg_match/S1={avg_matches:.3f}  best_iter={model.best_iteration_}"
          f"{' (early stop)' if stopped_early else ' (hit cap)'}  ({time.time()-t0:.0f}s)", flush=True)
    return {"label": label, "macro_f05": r["macro_f05"], "t": r["t"],
            "avg_matches_per_s1": avg_matches, "best_iteration": int(model.best_iteration_),
            "stopped_early": stopped_early, "n_features": len(cols)}


def main():
    fit = attach_labels(pd.read_parquet("artifacts/cand_fit200k_features.parquet"), str(GT_PATH))
    val = attach_labels(pd.read_parquet("artifacts/cand_val50k_features.parquet"), str(GT_PATH))
    scope = sample_val_scope(50_000)
    sizes = gold_sizes(str(GT_PATH))
    sizes = sizes[sizes.index.isin(scope)]
    cols = [c for c in feature_columns(fit) if c not in DROP]
    print(f"fit {len(fit):,} pairs / {fit['s1_id'].nunique():,} S1s; "
          f"val {len(val):,} pairs / {len(sizes):,} scope S1s; "
          f"{len(cols)} features (dropped {DROP})", flush=True)

    results = []
    results.append(fit_and_eval(fit, val, sizes, {"num_leaves": 127, "n_estimators": 3000}, cols,
                                "leaves=127, up to 3000 trees"))
    results.append(fit_and_eval(fit, val, sizes, {"num_leaves": 255, "n_estimators": 3000}, cols,
                                "leaves=255, up to 3000 trees"))

    json.dump(results, open("artifacts/sweep2_results.json", "w"), indent=2)
    print("\n| config | val F0.5 | threshold | avg matches/S1 | best iter | stopped |")
    print("|---|---|---|---|---|---|")
    for r in results:
        stop = "early" if r["stopped_early"] else "hit cap"
        print(f"| {r['label']} | {r['macro_f05']:.4f} | {r['t']:.2f} | "
              f"{r['avg_matches_per_s1']:.3f} | {r['best_iteration']} | {stop} |")


if __name__ == "__main__":
    main()
