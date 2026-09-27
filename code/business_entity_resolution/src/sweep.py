"""Config sweep on the real cand_fit200k/cand_val50k features: fit on fit200k, tune threshold on
the exact val50k scope with global ownership, report macro F0.5 / threshold / avg matches per S1
per config to EXPERIMENTS.md. No models are pushed; this is measurement only.

Usage: python -u src/sweep.py
"""
import json
import time

import lightgbm as lgb
import pandas as pd

from train import PARAMS, attach_labels, feature_columns, gold_sizes
from predict import assign_owners, sweep_threshold, ownership_report
from valsplit import GT_PATH, sample_val_scope

DEAD = ["name_missing", "was_non_latin", "postal_conflict", "is_s3", "postal_missing", "name_alt_token_set"]


def fit_and_eval(tr, va, sizes, params, cols, label):
    t0 = time.time()
    model = lgb.LGBMClassifier(**{**PARAMS, **params})
    model.fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    va = va.copy()
    va["p"] = model.predict_proba(va[cols])[:, 1]
    r = sweep_threshold(va, sizes)
    own = ownership_report(va, sizes, r["t"])
    kept = assign_owners(va, r["t"])
    avg_matches = len(kept) / len(sizes)
    print(f"{label:45s} F0.5={r['macro_f05']:.4f}  t={r['t']:.2f}  "
          f"no_own={own['without_ownership']:.4f}  avg_match/S1={avg_matches:.3f}  "
          f"best_iter={model.best_iteration_}  ({time.time()-t0:.0f}s)", flush=True)
    return {"label": label, "macro_f05": r["macro_f05"], "t": r["t"],
            "without_ownership": own["without_ownership"], "avg_matches_per_s1": avg_matches,
            "best_iteration": int(model.best_iteration_), "n_features": len(cols)}


def main():
    fit = pd.read_parquet("artifacts/cand_fit200k_features.parquet")
    val = pd.read_parquet("artifacts/cand_val50k_features.parquet")
    fit = attach_labels(fit, str(GT_PATH))
    val = attach_labels(val, str(GT_PATH))
    scope = sample_val_scope(50_000)
    sizes = gold_sizes(str(GT_PATH))
    sizes = sizes[sizes.index.isin(scope)]
    cols = feature_columns(fit)
    print(f"fit {len(fit):,} pairs / {fit['s1_id'].nunique():,} S1s; "
          f"val {len(val):,} pairs / {len(sizes):,} scope S1s (0 cand: "
          f"{int((~sizes.index.isin(val['s1_id'])).sum())}); {len(cols)} features", flush=True)

    results = []
    results.append(fit_and_eval(fit, val, sizes, {}, cols, "playbook baseline"))
    results.append(fit_and_eval(fit, val, sizes, {"num_leaves": 63}, cols, "num_leaves=63"))
    results.append(fit_and_eval(fit, val, sizes, {"num_leaves": 127}, cols, "num_leaves=127"))
    results.append(fit_and_eval(fit, val, sizes, {"min_child_samples": 100}, cols, "min_child_samples=100"))
    results.append(fit_and_eval(fit, val, sizes, {"learning_rate": 0.03, "n_estimators": 2000}, cols,
                                "lr=0.03, n_estimators=2000"))
    cols2 = [c for c in cols if c not in DEAD]
    results.append(fit_and_eval(fit, val, sizes, {}, cols2, f"drop {len(DEAD)} dead features ({len(cols2)} left)"))

    # K=5 check: re-filter val candidates (final_rank <= 5, keep blank-index rows), re-tune on the SAME
    # (baseline) model's config, since K only changes the candidate set fed to the model, not training.
    cand = pd.read_parquet("artifacts/cand_val50k.parquet", columns=["s1_id", "cand_id", "final_rank", "blank_addr_index"])
    k5_cand = cand[(cand["final_rank"] <= 5) | cand["blank_addr_index"]][["s1_id", "cand_id"]]
    val_k5 = val.merge(k5_cand, on=["s1_id", "cand_id"], how="inner")
    print(f"K=5 val candidates: {len(val_k5):,} pairs (full K=10: {len(val):,}), "
          f"avg {len(val_k5)/len(sizes):.2f} cands/S1 vs {len(val)/len(sizes):.2f}", flush=True)
    results.append(fit_and_eval(fit, val_k5, sizes, {}, cols, "K=5 (same baseline config)"))

    json.dump(results, open("artifacts/sweep_results.json", "w"), indent=2)
    print("\n| config | val F0.5 | threshold | avg matches/S1 | best iter |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['label']} | {r['macro_f05']:.4f} | {r['t']:.2f} | {r['avg_matches_per_s1']:.3f} | {r['best_iteration']} |")


if __name__ == "__main__":
    main()
