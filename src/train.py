"""Train the LightGBM pair classifier on train S1s, early-stop on the frozen validation S1s, tune the threshold."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import pyarrow as pa

from features import feature_columns as b_feature_columns
from predict import META, ownership_report, sweep_threshold
from valsplit import GT_PATH, load_val_ids

FORBIDDEN = {"country"}  # France is unseen: no feature may encode country


def load_gold_pairs(gt_path: str) -> pd.DataFrame:
    """Explode the ground truth into a (s1_id, cand_id, y=1) table."""
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    gt["cand_id"] = gt["matched_entity_ids"].str.split(",")
    e = gt.explode("cand_id")
    e = e[e["cand_id"].notna() & (e["cand_id"] != "")]
    out = e.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "cand_id"]]
    out["y"] = 1
    return out


def gold_sizes(gt_path: str) -> pd.Series:
    """Number of true matches per S1 id (0 for singletons)."""
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    n = gt["matched_entity_ids"].map(lambda x: 0 if not x else x.count(",") + 1)
    return pd.Series(n.values, index=gt["source1_entity_id"].values, name="ngold")


def attach_labels(feats: pd.DataFrame, gt_path: str = None) -> pd.DataFrame:
    """Set y from B's `label` column when present (built from the ground truth), else from the GT file.

    Also asserts one row per (s1_id, cand_id), since duplicate pairs would double-count in scoring.
    """
    df = feats
    key = pd.util.hash_pandas_object(df[["s1_id", "cand_id"]], index=False)
    if key.duplicated().any():
        raise ValueError("duplicate (s1_id, cand_id) rows: collapse the candidates first")
    if "label" in df.columns:
        df = df.assign(y=df["label"].astype("int8"))
    else:
        gold = load_gold_pairs(gt_path)
        df = df.merge(gold, on=["s1_id", "cand_id"], how="left")
        df["y"] = df["y"].fillna(0).astype("int8")
    return df


def feature_columns(df: pd.DataFrame) -> list:
    """Model-input columns: B's contract (features.feature_columns) minus this stage's own columns (y, p, source).

    B's list excludes keys, label and provenance by name, which is what protects against training on the
    answer; a dtype filter would keep the int8 label. Country-like columns are refused outright.
    """
    bad = FORBIDDEN & set(df.columns)
    if bad:
        raise ValueError(f"forbidden feature columns: {bad}")
    cols = b_feature_columns(pa.Table.from_pandas(df.iloc[:0], preserve_index=False))
    return [c for c in cols if c not in META]


PARAMS = dict(objective="binary", n_estimators=800, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, reg_lambda=1.0, random_state=42, n_jobs=4, verbosity=-1)


def train(df: pd.DataFrame, val_ids: set, gt_path: str, out_dir: str = "artifacts",
          fit_s1_frac: float = 1.0, params: dict = None, scope_ids: set = None,
          early_stop: int = 50) -> dict:
    """Fit LightGBM on fitting S1s, early-stop (log loss) on validation S1s, tune threshold after ownership.

    scope_ids: the S1s actually evaluated (a sample of validation S1s, including any with zero candidates).
    Defaults to the validation S1s present in df, which under-counts zero-candidate S1s: pass the full
    sample id list for a real run.
    fit_s1_frac < 1 keeps a deterministic subset of fitting S1s (full target search space retained).
    early_stop: patience in rounds on validation log loss (v3 used 100).
    Saves the model, and a bundle.json holding the ordered feature schema, best iteration, threshold,
    library versions and settings. Returns the sweep result plus ownership/baseline diagnostics.
    """
    import sklearn
    params = {**PARAMS, **(params or {})}
    cols = feature_columns(df)
    is_val = df["s1_id"].isin(val_ids)
    tr, va = df[~is_val], df[is_val].copy()
    if tr.empty or va.empty:
        raise ValueError(f"need pairs from BOTH fitting and validation S1s (fit={len(tr)}, val={len(va)}); "
                         "a validation-only candidate set (e.g. dev_candidates) cannot train the model")
    if fit_s1_frac < 1.0:
        keep = tr["s1_id"].drop_duplicates().sample(frac=fit_s1_frac, random_state=42)
        tr = tr[tr["s1_id"].isin(set(keep))]
    model = lgb.LGBMClassifier(**params)
    model.fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(early_stop), lgb.log_evaluation(100)])
    va["p"] = model.predict_proba(va[cols])[:, 1]
    sizes = gold_sizes(gt_path)
    scope = set(scope_ids) if scope_ids is not None else set(va["s1_id"])
    sizes = sizes[sizes.index.isin(scope & set(val_ids))]
    n_zero = int((~sizes.index.isin(va["s1_id"].unique())).sum())
    print(f"evaluating {len(sizes):,} val S1s; {n_zero:,} have no candidates and count as empty predictions", flush=True)
    res = sweep_threshold(va, sizes)
    res["ownership"] = ownership_report(va, sizes, res["t"])
    res["all_empty_baseline"] = float((sizes == 0).mean())
    Path(out_dir).mkdir(exist_ok=True)
    model.booster_.save_model(f"{out_dir}/lgbm.txt")
    bundle = {"features": cols, "best_iteration": int(model.best_iteration_), "threshold": res["t"],
              "val_macro_f05": res["macro_f05"], "ownership": res["ownership"],
              "all_empty_baseline": res["all_empty_baseline"], "params": params,
              "fit_s1_frac": fit_s1_frac, "n_fit_pairs": int(len(tr)), "n_val_pairs": int(len(va)),
              "pos_rate_fit": float(tr["y"].mean()),
              "versions": {"lightgbm": lgb.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__}}
    json.dump(bundle, open(f"{out_dir}/bundle.json", "w"), indent=2)
    json.dump({"t": res["t"]}, open(f"{out_dir}/threshold.json", "w"))
    imp = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print(imp.head(15))
    print(f"best t={res['t']} val macro F0.5={res['macro_f05']:.4f} | ownership {res['ownership']} | all-empty {res['all_empty_baseline']:.4f}")
    return res


def refit_full(df: pd.DataFrame, bundle_path: str = "artifacts/bundle.json", out_dir: str = "artifacts") -> None:
    """Final refit on ALL labelled S1s with the selected iteration count (no early stopping); keeps the tuned threshold."""
    b = json.load(open(bundle_path))
    cols = b["features"]
    params = {**b["params"], "n_estimators": b["best_iteration"]}
    model = lgb.LGBMClassifier(**params).fit(df[cols], df["y"])
    model.booster_.save_model(f"{out_dir}/lgbm_full.txt")
    b["refit_full_pairs"] = int(len(df))
    json.dump(b, open(bundle_path, "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="fitting-side feature parquet (features.py output)")
    ap.add_argument("--extra-features", nargs="*", default=[], help="validation-side feature parquet(s) to concatenate")
    ap.add_argument("--gt", default=str(GT_PATH))
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--fit-s1-frac", type=float, default=1.0)
    ap.add_argument("--scope-ids", help="text file of the S1 ids evaluated (one per line)")
    ap.add_argument("--scope-sample", type=int, default=0,
                    help="evaluate exactly blocking's validation sample of this many S1s (e.g. 50000); zero-candidate S1s count")
    ap.add_argument("--refit-full", action="store_true", help="after training, refit on all labelled S1s with the best iteration")
    # LightGBM settings (defaults = PARAMS; v3: --num-leaves 255 --n-estimators 3000 --early-stop 100)
    ap.add_argument("--num-leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--n-estimators", type=int, default=PARAMS["n_estimators"], help="maximum trees")
    ap.add_argument("--learning-rate", type=float, default=PARAMS["learning_rate"])
    ap.add_argument("--early-stop", type=int, default=50, help="early-stopping patience (rounds)")
    ap.add_argument("--n-jobs", type=int, default=PARAMS["n_jobs"], help="LightGBM threads")
    a = ap.parse_args()
    params = {"num_leaves": a.num_leaves, "n_estimators": a.n_estimators,
              "learning_rate": a.learning_rate, "n_jobs": a.n_jobs}
    feats = pd.concat([pd.read_parquet(f) for f in [a.features, *a.extra_features]], ignore_index=True)
    df = attach_labels(feats, a.gt)
    del feats
    if a.scope_ids:
        scope = {l.strip() for l in open(a.scope_ids) if l.strip()}
    elif a.scope_sample:
        from valsplit import sample_val_scope
        scope = sample_val_scope(a.scope_sample)
    else:
        scope = None
    train(df, load_val_ids(), a.gt, a.out_dir, a.fit_s1_frac, params=params, scope_ids=scope,
          early_stop=a.early_stop)
    if a.refit_full:
        refit_full(df, f"{a.out_dir}/bundle.json", a.out_dir)
