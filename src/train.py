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


def attach_labels(feats: pd.DataFrame, gt_path: str) -> pd.DataFrame:
    """Add y=1 for pairs in the ground truth, 0 otherwise."""
    gold = load_gold_pairs(gt_path)
    df = feats.merge(gold, on=["s1_id", "cand_id"], how="left")
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
          fit_s1_frac: float = 1.0, params: dict = None) -> dict:
    """Fit LightGBM on fitting S1s, early-stop (log loss) on validation S1s, tune threshold after ownership.

    fit_s1_frac < 1 keeps a deterministic subset of fitting S1s (full target search space retained).
    Saves the model, and a bundle.json holding the ordered feature schema, best iteration, threshold,
    library versions and settings. Returns the sweep result plus ownership/baseline diagnostics.
    """
    import sklearn
    params = {**PARAMS, **(params or {})}
    cols = feature_columns(df)
    is_val = df["s1_id"].isin(val_ids)
    tr, va = df[~is_val], df[is_val].copy()
    if fit_s1_frac < 1.0:
        keep = tr["s1_id"].drop_duplicates().sample(frac=fit_s1_frac, random_state=42)
        tr = tr[tr["s1_id"].isin(set(keep))]
    model = lgb.LGBMClassifier(**params)
    model.fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    va["p"] = model.predict_proba(va[cols])[:, 1]
    sizes = gold_sizes(gt_path)
    sizes = sizes[sizes.index.isin(val_ids)]
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
    ap.add_argument("--features", required=True, help="parquet: s1_id, cand_id, feature columns (from features.py)")
    ap.add_argument("--gt", default=str(GT_PATH))
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--fit-s1-frac", type=float, default=1.0)
    ap.add_argument("--refit-full", action="store_true", help="after training, refit on all labelled S1s with the best iteration")
    a = ap.parse_args()
    feats = pd.read_parquet(a.features)
    df = attach_labels(feats, a.gt)
    train(df, load_val_ids(), a.gt, a.out_dir, a.fit_s1_frac)
    if a.refit_full:
        refit_full(df, f"{a.out_dir}/bundle.json", a.out_dir)
