"""Train the LightGBM pair classifier on train S1s, early-stop on the frozen validation S1s, tune the threshold."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from ber.predict import META, sweep_threshold
from ber.split import load_val_ids

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
    """All non-meta columns; refuses country-like features."""
    cols = [c for c in df.columns if c not in META]
    bad = FORBIDDEN & set(cols)
    if bad:
        raise ValueError(f"forbidden feature columns: {bad}")
    return cols


def train(df: pd.DataFrame, val_ids: set, gt_path: str, out_dir: str = "artifacts", seed: int = 42) -> dict:
    """Fit LightGBM (train S1s), early-stop on validation S1s, sweep threshold with ownership; save model + threshold."""
    cols = feature_columns(df)
    is_val = df["s1_id"].isin(val_ids)
    tr, va = df[~is_val], df[is_val].copy()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=seed,
                  metric="average_precision", verbose=-1, num_threads=0)
    dtr = lgb.Dataset(tr[cols], tr["y"])
    dva = lgb.Dataset(va[cols], va["y"], reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    va["p"] = booster.predict(va[cols], num_iteration=booster.best_iteration)
    sizes = gold_sizes(gt_path)
    sizes = sizes[sizes.index.isin(val_ids)]
    res = sweep_threshold(va, sizes)
    Path(out_dir).mkdir(exist_ok=True)
    booster.save_model(f"{out_dir}/lgbm.txt", num_iteration=booster.best_iteration)
    json.dump({"t": res["t"], "val_macro_f05": res["macro_f05"], "n_features": len(cols),
               "best_iter": booster.best_iteration}, open(f"{out_dir}/threshold.json", "w"), indent=2)
    imp = pd.Series(booster.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print(imp.head(15))
    print(f"best t={res['t']} val macro F0.5={res['macro_f05']:.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="parquet: s1_id, cand_id, feature columns (from features.py)")
    ap.add_argument("--gt", default="../student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--val-ids", default="splits/val_s1_ids.txt.gz")
    ap.add_argument("--out-dir", default="artifacts")
    a = ap.parse_args()
    feats = pd.read_parquet(a.features)
    train(attach_labels(feats, a.gt), load_val_ids(a.val_ids), a.gt, a.out_dir)
