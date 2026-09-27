"""Error analysis for the best validation config: refit it with its selected iteration count, score the
val50k candidates, apply threshold + global ownership, then classify every miss and false merge.

Misses (true pairs not predicted) are split into: not retrieved (never a candidate), retrieved but
scored below t, and scored >= t but lost the target to another S1 under ownership. False merges
(predicted pairs not in the ground truth) are split into: target truly belongs to another S1
(wrong owner) and target belongs to no S1 (distractor). Samples 20 of each with raw text.

Usage: python -u src/error_analysis.py --leaves 255 --trees 1648 --t 0.70
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from common import read_tsv, source_path
from predict import assign_owners
from train import PARAMS, attach_labels, feature_columns, load_gold_pairs
from valsplit import GT_PATH, sample_val_scope

DROP = ["rev_rank", "rev_count", "rev_score_gap"]
SHOW = ["name_core_token_set", "addr_norm_token_set", "num_agree", "num_conflict", "retr_rank", "addr_blank"]


def raw_text(ids: set) -> pd.DataFrame:
    """Original business_name / business_address / country for the given ids, from the train TSVs."""
    want = pa.array(sorted(ids))
    out = []
    for src in ("S1", "S2", "S3"):
        t = read_tsv(source_path("train", src))
        out.append(t.filter(pc.is_in(t["entity_id"], value_set=want)).to_pandas())
        del t
    return pd.concat(out, ignore_index=True).set_index("entity_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--trees", type=int, default=1648)
    ap.add_argument("--t", type=float, default=0.70)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    fit = attach_labels(pd.read_parquet("artifacts/cand_fit200k_features.parquet"), str(GT_PATH))
    cols = [c for c in feature_columns(fit) if c not in DROP]
    model = lgb.LGBMClassifier(**{**PARAMS, "num_leaves": a.leaves, "n_estimators": a.trees})
    model.fit(fit[cols], fit["y"])
    del fit
    val = attach_labels(pd.read_parquet("artifacts/cand_val50k_features.parquet"), str(GT_PATH))
    val["p"] = model.predict_proba(val[cols])[:, 1].astype(np.float32)
    val[["s1_id", "cand_id", "y", "p"]].to_parquet("artifacts/val_scored_best.parquet")

    scope = sample_val_scope(50_000)
    gold = load_gold_pairs(str(GT_PATH))
    owner_of = gold.set_index("cand_id")["s1_id"]
    gold = gold[gold["s1_id"].isin(scope)]
    pred = assign_owners(val, a.t)
    pred_keys = set(zip(pred["s1_id"], pred["cand_id"]))
    cand_keys = set(zip(val["s1_id"], val["cand_id"]))
    p_of = dict(zip(zip(val["s1_id"], val["cand_id"]), val["p"]))

    miss = gold[[k not in pred_keys for k in zip(gold["s1_id"], gold["cand_id"])]].copy()
    def miss_kind(r):
        k = (r.s1_id, r.cand_id)
        if k not in cand_keys:
            return "not retrieved"
        return "below threshold" if p_of[k] < a.t else "lost ownership"
    miss["kind"] = [miss_kind(r) for r in miss.itertuples()]
    miss["p"] = [p_of.get((r.s1_id, r.cand_id), np.nan) for r in miss.itertuples()]

    fm = pred.merge(val[["s1_id", "cand_id", "y"]], on=["s1_id", "cand_id"])
    fm = fm[fm["y"] == 0].copy()
    fm["true_owner"] = fm["cand_id"].map(owner_of)
    fm["kind"] = np.where(fm["true_owner"].isna(), "distractor (no S1)", "belongs to another S1")

    summary = {"threshold": a.t, "true_pairs": len(gold), "predicted_pairs": len(pred),
               "misses": len(miss), "miss_kinds": miss["kind"].value_counts().to_dict(),
               "false_merges": len(fm), "false_merge_kinds": fm["kind"].value_counts().to_dict()}

    ms = miss.sample(n=min(a.n, len(miss)), random_state=42)
    fs = fm.sample(n=min(a.n, len(fm)), random_state=42)
    ids = set(ms["s1_id"]) | set(ms["cand_id"]) | set(fs["s1_id"]) | set(fs["cand_id"])
    txt = raw_text(ids)
    feats = val.set_index(["s1_id", "cand_id"])[SHOW]

    def rows(df):
        out = []
        for r in df.itertuples():
            k = (r.s1_id, r.cand_id)
            f = feats.loc[k].round(2).to_dict() if k in feats.index else {}
            out.append({"s1_id": r.s1_id, "cand_id": r.cand_id, "kind": r.kind,
                        "p": None if pd.isna(r.p) else round(float(r.p), 3),
                        "country": txt.loc[r.s1_id, "country"],
                        "s1_name": txt.loc[r.s1_id, "business_name"], "s1_addr": txt.loc[r.s1_id, "business_address"],
                        "cand_name": txt.loc[r.cand_id, "business_name"], "cand_addr": txt.loc[r.cand_id, "business_address"],
                        "features": f})
        return out

    res = {"summary": summary, "misses": rows(ms), "false_merges": rows(fs)}
    json.dump(res, open("artifacts/error_analysis.json", "w"), indent=2, ensure_ascii=False, default=str)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
