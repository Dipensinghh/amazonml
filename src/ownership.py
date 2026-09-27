"""Ownership rule: each S2/S3 id is assigned to at most one S1.

EDA: every S2/S3 id matches at most one S1, so for each target we keep only its
highest-scoring S1 (ties -> lexicographically smaller S1 id), and only if that
score >= t.
"""
import pandas as pd


def decide(scores_df: pd.DataFrame, t: float) -> pd.DataFrame:
    """Assign each cand_id to its best S1 and keep the pair if score >= t.

    scores_df: columns s1_id, cand_id, score (duplicate pairs allowed; the max
    score wins). Returns columns s1_id, cand_id, score sorted by s1_id, cand_id.
    Filtering by t before the argmax gives the same result as after it (the
    winner's score is the target's max, so it survives iff max >= t) and makes
    the sort much smaller.
    """
    missing = {"s1_id", "cand_id", "score"} - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df: missing columns {sorted(missing)}")
    df = scores_df[["s1_id", "cand_id", "score"]]
    if df["score"].isna().any():
        raise ValueError("scores_df: NaN scores")
    df = df[df["score"] >= t]
    df = df.sort_values(["cand_id", "score", "s1_id"], ascending=[True, False, True], kind="mergesort")
    df = df.drop_duplicates("cand_id", keep="first")
    return df.sort_values(["s1_id", "cand_id"], ignore_index=True)
