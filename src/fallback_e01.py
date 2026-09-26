"""E01 rules-based sample baseline (D; the submission fallback is A's src/fallback.py): word-level TF-IDF retrieval + a hand-weighted score + tuned threshold + global ownership.

Stand-in for the real blocker/features (owned by C and B): it uses its own light text cleaning and a
sample of validation S1s searched against ALL train S2/S3 records of the same country and source, so the
distractor universe is intact. No learning is involved; the only fitted values are the name/address
weight and the threshold, both chosen on the validation sample.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer

from predict import assign_owners, sweep_threshold, ownership_report
from valsplit import load_val_ids
from train import gold_sizes, load_gold_pairs

from valsplit import TRAIN_DIR, GT_PATH
DATA = str(TRAIN_DIR)
NAME_STOP = {"pvt", "ltd", "private", "limited", "llc", "inc", "corp", "corporation", "co", "company", "sarl",
             "sas", "sasu", "eurl", "sa", "gmbh", "and", "the", "llp", "opc", "dba", "fka", "of"}
ADDR_MAP = {"rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
            "ct": "court", "hwy": "highway", "apt": "apartment", "ste": "suite", "no": "number", "nr": "near"}


def clean(s: str) -> str:
    """Transliterate to ASCII, lowercase, keep letters/digits only (one space between tokens)."""
    return re.sub(r"[^a-z0-9]+", " ", anyascii(s).lower()).strip()


def clean_addr(s: str) -> str:
    """As clean(), plus expansion of a few common address abbreviations."""
    return " ".join(ADDR_MAP.get(t, t) for t in clean(s).split())


def read(path: str) -> pd.DataFrame:
    """Read a source TSV as strings (tab, no quoting, blanks preserved)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def topk_rows(S: sp.csr_matrix, k: int):
    """Top-k (row, col, score) triples per row of a sparse similarity chunk."""
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        a, b = S.indptr[i], S.indptr[i + 1]
        if a == b:
            continue
        v, c = S.data[a:b], S.indices[a:b]
        if b - a > k:
            idx = np.argpartition(-v, k - 1)[:k]
            v, c = v[idx], c[idx]
        rows.append(np.full(len(v), i)); cols.append(c); vals.append(v)
    if not rows:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def rowdot(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    """Row-aligned cosine (A and B rows are L2-normalised): sum of elementwise product per row."""
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def build_candidates(s1: pd.DataFrame, targets: pd.DataFrame, k: int, chunk: int = 500) -> pd.DataFrame:
    """Candidates for one (country, source): union of name top-k and address top-k, with both cosines for every pair."""
    tn = targets["business_name"].map(clean).values
    ta = targets["business_address"].map(clean_addr).values
    qn = s1["business_name"].map(clean).values
    qa = s1["business_address"].map(clean_addr).values
    vn = TfidfVectorizer(token_pattern=r"[a-z0-9]+", stop_words=list(NAME_STOP), sublinear_tf=True,
                         max_df=0.02, dtype=np.float32)
    va = TfidfVectorizer(token_pattern=r"[a-z0-9]+", sublinear_tf=True, max_df=0.02, dtype=np.float32)
    Tn, Ta = vn.fit_transform(tn), va.fit_transform(ta)
    Qn, Qa = vn.transform(qn), va.transform(qa)
    TnT, TaT = Tn.T.tocsr(), Ta.T.tocsr()
    out = []
    for s in range(0, len(s1), chunk):
        sl = slice(s, s + chunk)
        pairs = set()
        for Q, TT in ((Qn[sl], TnT), (Qa[sl], TaT)):
            r, c, _ = topk_rows((Q @ TT).tocsr(), k)
            pairs.update(zip((r + s).tolist(), c.tolist()))
        if not pairs:
            continue
        qi, ti = (np.array(x) for x in zip(*pairs))
        out.append(pd.DataFrame({
            "s1_id": s1["entity_id"].values[qi], "cand_id": targets["entity_id"].values[ti],
            "name_cos": rowdot(Qn[qi], Tn[ti]), "addr_cos": rowdot(Qa[qi], Ta[ti]),
            "addr_blank": (ta[ti] == "").astype(np.int8) | (qa[qi] == "").astype(np.int8)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def rule_score(df: pd.DataFrame, w: float) -> pd.Series:
    """Hand-weighted score: w*name + (1-w)*address; if either address is blank, use the name cosine alone."""
    both = w * df["name_cos"] + (1 - w) * df["addr_cos"]
    return both.where(df["addr_blank"] == 0, df["name_cos"])


def run(n_s1: int, k: int, out_dir: str, seed: int = 42) -> dict:
    """Sample validation S1s, retrieve against all train targets, tune weight + threshold, report macro F0.5."""
    t0 = time.time()
    val = load_val_ids()
    s1 = read(f"{DATA}/train_source1.tsv")
    s1 = s1[s1["entity_id"].isin(val)].sample(n=n_s1, random_state=seed).reset_index(drop=True)
    print(f"sample {len(s1)} val S1s; countries {s1['country'].value_counts().to_dict()}", flush=True)
    parts = []
    for src in ("source2", "source3"):
        tg = read(f"{DATA}/train_{src}.tsv")
        for ctry, q in s1.groupby("country"):
            t1 = time.time()
            tt = tg[tg["country"] == ctry].reset_index(drop=True)
            c = build_candidates(q.reset_index(drop=True), tt, k)
            c["source"] = src
            parts.append(c)
            print(f"  {src} {ctry}: targets={len(tt):,} queries={len(q):,} pairs={len(c):,} {time.time()-t1:.0f}s", flush=True)
        del tg
    cand = pd.concat(parts, ignore_index=True)
    gold = load_gold_pairs(f"{DATA}/train_ground_truth.tsv")
    cand = cand.merge(gold, on=["s1_id", "cand_id"], how="left")
    cand["y"] = cand["y"].fillna(0).astype("int8")
    sizes = gold_sizes(f"{DATA}/train_ground_truth.tsv")
    sizes = sizes[sizes.index.isin(set(s1["entity_id"]))]
    n_true = int(sizes.sum())
    diag = {"n_s1": len(s1), "pairs": len(cand), "mean_cands": len(cand) / len(s1),
            "pair_recall": float(cand["y"].sum() / n_true), "n_true": n_true,
            "positive_rate": float(cand["y"].mean())}
    best = None
    for w in (0.5, 0.35, 0.25, 0.2, 0.15):
        cand["p"] = rule_score(cand, w)
        r = sweep_threshold(cand, sizes)
        print(f"  w_name={w}: best t={r['t']} macro F0.5={r['macro_f05']:.4f}", flush=True)
        if best is None or r["macro_f05"] > best[1]["macro_f05"]:
            best = (w, r)
    w, r = best
    cand["p"] = rule_score(cand, w)
    own = ownership_report(cand, sizes, r["t"])
    res = {**diag, "w_name": w, "threshold": r["t"], "macro_f05": r["macro_f05"], "ownership": own,
           "all_empty_baseline": float((sizes == 0).mean()), "k": k, "runtime_s": round(time.time() - t0)}
    Path(out_dir).mkdir(exist_ok=True)
    cand.to_parquet(f"{out_dir}/fallback_val_cands.parquet")
    json.dump(res, open(f"{out_dir}/fallback_result.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", default="artifacts")
    a = ap.parse_args()
    run(a.n_s1, a.k, a.out_dir)
