"""Rule-based fallback matcher over any candidate parquet (s1_id, cand_id).

score = (w1*ts(name_core) + w2*ts(addr_norm) + w3*ts(name_phon)) / (100 * sum(w))
where ts = rapidfuzz token_set_ratio, computed pairwise with cpdist (workers=-1)
in batches. When either address is blank, the address term is dropped and the
weights renormalized, so a blank address neither helps nor hurts.

Then ownership.decide(scores, t) keeps each S2/S3 id only for its best S1.

Dev/tuning (validation):
  python src/fallback.py --root C:/Users/sdipe/amazonmlchallenge \
      --candidates artifacts/cand_val50k.parquet
  -> prints the grid, writes src/fallback_params.json (best weights + t).
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from ownership import decide

FIELDS = ("name_core", "addr_norm", "name_phon")
PARAMS_PATH = Path(__file__).with_name("fallback_params.json")
BATCH = 2_000_000
SEED = 42


# ---------------------------------------------------------------- features
def load_features(paths, ids: pa.Array) -> pa.Table:
    """entity_id + FIELDS + addr_blank for the given ids, from normalized parquets."""
    parts = []
    for p in paths:
        t = pq.read_table(p, columns=["entity_id", *FIELDS, "addr_blank"])
        parts.append(t.filter(pc.is_in(t["entity_id"], value_set=ids)))
    return pa.concat_tables(parts).combine_chunks()


def pair_sims(pairs: pd.DataFrame, s1_feat: pa.Table, tgt_feat: pa.Table,
              batch: int = BATCH, log=print) -> dict:
    """Per-pair token_set_ratio (0-100, float32) for each field, plus a blank-address flag.

    pairs: pandas DataFrame or Arrow table with s1_id, cand_id.
    Pairs whose ids are missing from the feature tables raise KeyError.
    """
    def col(c):  # pandas or Arrow input; Arrow avoids Python strings at test scale
        return pairs[c] if isinstance(pairs, pa.Table) else pa.array(pairs[c].to_numpy())

    si = pc.index_in(col("s1_id"), value_set=s1_feat["entity_id"])
    ti = pc.index_in(col("cand_id"), value_set=tgt_feat["entity_id"])
    if si.null_count or ti.null_count:
        raise KeyError(f"{si.null_count} s1 / {ti.null_count} cand ids have no features")
    si, ti = si.to_numpy(), ti.to_numpy()
    n = len(pairs)
    out = {f: np.empty(n, dtype=np.float32) for f in FIELDS}
    t0 = time.perf_counter()
    for lo in range(0, n, batch):
        hi = min(lo + batch, n)
        a_idx, b_idx = pa.array(si[lo:hi]), pa.array(ti[lo:hi])
        for f in FIELDS:
            a = s1_feat[f].take(a_idx).to_pylist()
            b = tgt_feat[f].take(b_idx).to_pylist()
            out[f][lo:hi] = cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        log(f"  sims {hi:,}/{n:,} pairs ({time.perf_counter() - t0:.0f}s)")
    out["addr_blank"] = (s1_feat["addr_blank"].to_numpy(zero_copy_only=False)[si]
                         | tgt_feat["addr_blank"].to_numpy(zero_copy_only=False)[ti])
    return out


def combine(sims: dict, w) -> np.ndarray:
    """Weighted score in [0, 1]; the address weight is dropped where an address is blank."""
    w1, w2, w3 = w
    num = w1 * sims["name_core"] + w3 * sims["name_phon"]
    w_addr = np.where(sims["addr_blank"], 0.0, w2)
    return ((num + w_addr * sims["addr_norm"]) / (100.0 * (w1 + w3 + w_addr))).astype(np.float32)


# ---------------------------------------------------------------- evaluation
def macro_f05(pred: pd.DataFrame, gt: pd.DataFrame, s1_ids) -> float:
    """Exact macro F0.5 over s1_ids.

    pred / gt: long (s1_id, cand_id) pairs. Per S1: empty pred and empty truth
    -> 1; exactly one of them empty -> 0; else F0.5 from precision and recall.
    """
    s1 = pd.Index(pd.unique(pd.Series(list(s1_ids))))
    pred = pred[["s1_id", "cand_id"]].drop_duplicates()
    gt = gt[["s1_id", "cand_id"]].drop_duplicates()
    pred, gt = pred[pred["s1_id"].isin(s1)], gt[gt["s1_id"].isin(s1)]
    n_pred = pred.groupby("s1_id").size().reindex(s1, fill_value=0).to_numpy(float)
    n_true = gt.groupby("s1_id").size().reindex(s1, fill_value=0).to_numpy(float)
    tp = pred.merge(gt, on=["s1_id", "cand_id"]).groupby("s1_id").size()
    tp = tp.reindex(s1, fill_value=0).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n_pred, tp / n_true
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((n_pred == 0) & (n_true == 0), 1.0, f)
    return float(f.mean())


def gt_pairs(root: Path) -> pd.DataFrame:
    """Train ground truth exploded to (s1_id, cand_id)."""
    gt = pd.read_csv(root / "dataset" / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False, quoting=3)
    p = gt.assign(cand_id=gt["matched_entity_ids"].str.split(",")).explode("cand_id")
    p = p[p["cand_id"].notna() & (p["cand_id"] != "")]
    return p.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "cand_id"]]


def val_sample(root: Path, n: int = 50_000) -> np.ndarray:
    """The blocking study's val S1 sample: seed-42 choice of n from the sorted val ids."""
    split = pq.read_table(root / "artifacts" / "split_train.parquet").to_pandas()
    ids = np.sort(split.loc[split["is_val"], "entity_id"].to_numpy())
    if n <= 0 or n >= len(ids):
        return ids
    return np.sort(np.random.default_rng(SEED).choice(ids, size=n, replace=False))


def load_candidates(path: Path, channels=None, k=None) -> pd.DataFrame:
    """Candidate pairs (s1_id, cand_id); optionally filter channel / rank < k (old k20 file)."""
    t = pq.read_table(path)
    if channels is not None:
        t = t.filter(pc.is_in(pc.cast(t["channel"], pa.string()), value_set=pa.array(channels)))
    if k is not None:
        t = t.filter(pc.less(t["rank"], k))
    return t.select(["s1_id", "cand_id"]).to_pandas().drop_duplicates(ignore_index=True)


# ---------------------------------------------------------------- tuning
WEIGHTS = [(1, 1, 0), (1, 1, 0.5), (1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 1, 0), (1, 0.5, 0.5),
           (1, 0, 0), (1, 0, 1), (1, 1.5, 0.5), (1, 3, 1), (1, 3, 0.5), (1, 4, 1), (1, 3, 2),
           (0.5, 2, 1)]
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 .. 0.95


class FastEval:
    """Integer-coded macro F0.5 for many masks over one fixed candidate set.

    Same definition as macro_f05 (which stays the exact reference); used only to
    make the grid cheap. cands must be deduplicated.
    """

    def __init__(self, cands: pd.DataFrame, gt: pd.DataFrame, s1_ids):
        s1 = pd.Index(np.sort(np.asarray(s1_ids, dtype=object)))
        self.n = len(s1)
        self.s1c = s1.get_indexer(cands["s1_id"])
        self.is_gt = (cands.merge(gt.drop_duplicates(), on=["s1_id", "cand_id"], how="left", indicator=True)
                      ["_merge"].eq("both").to_numpy())
        g = gt.drop_duplicates()
        g = g[g["s1_id"].isin(s1)]
        self.n_true = np.bincount(s1.get_indexer(g["s1_id"]), minlength=self.n).astype(float)

    def __call__(self, mask: np.ndarray) -> float:
        """Macro F0.5 when exactly the candidate rows in mask are predicted."""
        n_pred = np.bincount(self.s1c[mask], minlength=self.n).astype(float)
        tp = np.bincount(self.s1c[mask & self.is_gt], minlength=self.n).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            p, r = tp / n_pred, tp / self.n_true
            f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
        return float(np.where((n_pred == 0) & (self.n_true == 0), 1.0, f).mean())


def grid(cands: pd.DataFrame, sims: dict, gt: pd.DataFrame, s1_ids, log=print) -> pd.DataFrame:
    """F0.5 for every (weights, t), with and without ownership.

    Ownership: the winning S1 of each target does not depend on t, so
    decide(scored, -inf) is computed once per weight combo and
    decide(scored, t) == that winner set restricted to score >= t.
    """
    ev = FastEval(cands, gt, s1_ids)
    rows = []
    for w in WEIGHTS:
        score = combine(sims, w)
        won = decide(cands.assign(score=score), -np.inf)
        is_owner = (cands.merge(won[["s1_id", "cand_id"]], on=["s1_id", "cand_id"], how="left",
                                indicator=True)["_merge"].eq("both").to_numpy())
        for t in THRESHOLDS:
            above = score >= t
            rows.append({"w": w, "t": t, "f05_own": ev(above & is_owner), "f05_raw": ev(above)})
        best = max(rows[-len(THRESHOLDS):], key=lambda r: r["f05_own"])
        log(f"  w={w}: best t={best['t']:.2f} F0.5 own={best['f05_own']:.4f} raw={best['f05_raw']:.4f}")
    return pd.DataFrame(rows)


def main() -> int:
    """Tune weights and t on validation candidates; write fallback_params.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="checkout holding dataset/ and artifacts/")
    ap.add_argument("--candidates", required=True, help="parquet with s1_id, cand_id (relative to root ok)")
    ap.add_argument("--channels", nargs="*", help="old k20 file only: keep these channels")
    ap.add_argument("--k", type=int, help="old k20 file only: keep rank < k")
    ap.add_argument("--sample", type=int, default=50_000)
    ap.add_argument("--no-save", action="store_true", help="do not write fallback_params.json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(a.root)
    t0 = time.perf_counter()
    s1_ids = val_sample(root, a.sample)
    cands = load_candidates(root / a.candidates, a.channels, a.k)
    extra = set(cands["s1_id"].unique()) - set(s1_ids)
    if extra:
        raise ValueError(f"{len(extra)} candidate S1s are outside the val sample, e.g. {sorted(extra)[:3]}")
    gt = gt_pairs(root)
    gt = gt[gt["s1_id"].isin(set(s1_ids))]
    have = cands.merge(gt, on=["s1_id", "cand_id"])
    print(f"S1 {len(s1_ids):,} | candidates {len(cands):,} ({len(cands) / len(s1_ids):.1f}/S1) | "
          f"GT pairs {len(gt):,}, recall ceiling {len(have) / len(gt):.4f} | "
          f"F0.5 if we output exactly the GT-covered candidates: "
          f"{macro_f05(have, gt, s1_ids):.4f}", flush=True)

    art = root / "artifacts"
    ids1 = pa.array(s1_ids)
    s1_feat = load_features([art / "train_s1_norm.parquet"], ids1)
    tgt_feat = load_features([art / "train_s2_norm.parquet", art / "train_s3_norm.parquet"],
                             pa.array(cands["cand_id"].unique()))
    print(f"features loaded ({time.perf_counter() - t0:.0f}s)", flush=True)
    sims = pair_sims(cands, s1_feat, tgt_feat)
    res = grid(cands, sims, gt, s1_ids)
    best = res.loc[res["f05_own"].idxmax()]
    best_raw = res.loc[res["f05_raw"].idxmax()]
    same = res[(res["w"] == best["w"]) & (res["t"] == best["t"])].iloc[0]
    # cross-check the chosen point with the exact reference path (ownership.decide + macro_f05)
    ref = macro_f05(decide(cands.assign(score=combine(sims, best["w"])), best["t"]), gt, s1_ids)
    assert abs(ref - best["f05_own"]) < 1e-9, (ref, best["f05_own"])
    print(f"reference check (decide + macro_f05) at best point: {ref:.6f} == grid {best['f05_own']:.6f}")
    print(f"\nBEST with ownership: w={best['w']} t={best['t']:.2f} macro F0.5 {best['f05_own']:.4f} "
          f"(same w/t without ownership: {same['f05_raw']:.4f})")
    print(f"best without ownership: w={best_raw['w']} t={best_raw['t']:.2f} F0.5 {best_raw['f05_raw']:.4f}")
    print(f"runtime {time.perf_counter() - t0:.0f}s")
    if not a.no_save:
        PARAMS_PATH.write_text(json.dumps({
            "weights": list(best["w"]), "t": float(best["t"]), "val_f05": round(float(best["f05_own"]), 4),
            "candidates": str(a.candidates), "channels": a.channels, "k": a.k}, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        print(f"wrote {PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
