"""Diagnose India misses and test bounded frozen-blocker variants on the 50k val scope.

This is an experiment driver only; it imports the frozen blocker without changing it. All
variants keep K=10 plus the blank-address quota, so candidate counts stay near 24/S1.

Usage: python -u src/india_retrieval.py --jobs 8
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import blocking as bl


def canonical(s: pd.Series) -> pd.Series:
    """Small transliteration equivalence key requested for the India experiment."""
    return (s.str.replace("ph", "f", regex=False)
             .str.replace("w", "v", regex=False)
             .str.replace("aa", "a", regex=False)
             .str.replace(r"([a-z])\1+", r"\1", regex=True))


def augment(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Add experiment text columns while retaining every frozen-blocker input column."""
    if mode == "base":
        return df
    out = df.copy()
    if mode == "translit_variants":
        out["exp_name"] = (out["name_core"] + " " + canonical(out["name_core"]) + " "
                           + canonical(out["name_phon"]))
        out["exp_addr"] = (out["addr_norm"] + " " + canonical(out["addr_norm"]) + " "
                           + canonical(out["addr_phon"]))
    elif mode == "phonetic_address_pairs":
        out["exp_name"] = out["name_core"]
        out["exp_addr"] = out["addr_norm"] + " " + out["addr_phon"]
    else:
        raise ValueError(mode)
    return out


def run_config(q_base: pd.DataFrame, mode: str, caps: list[int], jobs: int) -> dict[int, pd.DataFrame]:
    """Run build_pools on India only with a temporary loader/column override."""
    original_load, original_addr, original_name = bl.load, bl.ADDR_COL, bl.NAME_COL

    def load(src: str, split: str = "train") -> pd.DataFrame:
        return augment(original_load(src, split), mode)

    try:
        bl.load = load
        if mode != "base":
            bl.ADDR_COL, bl.NAME_COL = "exp_addr", "exp_name"
        q = augment(q_base, mode)
        clock = {"fixed": 0.0}
        with Parallel(n_jobs=jobs) as pool:
            return bl.build_pools(q, "train", caps, pool, clock,
                                  select={"alpha": bl.FROZEN["alpha"], "k": bl.FROZEN["k"]})
    finally:
        bl.load, bl.ADDR_COL, bl.NAME_COL = original_load, original_addr, original_name


def recall(pool: pd.DataFrame, q_ids: np.ndarray, truth: set[tuple[str, str]]) -> dict:
    """Pair recall and candidate count for one selected pool."""
    pairs = set(zip(q_ids[pool["q_row"].to_numpy()], pool["cand_id"]))
    return {"recall": len(pairs & truth) / len(truth),
            "candidates_per_s1": len(pool) / len(q_ids), "pairs": len(pool)}


def classify_misses(q: pd.DataFrame, missed: pd.DataFrame, cap: int) -> tuple[dict, dict]:
    """Classify misses using mutually exclusive evidence from the normalized true pair."""
    qidx = q.set_index("entity_id")
    targets = {}
    frames = {}
    for src in bl.TARGETS:
        t = bl.load(src, "train")
        t = t[t["country"].eq("India")].reset_index(drop=True)
        frames[src] = t
        wanted = set(missed.loc[missed["cand_id"].str.startswith(src + "-"), "cand_id"])
        targets.update(t[t["entity_id"].isin(wanted)].set_index("entity_id").to_dict("index"))

    qa, ta, qn, tn, nonlatin, blank = [], [], [], [], [], []
    for s1, cand in missed.itertuples(index=False):
        a, b = qidx.loc[s1], targets[cand]
        qa.append(a["addr_norm"]); ta.append(b["addr_norm"])
        qn.append(a["name_core"]); tn.append(b["name_core"])
        nonlatin.append(bool(b["name_non_latin"] or b["addr_non_latin"]))
        blank.append(not b["addr_norm"])
    core_sim = cpdist(qn, tn, scorer=fuzz.token_set_ratio, workers=-1)
    qphon = [qidx.loc[s]["name_phon"] for s in missed["s1_id"]]
    tphon = [targets[c]["name_phon"] for c in missed["cand_id"]]
    phon_sim = cpdist(qphon, tphon, scorer=fuzz.token_set_ratio, workers=-1)
    translit = np.asarray(nonlatin) & (phon_sim >= 80) & ((phon_sim - core_sim) >= 15)

    # Determine whether exact frozen tokens exist but every shared token is above the cap.
    high_df = np.zeros(len(missed), dtype=bool)
    for src, target in frames.items():
        rows = np.flatnonzero(missed["cand_id"].str.startswith(src + "-").to_numpy())
        active = rows[~np.asarray(blank)[rows] & ~translit[rows]]
        if not len(active):
            continue
        common_a, common_n = bl.common_words(target["addr_norm"]), bl.common_words(target["name_core"])
        shared_by_row, interest = {}, set()
        for i in active:
            a, b = qidx.loc[missed.iloc[i]["s1_id"]], targets[missed.iloc[i]["cand_id"]]
            shared = ((set(bl.addr_tokens(a["addr_norm"], common_a)) &
                       set(bl.addr_tokens(b["addr_norm"], common_a))) |
                      (set(bl.name_tokens(a["name_core"], common_n)) &
                       set(bl.name_tokens(b["name_core"], common_n))))
            shared_by_row[i] = shared
            interest.update(shared)
        df = Counter()
        if interest:
            for addr, name in zip(target["addr_norm"], target["name_core"]):
                toks = set(bl.addr_tokens(addr, common_a)) | set(bl.name_tokens(name, common_n))
                df.update(toks & interest)
        for i, shared in shared_by_row.items():
            high_df[i] = bool(shared) and min(df[x] for x in shared) > cap

    category = np.full(len(missed), "other", dtype=object)
    category[high_df] = "generic_tokens_dropped_by_cap"
    category[translit] = "transliteration_mismatch"
    category[np.asarray(blank)] = "blank_address"
    counts = pd.Series(category).value_counts().to_dict()
    examples = {c: missed.iloc[np.flatnonzero(category == c)[:5]].to_dict("records")
                for c in sorted(set(category))}
    return {k: int(v) for k, v in counts.items()}, examples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--sample", type=int, default=50_000)
    ap.add_argument("--out", default=str(bl.ART / "india_retrieval_report.json"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    s1 = bl.load("S1", "train")
    is_val = bl.make_split(s1["entity_id"])
    sample = set(bl.sample_ids(s1.loc[is_val, "entity_id"].to_numpy(), args.sample))
    q = s1[s1["entity_id"].isin(sample) & s1["country"].eq("India")].reset_index(drop=True)
    gt = bl.gt_pairs()
    gt = gt[gt["s1_id"].isin(set(q["entity_id"]))][["s1_id", "cand_id"]]
    truth = set(map(tuple, gt.to_numpy()))
    q_ids = q["entity_id"].to_numpy()
    print(f"India scope: {len(q):,} S1, {len(truth):,} true pairs", flush=True)

    results, pools = {}, {}
    base = run_config(q, "base", [200, 500, 1000], args.jobs)
    for cap, pool in base.items():
        name = f"india_cap_{cap}"
        pools[name], results[name] = pool, recall(pool, q_ids, truth)
    for mode in ("translit_variants", "phonetic_address_pairs"):
        pool = run_config(q, mode, [200], args.jobs)[200]
        pools[mode], results[mode] = pool, recall(pool, q_ids, truth)

    baseline_pairs = set(zip(q_ids[pools["india_cap_200"]["q_row"].to_numpy()],
                             pools["india_cap_200"]["cand_id"]))
    missed = pd.DataFrame(sorted(truth - baseline_pairs), columns=["s1_id", "cand_id"])
    counts, examples = classify_misses(q, missed, 200)
    base_recall = results["india_cap_200"]["recall"]
    for value in results.values():
        value["lift_points"] = 100 * (value["recall"] - base_recall)
        value["gains_two_points"] = value["recall"] - base_recall >= 0.02
    report = {"scope": {"india_s1": len(q), "true_pairs": len(truth), "missed_pairs": len(missed)},
              "miss_categories": counts, "miss_examples": examples, "experiments": results,
              "runtime_seconds": time.perf_counter() - t0}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nconfig                         recall   cands/S1   lift")
    for name, value in results.items():
        print(f"{name:<30}{value['recall']:.4f}   {value['candidates_per_s1']:8.2f}   "
              f"{value['lift_points']:+.2f} pp")
    print(f"\nmiss categories: {counts}\nwrote {out}\nruntime {report['runtime_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
