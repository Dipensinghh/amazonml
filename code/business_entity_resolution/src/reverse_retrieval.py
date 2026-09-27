"""Evaluate target-to-S1 rare-token retrieval unioned with frozen forward candidates.

For every country, the full training S1 corpus is the frozen token-pair index and every S2/S3
target queries it. The top-30 token pool is reranked with the frozen alpha; only reverse ranks
1-3 whose S1 belongs to the exact seeded 50k validation scope are retained for evaluation.
`blocking.py` is imported unchanged.

Usage: python -u src/reverse_retrieval.py --jobs 8
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel

import blocking as bl


def validation_scope(sample: int) -> pd.DataFrame:
    """Exact blocking.py validation sample with country attached."""
    s1 = bl.load("S1", "train")
    is_val = bl.make_split(s1["entity_id"])
    ids = set(bl.sample_ids(s1.loc[is_val, "entity_id"].to_numpy(), sample))
    return s1[s1["entity_id"].isin(ids)].reset_index(drop=True)


def top_reverse(r: pd.DataFrame, q: pd.DataFrame, s1: pd.DataFrame, max_r: int) -> pd.DataFrame:
    """Frozen name rerank of a target-query pool, retaining its top max_r S1s."""
    if r.empty:
        return r
    sim = bl.name_sim(q, s1, r["q_row"].to_numpy(), r["t_row"].to_numpy())
    key = r["score"].to_numpy() + bl.FROZEN["alpha"] * sim / 100.0
    out = r.assign(name_sim=sim, key=key).sort_values(
        ["q_row", "key"], ascending=[True, False], kind="stable")
    out["reverse_rank"] = out.groupby("q_row").cumcount().astype(np.int8)
    return out[out["reverse_rank"] < max_r]


def reverse_pairs(scope: pd.DataFrame, jobs: int, chunk: int, max_r: int) -> pd.DataFrame:
    """Run full target-to-S1 retrieval and return reverse pairs owned by sampled S1s."""
    scope_ids = set(scope["entity_id"])
    s1_all = bl.load("S1", "train")
    parts = []
    with Parallel(n_jobs=jobs) as pool:
        for country in sorted(scope["country"].unique()):
            s1 = s1_all[s1_all["country"].eq(country)].reset_index(drop=True)
            t0 = time.perf_counter()
            index = bl.TokenIndex(s1, pool)
            idf = index.for_cap(bl.FROZEN["cap"])
            print(f"{country}: indexed {len(s1):,} S1 in {time.perf_counter() - t0:.0f}s", flush=True)
            id_values = s1["entity_id"].to_numpy()
            for source in bl.TARGETS:
                targets = bl.load(source, "train")
                targets = targets[targets["country"].eq(country)].reset_index(drop=True)
                kept = 0
                for start in range(0, len(targets), chunk):
                    q = targets.iloc[start:start + chunk].reset_index(drop=True)
                    qa, qn = index.queries(q, pool)
                    found = bl.search(bl.weight_queries(qa, qn, idf),
                                      index.tt, top_n=bl.POOL)
                    selected = top_reverse(found, q, s1, max_r)
                    selected_s1 = id_values[selected["t_row"].to_numpy()]
                    use = np.fromiter((x in scope_ids for x in selected_s1), bool, len(selected_s1))
                    if use.any():
                        selected = selected[use]
                        rows = selected["q_row"].to_numpy()
                        part = pd.DataFrame({
                            "s1_id": selected_s1[use],
                            "cand_id": q["entity_id"].to_numpy()[rows],
                            "source": source,
                            "reverse_rank": selected["reverse_rank"].to_numpy(),
                            "reverse_tok_score": selected["score"].to_numpy(np.float32),
                            "reverse_name_sim": selected["name_sim"].to_numpy(np.float32),
                        })
                        parts.append(part); kept += len(part)
                    done = min(start + chunk, len(targets))
                    print(f"  {country}/{source}: {done:,}/{len(targets):,} targets; "
                          f"{kept:,} sampled-S1 pairs", flush=True)
                del targets
            del index, idf
    return pd.concat(parts, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])


def metrics(scope: pd.DataFrame, forward: set, reverse: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    """Recall, candidate cost, and recovery of forward misses for R=1,2,3."""
    country_by_s1 = scope.set_index("entity_id")["country"].to_dict()
    truth_by = {country: set(map(tuple, g[["s1_id", "cand_id"]].to_numpy()))
                for country, g in truth_df.assign(
                    country=truth_df["s1_id"].map(country_by_s1)).groupby("country")}
    truth_by["overall"] = set().union(*truth_by.values())
    scope_n = scope["country"].value_counts().to_dict()
    scope_n["overall"] = len(scope)
    rows = {}
    for r in (1, 2, 3):
        rev = set(map(tuple, reverse.loc[reverse["reverse_rank"] < r, ["s1_id", "cand_id"]].to_numpy()))
        union = forward | rev
        row = {}
        for country, truth in truth_by.items():
            cands = union if country == "overall" else {p for p in union if country_by_s1[p[0]] == country}
            missed = truth - forward
            row[country] = {
                "recall": len(truth & union) / len(truth),
                "candidates_per_s1": len(cands) / scope_n[country],
                "forward_missed_recall": len(missed & rev) / max(len(missed), 1),
                "forward_missed_recovered": len(missed & rev),
                "forward_missed_pairs": len(missed),
            }
        rows[f"R={r}"] = row
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--sample", type=int, default=50_000)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--forward", default=str(bl.ART / "cand_val50k.parquet"))
    ap.add_argument("--out", default=str(bl.ART / "reverse_retrieval_report.json"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    scope = validation_scope(args.sample)
    forward_table = pq.read_table(args.forward, columns=["s1_id", "cand_id"])
    forward = set(zip(forward_table["s1_id"].to_pylist(), forward_table["cand_id"].to_pylist()))
    truth = bl.gt_pairs()
    truth = truth[truth["s1_id"].isin(set(scope["entity_id"]))].reset_index(drop=True)
    reverse = reverse_pairs(scope, args.jobs, args.chunk, 3)
    result = metrics(scope, forward, reverse, truth)

    base = len(set(map(tuple, truth[["s1_id", "cand_id"]].to_numpy())) & forward) / len(truth)
    for values in result.values():
        values["overall"]["lift_points"] = 100 * (values["overall"]["recall"] - base)
        values["gate_pass"] = (values["overall"]["recall"] - base >= 0.02 and
                               values["overall"]["candidates_per_s1"] <= 30)
    report = {"scope": {"s1": len(scope), "true_pairs": len(truth)},
              "forward": {"recall": base, "pairs": len(forward),
                          "candidates_per_s1": len(forward) / len(scope)},
              "reverse_pairs_r3": len(reverse), "results": result,
              "runtime_seconds": time.perf_counter() - t0,
              "gate": ">= +2 overall recall points at <= 30 candidates/S1"}
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pq.write_table(pa.Table.from_pandas(reverse, preserve_index=False),
                   bl.ART / "reverse_val50k_r3.parquet", compression="zstd")

    print(f"\nforward: recall {base:.4f} @ {len(forward) / len(scope):.2f} candidates/S1")
    print("R  overall recall/cands  US recall/cands  India recall/cands  missed recovered  gate")
    for name, values in result.items():
        o, us, ind = values["overall"], values["US"], values["India"]
        print(f"{name[-1]}  {o['recall']:.4f}/{o['candidates_per_s1']:.2f}       "
              f"{us['recall']:.4f}/{us['candidates_per_s1']:.2f}     "
              f"{ind['recall']:.4f}/{ind['candidates_per_s1']:.2f}        "
              f"{o['forward_missed_recall']:.2%}          {values['gate_pass']}")
    print(f"wrote {path}; runtime {report['runtime_seconds']:.0f}s; peak {bl.peak_ram_gb():.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
