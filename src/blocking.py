"""Stage 2 — candidate generation (blocking) and recall evaluation on the validation split.

Validation split: 20% of train S1 ids (seed 42), cached in artifacts/split_train.parquet.
For speed the recall study uses a fixed random sample of val S1s (default 50,000, seed 42);
queries are still searched against ALL train S2/S3 rows of their country, so unmatched rows
and non-val entities stay in as distractors.

Channels (per country — open set, taken from the query rows — and per target source):
  A  char 3-gram TF-IDF (char_wb) on name_core            -> top-k per source
  P  char 3-gram TF-IDF on name_phon (phonetic skeleton)  -> top-k per source
  B  word TF-IDF on addr_phon                             -> top-k per source
  C  exact acronym: acronym(S1) == single-token name(S2/S3) and vice versa,
     skipping keys shared by more than --acr-cap target rows
Top-20 is computed once per channel; k = 3/5/10/20 are rank slices of it.
Names with a dba/fka part are indexed twice (main + alternate) and deduped per S1.

Speed: sparse_dot_topn's own threading barely scales on this machine (122 q/s at 1 thread,
161 q/s at 20), so query chunks run in a joblib process pool (target matrix memory-mapped).

Outputs:
  artifacts/split_train.parquet               entity_id, is_val
  artifacts/val_blocking_sample_k20.parquet   s1_id, cand_id, channel, score, rank
Prints progress per chunk, a recall / avg-candidates table after each channel,
per-channel timing, peak RAM, and an extrapolated full-test runtime per channel.

Usage: python -u src/blocking.py [--sample 50000] [--jobs 10] [--chunk 2000]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import ART, DATA, SEED, norm_path, peak_ram_gb, read_tsv, utf8_stdout

KS = (3, 5, 10, 20)
KMAX = max(KS)
TARGETS = ("S2", "S3")
CHANNELS = ("A", "P", "B", "C")
TEST_S1 = 1_732_544  # rows in dataset/test/test_source1.tsv (for runtime extrapolation)
# channel -> (text column, optional alternate-name column, vectorizer kwargs)
TFIDF_CHANNELS = {
    "A": ("name_core", "name_alt_core",
          dict(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, max_df=0.2)),
    "P": ("name_phon", "name_alt_phon",
          dict(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, max_df=0.2)),
    "B": ("addr_phon", None,
          dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, max_df=0.1)),
}
# (stage label, channels run in that stage); combos reported once their channels exist
STAGES = (("A+P (names)", ("A", "P")), ("B (address)", ("B",)), ("C (acronym)", ("C",)))
COMBOS = {"A": "A", "P": "P", "A+P": "AP", "A+B": "AB", "A+P+B": "APB",
          "A+B+C": "ABC", "A+P+B+C": "APBC"}
COLS = ["entity_id", "country", "name_core", "name_alt_core", "name_phon", "name_alt_phon",
        "name_acronym", "addr_phon"]


# ---------------------------------------------------------------- data
def load(src: str) -> pd.DataFrame:
    """Normalized train source (only the columns blocking needs)."""
    return pq.read_table(norm_path("train", src), columns=COLS).to_pandas()


def make_split(s1_ids: pd.Series, frac: float = 0.2) -> np.ndarray:
    """Boolean is_val per S1 row: a seeded random 20% of the (sorted) ids; cached to parquet."""
    ids = np.sort(s1_ids.to_numpy())
    rng = np.random.default_rng(SEED)
    val = set(ids[rng.permutation(len(ids))[: int(round(frac * len(ids)))]])
    is_val = s1_ids.isin(val).to_numpy()
    ART.mkdir(exist_ok=True)
    pq.write_table(pa.table({"entity_id": s1_ids, "is_val": is_val}), ART / "split_train.parquet")
    return is_val


def sample_ids(ids: np.ndarray, n: int) -> np.ndarray:
    """Fixed random sample of n ids (seed 42, drawn from the sorted id list)."""
    ids = np.sort(ids)
    if n <= 0 or n >= len(ids):
        return ids
    return np.sort(np.random.default_rng(SEED).choice(ids, size=n, replace=False))


def gt_pairs() -> pd.DataFrame:
    """Ground truth exploded to (s1_id, cand_id) rows."""
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv").to_pandas()
    p = gt.assign(cand_id=gt["matched_entity_ids"].str.split(",")).explode("cand_id")
    p = p[p["cand_id"].notna() & (p["cand_id"] != "")]
    return p.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "cand_id"]]


def rss_gb() -> float:
    """Current resident memory of this process plus its worker processes, in GB."""
    me = psutil.Process()
    total = me.memory_info().rss
    for c in me.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.Error:
            pass
    return total / 2**30


# ---------------------------------------------------------------- channels
def docs_with_alt(df: pd.DataFrame, col: str, alt: str | None) -> tuple[list, np.ndarray]:
    """Texts to index plus the owning row of each text (alternate names add extra docs)."""
    texts, owner = df[col].tolist(), np.arange(len(df))
    if alt:
        m = (df[alt] != "").to_numpy()
        texts += df.loc[m, alt].tolist()
        owner = np.concatenate([owner, np.flatnonzero(m)])
    keep = np.array([t != "" for t in texts], dtype=bool)
    return [t for t, k in zip(texts, keep) if k], owner[keep]


def _topk_chunk(qm, tm, top_n: int):
    """Worker: single-threaded top-n cosine for one query chunk -> (row, col, score) arrays."""
    from scipy.sparse import csr_matrix
    # joblib hands large arrays over as np.memmap; the nanobind extension wants plain ndarrays
    tm = csr_matrix((np.asarray(tm.data), np.asarray(tm.indices), np.asarray(tm.indptr)),
                    shape=tm.shape, copy=False)
    r = sp_matmul_topn(qm, tm, top_n=top_n, threshold=1e-6, n_threads=1).tocoo()
    return r.row.astype(np.int32), r.col.astype(np.int32), r.data.astype(np.float32)


def tfidf_topk(q, t, col, alt, params, pool, chunk, tag, clock) -> pd.DataFrame:
    """Top-KMAX target rows per query row by TF-IDF cosine (IDF fitted on the target side).

    Runs query chunks in the process pool and prints one progress line per chunk.
    Returns q_row, t_row, score, rank (0-based; deduped when an entity has 2 docs).
    `clock` accumulates fit / query seconds and query counts for this channel.
    """
    empty = pd.DataFrame({"q_row": [], "t_row": [], "score": [], "rank": []})
    t0 = time.perf_counter()
    t_txt, t_own = docs_with_alt(t, col, alt)
    q_txt, q_own = docs_with_alt(q, col, alt)
    if not t_txt or not q_txt:
        return empty
    vec = TfidfVectorizer(dtype=np.float32, **params)
    tm = vec.fit_transform(t_txt).T.tocsr()          # features x docs
    qm = vec.transform(q_txt).tocsr()
    t_fit = time.perf_counter() - t0
    clock["fit"] += t_fit
    top_n = KMAX + (5 if alt else 0)                  # slack for duplicate owners
    starts = list(range(0, qm.shape[0], chunk))
    tasks = (delayed(_topk_chunk)(qm[s:s + chunk], tm, top_n) for s in starts)
    out, t1, done = [], time.perf_counter(), 0
    for i, (s, (rr, cc, dd)) in enumerate(zip(starts, pool(tasks)), 1):
        out.append(pd.DataFrame({"q_row": q_own[s + rr], "t_row": t_own[cc], "score": dd}))
        done += min(chunk, qm.shape[0] - s)
        el = time.perf_counter() - t1
        print(f"  [{tag}] chunk {i}/{len(starts)}  fit {t_fit:5.1f}s  query {el:6.1f}s  "
              f"{done / max(el, 1e-9):7,.0f} q/s  RSS {rss_gb():4.1f} GB", flush=True)
    clock["query"] += time.perf_counter() - t1
    clock["n_q"] += len(q)
    res = pd.concat(out, ignore_index=True)
    res = (res.sort_values(["q_row", "score"], ascending=[True, False])
              .drop_duplicates(["q_row", "t_row"]))
    res["rank"] = res.groupby("q_row").cumcount().astype(np.int16)
    return res[res["rank"] < KMAX]


def acronym_matches(q: pd.DataFrame, t: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Exact acronym links in both directions; keys shared by > cap target rows are skipped."""
    def single(df):  # single-token core usable as an acronym key: letters only, len 2-8
        s = df["name_core"]
        return s.where(s.str.fullmatch(r"[a-z]{2,8}"), "")

    links = []
    for qk, tk in ((q["name_acronym"], single(t)), (single(q), t["name_acronym"])):
        qa = pd.DataFrame({"key": qk.to_numpy(), "q_row": np.arange(len(q))})
        ta = pd.DataFrame({"key": tk.to_numpy(), "t_row": np.arange(len(t))})
        qa, ta = qa[qa["key"] != ""], ta[ta["key"] != ""]
        sizes = ta["key"].value_counts()
        ta = ta[ta["key"].map(sizes).to_numpy() <= cap]
        links.append(qa.merge(ta, on="key")[["q_row", "t_row"]])
    res = pd.concat(links).drop_duplicates()
    return res.assign(score=np.float32(1.0), rank=np.int16(0))


# ---------------------------------------------------------------- evaluation
def summarize(pair, chan, rank, is_gt, n_gt, n_q, have: set) -> list:
    """Recall (share of GT pairs covered) and avg candidates per S1 for each available combo x k."""
    rows = []
    for k in KS:
        at_k = rank < k
        for name, letters in COMBOS.items():
            if not set(letters) <= have:
                continue
            m = at_k & np.isin(chan, [CHANNELS.index(c) for c in letters])
            rows.append({"k": k, "channels": name,
                         "recall": len(pd.unique(pair[m & is_gt])) / max(n_gt, 1),
                         "cands_per_s1": len(pd.unique(pair[m])) / max(n_q, 1)})
    return rows


def print_table(rows: list, title: str) -> None:
    """One line per k with 'recall @ cands/S1' for each channel combo."""
    df = pd.DataFrame(rows)
    combos = list(dict.fromkeys(df["channels"]))
    print(f"\n{title}\n{'k':>4} " + "".join(f"{c:>18}" for c in combos))
    for k, g in df.groupby("k"):
        g = g.set_index("channels")
        print(f"{k:>4} " + "".join(f"{g.at[c, 'recall']:>10.4f} @{g.at[c, 'cands_per_s1']:>5.1f}"
                                   for c in combos))


def report(parts, q_country, gt_s1, gt_pair, n_cand, have, header) -> None:
    """Per-country and overall recall tables for the channels computed so far."""
    s1r, cand, chan, _, rank = (np.concatenate(x) for x in zip(*parts))
    pair = s1r.astype(np.int64) * n_cand + cand
    is_gt = np.isin(pair, gt_pair)
    print(f"\n=== after {header}: recall ceiling (share of sample GT pairs covered) "
          f"@ avg candidates per S1 (S2+S3; k per source per channel) ===", flush=True)
    for country in sorted(pd.unique(q_country)):
        in_c = q_country[s1r] == country
        print_table(summarize(pair[in_c], chan[in_c], rank[in_c], is_gt[in_c],
                              int((q_country[gt_s1] == country).sum()),
                              int((q_country == country).sum()), have),
                    f"[{country}]  S1 = {(q_country == country).sum():,}")
    print_table(summarize(pair, chan, rank, is_gt, len(gt_pair), len(q_country), have),
                f"[overall]  S1 = {len(q_country):,}")
    sys.stdout.flush()


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=50_000, help="val S1 sample size (0 = all val)")
    ap.add_argument("--jobs", type=int, default=10, help="worker processes for top-k")
    ap.add_argument("--chunk", type=int, default=2000, help="queries per worker task")
    ap.add_argument("--acr-cap", type=int, default=10, help="max target rows per acronym key")
    args = ap.parse_args()
    utf8_stdout()
    t_start = time.perf_counter()

    s1 = load("S1")
    is_val = make_split(s1["entity_id"])
    keep = set(sample_ids(s1.loc[is_val, "entity_id"].to_numpy(), args.sample))
    q_all = s1[s1["entity_id"].isin(keep)].reset_index(drop=True)
    del s1
    targets = {src: load(src) for src in TARGETS}
    offset = {"S2": 0, "S3": len(targets["S2"])}             # cand code = offset + row
    cand_ids = pa.concat_arrays([pa.array(targets[s]["entity_id"].to_numpy()) for s in TARGETS])
    cand_country = np.concatenate([targets[s]["country"].to_numpy() for s in TARGETS])
    n_cand = len(cand_ids)

    # ground truth restricted to the sample, as integer (s1_row, cand_code) pairs
    gt = gt_pairs()
    s1_row = pd.Index(q_all["entity_id"]).get_indexer(gt["s1_id"])
    gt = gt[s1_row >= 0].assign(s1_row=s1_row[s1_row >= 0])
    gt["cand"] = pd.Index(cand_ids.to_pandas()).get_indexer(gt["cand_id"])
    assert (gt["cand"] >= 0).all(), "GT candidate id missing from train S2/S3"
    gt_pair = gt["s1_row"].to_numpy(np.int64) * n_cand + gt["cand"].to_numpy(np.int64)
    gt_s1 = gt["s1_row"].to_numpy()
    q_country = q_all["country"].to_numpy()
    xc = (q_country[gt_s1] != cand_country[gt["cand"]]).mean()
    print(f"sample: {len(q_all):,} val S1 of {int(is_val.sum()):,} "
          f"({1 - gt['s1_row'].nunique() / len(q_all):.1%} singletons) | GT pairs {len(gt):,} "
          f"| GT pairs crossing countries {xc:.3%} | targets S2 {offset['S3']:,} S3 {n_cand - offset['S3']:,}",
          flush=True)

    parts, have = [], set()
    clocks = {ch: {"fit": 0.0, "query": 0.0, "n_q": 0} for ch in CHANNELS}
    with Parallel(n_jobs=args.jobs, max_nbytes="1M", mmap_mode="c", return_as="generator") as pool:
        for header, chans in STAGES:
            for ch in chans:
                for country in sorted(pd.unique(q_country)):
                    q_idx = np.flatnonzero(q_country == country)
                    q = q_all.iloc[q_idx].reset_index(drop=True)
                    for src in TARGETS:
                        t_idx = np.flatnonzero(targets[src]["country"].to_numpy() == country)
                        t = targets[src].iloc[t_idx].reset_index(drop=True)
                        tag = f"{ch} {country}/{src} q={len(q):,} t={len(t):,}"
                        if ch == "C":
                            t0 = time.perf_counter()
                            r = acronym_matches(q, t, args.acr_cap)
                            # hash join over all targets: fixed cost, not per query
                            clocks["C"]["fit"] += time.perf_counter() - t0
                            print(f"  [{tag}] {len(r):,} links in {time.perf_counter() - t0:.1f}s", flush=True)
                        else:
                            col, alt, params = TFIDF_CHANNELS[ch]
                            r = tfidf_topk(q, t, col, alt, params, pool, args.chunk, tag, clocks[ch])
                        parts.append((q_idx[r["q_row"].to_numpy(np.int64)],
                                      offset[src] + t_idx[r["t_row"].to_numpy(np.int64)],
                                      np.full(len(r), CHANNELS.index(ch), np.int8),
                                      r["score"].to_numpy(np.float32), r["rank"].to_numpy(np.int16)))
                c = clocks[ch]
                print(f"channel {ch}: fit {c['fit']:.0f}s + query {c['query']:.0f}s "
                      f"({c['n_q'] / max(c['query'], 1e-9):,.0f} q/s)", flush=True)
                have.add(ch)
            report(parts, q_country, gt_s1, gt_pair, n_cand, have, header)
    del targets

    s1r, cand, chan, score, rank = (np.concatenate(x) for x in zip(*parts))
    pair = s1r.astype(np.int64) * n_cand + cand
    is_gt = np.isin(pair, gt_pair)
    pq.write_table(pa.table({
        "s1_id": pa.array(q_all["entity_id"].to_numpy()).take(pa.array(s1r)),
        "cand_id": cand_ids.take(pa.array(cand)),
        "channel": pa.array(np.array(CHANNELS)[chan]).dictionary_encode(),
        "score": score, "rank": rank}), ART / f"val_blocking_sample_k{KMAX}.parquet",
        compression="zstd")

    # exclusive contribution of each channel at k=20: GT pairs no other channel found
    hp = pd.DataFrame({"pair": pair[is_gt], "bit": (1 << chan[is_gt].astype(np.int64))})
    bits = hp.drop_duplicates().groupby("pair")["bit"].sum()
    print(f"\nGT pairs by exclusive channel at k={KMAX} (share of sample GT pairs):")
    for i, ch in enumerate(CHANNELS):
        print(f"  only {ch}: {(bits == (1 << i)).sum() / len(gt):.4%}")
    print(f"  missed by all channels: {1 - len(bits) / len(gt):.4%}")

    print(f"\nTiming per channel (sample of {len(q_all):,} S1; {args.jobs} worker processes) and "
          f"extrapolated full TEST run ({TEST_S1:,} S1 queries, fit cost assumed unchanged):")
    total_test = 0.0
    for ch in CHANNELS:
        c = clocks[ch]
        rate = c["n_q"] / max(c["query"], 1e-9)
        est = c["fit"] + c["query"] * TEST_S1 / len(q_all)  # query time scales with #S1
        total_test += est
        print(f"  {ch}: fit {c['fit']:6.0f}s  query {c['query']:6.0f}s  {rate:8,.0f} q/s "
              f"(both sources)  -> test ~{est / 60:6.1f} min")
    print(f"  all channels -> test ~{total_test / 60:.0f} min")
    print(f"\nruntime {time.perf_counter() - t_start:.0f}s total; peak RAM (main process) "
          f"{peak_ram_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
