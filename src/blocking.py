"""Stage 2 — candidate generation (blocking) by rare shared tokens.

For every country (open set, taken from the S1 query rows) and every target source (S2, S3):
  * tokens = words of addr_norm (numbers and postal-like tokens included) and words of
    name_core, kept in separate namespaces;
  * document frequency (df) is counted on the target side; tokens with df > cap are dropped
    (caps 200 / 500 / 2000 are compared);
  * candidate score = sum over shared kept tokens of IDF * weight (address 1.0, name 0.5),
    computed as ONE sparse matmul: weighted query matrix x binary target matrix, top-N per
    query row via sparse_dot_topn. Capped postings keep the product cheap;
  * a pool of the POOL best targets per source is optionally reranked with a pairwise name
    similarity (rapidfuzz token_set_ratio on name_core, computed on pairs only), and the top
    K per source (K = 5 / 10 / 15) is kept.
S1 rows with a blank address simply have no address tokens (name tokens only).

Evaluation: the fixed 50,000-row sample of validation S1s (seed 42) searched against ALL
train S2/S3 rows of their country. Reports recall / avg candidates per S1 per cap x K x
rerank weight, runtime, peak RAM and an extrapolated full-test runtime.

Outputs:
  artifacts/split_train.parquet                       entity_id, is_val
  artifacts/val_blocking_pool_cap{cap}.parquet        best cap's pool: s1_id, cand_id,
                                                      tok_score, name_sim, tok_rank

Usage: python -u src/blocking.py [--sample 50000] [--caps 200 500 2000]
"""
import argparse
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import ART, DATA, SEED, norm_path, peak_ram_gb, read_tsv, utf8_stdout

TARGETS = ("S2", "S3")
KS = (5, 10, 15)
POOL = 30                      # targets per source kept before reranking
KB = 2                         # quota per source from the blank-address (name-only) index
# frozen production config (set from the 50k validation study; see EXPERIMENTS.md)
FROZEN = dict(cap=200, alpha=6.0, k=10)   # val50k: recall 0.9293 @ 23.7 cands/S1
W_ADDR, W_NAME = 1.0, 0.5      # token weights
ALPHAS = (0.0, 3.0, 6.0)       # rerank: score + alpha * name_sim/100
TEST_S1 = 1_732_544            # rows in dataset/test/test_source1.tsv
TARGET_RECALL = 0.93
COLS = ["entity_id", "country", "name_core", "name_phon", "addr_norm", "addr_phon"]
ADDR_COL, NAME_COL = "addr_norm", "name_core"   # token source columns (overridable via CLI)


# ---------------------------------------------------------------- data
def load(src: str, split: str = "train") -> pd.DataFrame:
    """Normalized source (only the columns blocking needs)."""
    return pq.read_table(norm_path(split, src), columns=COLS).to_pandas()


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


# ---------------------------------------------------------------- tokens
N_FEATURES = 2 ** 26           # hashed token space (pairs make the vocabulary too big for a dict)
PAIR_BASE_DF = 0.01            # words in > 1% of targets are too common to seed pair tokens
RUN_RE = re.compile(r"[a-z]+|\d+")
AFFIX_RE = re.compile(r"^(?:www)|(?:com|net|org|biz|info|co|in|us|fr)$")


def addr_tokens(s: str, common: frozenset) -> list:
    """Address tokens: words; digit-prefix + last-digit-truncated number variants
    (house numbers are perturbed: 1125 ~ 1127, 423 ~ 423c); unordered pairs of the
    non-common words/number variants (robust to reordered address parts)."""
    # split letter/digit runs ("no8" -> no 8, "423c" -> 423 c) and drop leading zeros
    words = [p.lstrip("0") or "0" if p[0].isdigit() else p for p in RUN_RE.findall(s)]
    toks = set(words)
    base = set()
    for w in words:
        if w[0].isdigit():
            toks.add(w)
            base.add(w)
            if len(w) >= 3:
                toks.add(w[:-1] + "#")
                base.add(w[:-1] + "#")
        elif w not in common:
            base.add(w)
    base = sorted(base)[:10]
    toks.update(f"{a}|{b}" for i, a in enumerate(base) for b in base[i + 1:])
    return ["a:" + t for t in toks]


def name_tokens(s: str, common: frozenset) -> list:
    """Name tokens: words; unordered pairs of non-common words; the concatenated name
    ('salinas bright arrived' ~ 'salinasbrightarrivedcom') with web affixes stripped."""
    words = s.split()
    toks = set(words)
    base = sorted({w for w in words if w not in common})[:8]
    toks.update(f"{a}|{b}" for i, a in enumerate(base) for b in base[i + 1:])
    if words:
        cat = "".join(words)
        toks.add("=" + cat)
        stripped = AFFIX_RE.sub("", cat)
        if stripped and stripped != cat:
            toks.add("=" + stripped)
    return ["n:" + t for t in toks]


def _hash_chunk(texts: list, kind: str, common: frozenset) -> sp.csr_matrix:
    """Worker: binary hashed token matrix for a chunk of texts."""
    fn = addr_tokens if kind == "addr" else name_tokens
    hv = HashingVectorizer(analyzer=lambda s: fn(s, common), n_features=N_FEATURES,
                           alternate_sign=False, norm=None, binary=True, dtype=np.float32)
    return hv.transform(texts)


def hash_texts(texts: list, kind: str, common: frozenset, pool, chunk: int = 200_000) -> sp.csr_matrix:
    """Binary hashed token matrix for all texts, built in parallel over chunks."""
    parts = pool(delayed(_hash_chunk)(texts[i:i + chunk], kind, common)
                 for i in range(0, len(texts), chunk))
    return sp.vstack(list(parts), format="csr")


def common_words(texts: pd.Series) -> frozenset:
    """Words (letter/digit runs) whose document frequency exceeds PAIR_BASE_DF of the texts."""
    df = Counter(w for s in texts for w in set(RUN_RE.findall(s)))
    lim = PAIR_BASE_DF * len(texts)
    return frozenset(w for w, c in df.items() if c > lim)


# ---------------------------------------------------------------- token index
class TokenIndex:
    """Hashed binary token matrix for one (country, source) target set.

    Address and name tokens share one hashed column space (namespaced by prefix).
    `for_cap(cap)` gives per-column IDF with columns of df > cap zeroed; the query matrix
    is weighted by IDF x (1.0 address / 0.5 name), so dropped tokens never touch postings.
    """

    def __init__(self, t: pd.DataFrame, pool):
        self.n = len(t)
        self.common_a = common_words(t[ADDR_COL])
        self.common_n = common_words(t[NAME_COL])
        x = (hash_texts(t[ADDR_COL].tolist(), "addr", self.common_a, pool)
             + hash_texts(t[NAME_COL].tolist(), "name", self.common_n, pool))
        x.data[:] = 1.0                                        # binary even on hash collisions
        self.df = np.bincount(x.indices, minlength=N_FEATURES)
        self.tt = x.T.tocsr()                                  # tokens x docs
        del x

    def queries(self, q: pd.DataFrame, pool) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Binary hashed (address, name) query matrices using the target-side common words."""
        return (hash_texts(q[ADDR_COL].tolist(), "addr", self.common_a, pool),
                hash_texts(q[NAME_COL].tolist(), "name", self.common_n, pool))

    def for_cap(self, cap: int) -> np.ndarray:
        """Per-column IDF, zero for tokens absent from targets or with df > cap."""
        keep = (self.df > 0) & (self.df <= cap)
        idf = np.log(self.n / np.maximum(self.df, 1)).astype(np.float32)
        return np.where(keep, idf, 0).astype(np.float32)


def weight_queries(qa: sp.csr_matrix, qn: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """Query matrix with entries IDF x namespace weight; dropped tokens removed."""
    qa, qn = qa.copy(), qn.copy()
    qa.data = idf[qa.indices] * W_ADDR
    qn.data = idf[qn.indices] * W_NAME
    qw = (qa + qn).tocsr()
    qw.eliminate_zeros()
    return qw


def search(qw: sp.csr_matrix, tt: sp.csr_matrix, top_n: int, chunk: int = 50_000) -> pd.DataFrame:
    """Top-n targets per query by summed IDF of shared kept tokens -> q_row, t_row, score, rank."""
    out = []
    for i in range(0, qw.shape[0], chunk):
        r = sp_matmul_topn(qw[i:i + chunk], tt, top_n=top_n, threshold=1e-6,
                           sort=True, n_threads=8).tocoo()
        out.append(pd.DataFrame({"q_row": (r.row + i).astype(np.int32),
                                 "t_row": r.col.astype(np.int32),
                                 "score": r.data.astype(np.float32)}))
    res = pd.concat(out, ignore_index=True)
    res = res.sort_values(["q_row", "score"], ascending=[True, False], kind="stable")
    res["rank"] = res.groupby("q_row").cumcount().astype(np.int16)
    return res


def name_sim(q: pd.DataFrame, t: pd.DataFrame, qi: np.ndarray, ti: np.ndarray) -> np.ndarray:
    """Pairwise name similarity for rerank: max token_set_ratio over name_core and name_phon
    (the phonetic form rescues transliterated names). Computed on pairs only."""
    sims = [cpdist(q[c].to_numpy()[qi], t[c].to_numpy()[ti], scorer=fuzz.token_set_ratio, workers=-1)
            for c in ("name_core", "name_phon")]
    return np.maximum(*sims).astype(np.float32)


# ---------------------------------------------------------------- evaluation
def topk_by(res: pd.DataFrame, key: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of rows among the top-k per (q_row, src) when ordered by `key` desc."""
    order = np.lexsort((-key, res["src"].to_numpy(), res["q_row"].to_numpy()))
    grp = res["q_row"].to_numpy()[order].astype(np.int64) * 4 + res["src"].to_numpy()[order]
    start = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
    pos = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    # groups src 0/1 = main S2/S3 index (quota k); src 2/3 = blank-address S2/S3 (quota KB)
    quota = np.where(res["src"].to_numpy()[order] >= 2, min(KB, k), k)
    mask = np.zeros(len(res), dtype=bool)
    mask[order[pos < quota]] = True
    return mask


def evaluate(res: pd.DataFrame, is_gt: np.ndarray, q_country: np.ndarray, n_gt_by: dict,
             cap: int) -> list:
    """Recall and avg candidates per S1, per country and overall, for each K x alpha."""
    rows = []
    for alpha in ALPHAS:
        key = res["score"].to_numpy() + alpha * res["name_sim"].to_numpy() / 100.0
        for k in KS:
            m = topk_by(res, key, k)
            c = q_country[res["q_row"].to_numpy()]
            for country in list(n_gt_by) :
                sel = m if country == "overall" else m & (c == country)
                n_q = len(q_country) if country == "overall" else int((q_country == country).sum())
                rows.append({"cap": cap, "alpha": alpha, "K": k, "country": country,
                             "recall": (sel & is_gt).sum() / max(n_gt_by[country], 1),
                             "cands_per_s1": sel.sum() / n_q})
    return rows


# ---------------------------------------------------------------- pool construction
def build_pools(q_all: pd.DataFrame, split: str, caps: list, jpool, clock: dict,
                select: dict | None = None) -> dict:
    """Candidate pools per cap for all query rows, searching `split` S2/S3 per country.

    Each (country, source) target set is split into a main index (rows with an address)
    and a small name-only index (blank-address rows) whose candidates form separate groups
    (src 2/3) with their own quota. If `select` (alpha, k) is given, each group is cut to
    its final top-k right away (keeps memory bounded for the full test run).
    Returns {cap: DataFrame(q_row, src, cand_id, score, name_sim, tok_rank)}.
    """
    q_country = q_all["country"].to_numpy()
    pools = {cap: [] for cap in caps}
    for src_code, src in enumerate(TARGETS):
        tgt = load(src, split)
        for country in sorted(pd.unique(q_country)):
            q_idx = np.flatnonzero(q_country == country)
            q = q_all.iloc[q_idx].reset_index(drop=True)
            tc = tgt[tgt["country"].to_numpy() == country]
            blank = (tc["addr_norm"] == "").to_numpy()
            for sub, t in ((0, tc[~blank].reset_index(drop=True)), (2, tc[blank].reset_index(drop=True))):
                if len(t) == 0:
                    continue
                t0 = time.perf_counter()
                idx = TokenIndex(t, jpool)
                clock["fixed"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                qa, qn = idx.queries(q, jpool)
                t_qx = time.perf_counter() - t0
                for cap in caps:
                    t0 = time.perf_counter()
                    r = search(weight_queries(qa, qn, idx.for_cap(cap)), idx.tt,
                               POOL if sub == 0 else KB * 3)
                    r["name_sim"] = name_sim(q, t, r["q_row"].to_numpy(), r["t_row"].to_numpy())
                    part = pd.DataFrame({
                        "q_row": q_idx[r["q_row"].to_numpy()], "src": np.int8(src_code + sub),
                        "cand_id": t["entity_id"].to_numpy()[r["t_row"].to_numpy()],
                        "score": r["score"].to_numpy(), "name_sim": r["name_sim"].to_numpy(),
                        "tok_rank": r["rank"].to_numpy()})
                    if select:
                        key = part["score"].to_numpy() + select["alpha"] * part["name_sim"].to_numpy() / 100
                        part = part[topk_by(part, key, select["k"])]
                    pools[cap].append(part)
                    dt = time.perf_counter() - t0 + t_qx
                    clock.setdefault(cap, 0.0)
                    clock[cap] += dt
                    print(f"  {country}/{src}{'/blank' if sub else ''} cap={cap:<5} q={len(q):,} t={len(t):,} "
                          f"query+sim {dt:6.1f}s ({len(q) / dt:,.0f} q/s)  pairs {len(part):,}  "
                          f"fixed so far {clock['fixed']:.0f}s  peak {peak_ram_gb():.1f} GB", flush=True)
                del idx, qa, qn
        del tgt
    return {cap: pd.concat(p, ignore_index=True) for cap, p in pools.items()}


def select_from_pool(pool_path: str) -> None:
    """Build artifacts/cand_val50k.parquet from a saved eval pool (no recomputation).

    The eval pool holds the top-POOL per group with token score and name_sim; the source
    comes from the id prefix and the blank-address group from the target's addr_norm.
    Only valid when the pool was produced with FROZEN['cap'] on the same 50k sample.
    """
    t_start = time.perf_counter()
    p = pq.read_table(pool_path).to_pandas()
    cfg = FROZEN
    assert f"cap{cfg['cap']}" in str(pool_path), f"pool {pool_path} is not cap {cfg['cap']}"
    blank_ids = set()
    for src in TARGETS:
        t = pq.read_table(norm_path("train", src), columns=["entity_id", "addr_norm"]).to_pandas()
        blank_ids |= set(t.loc[t["addr_norm"] == "", "entity_id"])
    src_code = (p["cand_id"].str.startswith("S3").to_numpy().astype(np.int8)
                + 2 * p["cand_id"].isin(blank_ids).to_numpy().astype(np.int8))
    q_row, uniq = pd.factorize(p["s1_id"])
    res = pd.DataFrame({"q_row": q_row, "src": src_code, "cand_id": p["cand_id"].to_numpy(),
                        "score": p["tok_score"].to_numpy(), "name_sim": p["name_sim"].to_numpy()})
    key = res["score"].to_numpy() + cfg["alpha"] * res["name_sim"].to_numpy() / 100
    res = res[topk_by(res, key, cfg["k"])]
    write_candidates(res, np.asarray(uniq), cfg, ART / "cand_val50k.parquet")
    print(f"(from pool {pool_path}; S1 with >=1 pool row: {len(uniq):,}) runtime "
          f"{time.perf_counter() - t_start:.0f}s", flush=True)


def write_candidates(res: pd.DataFrame, s1_ids: np.ndarray, cfg: dict, out) -> None:
    """Final ranking per S1 by score + alpha*name_sim and write the candidate parquet."""
    key = res["score"].to_numpy() + cfg["alpha"] * res["name_sim"].to_numpy() / 100
    res = res.assign(key=key).sort_values(["q_row", "key"], ascending=[True, False])
    res["final_rank"] = res.groupby("q_row").cumcount().astype(np.int16)
    pq.write_table(pa.table({
        "s1_id": s1_ids[res["q_row"].to_numpy()],
        "cand_id": res["cand_id"].to_numpy(),
        "source": np.array(["S2", "S3", "S2", "S3"])[res["src"].to_numpy()],
        "blank_addr_index": res["src"].to_numpy() >= 2,
        "tok_score": res["score"].to_numpy(), "name_sim": res["name_sim"].to_numpy(),
        "final_rank": res["final_rank"].to_numpy()}), out, compression="zstd")
    print(f"wrote {out}: {len(res):,} pairs over {res['q_row'].nunique():,} S1 "
          f"({len(res) / max(len(s1_ids), 1):.2f} cands per listed S1)", flush=True)


def produce(name: str, jobs: int) -> None:
    """Write artifacts/cand_{name}.parquet with the FROZEN config.

    name: 'test' (all test S1), 'val50k' (the 50k val sample), 'fit200k' (200k non-val
    train S1s, seed 42). Schema: s1_id, cand_id, source, tok_score, name_sim, final_rank.
    """
    t_start, clock = time.perf_counter(), {"fixed": 0.0}
    if name == "test":
        split, q_all = "test", load("S1", "test")
    else:
        split, s1 = "train", load("S1", "train")
        is_val = make_split(s1["entity_id"])
        pick = (sample_ids(s1.loc[is_val, "entity_id"].to_numpy(), 50_000) if name == "val50k"
                else sample_ids(s1.loc[~is_val, "entity_id"].to_numpy(), 200_000))
        q_all = s1[s1["entity_id"].isin(set(pick))].reset_index(drop=True)
        del s1
    cfg = FROZEN
    print(f"produce {name}: {len(q_all):,} S1 queries vs {split} S2/S3, config {cfg}, KB={KB}", flush=True)
    with Parallel(n_jobs=jobs) as jpool:
        res = build_pools(q_all, split, [cfg["cap"]], jpool, clock, select=cfg)[cfg["cap"]]
    out = ART / f"cand_{name}.parquet"
    write_candidates(res, q_all["entity_id"].to_numpy(), cfg, out)
    n_q = len(q_all)
    print(f"wrote {out}: {len(res):,} pairs, {len(res) / n_q:.2f} cands/S1, "
          f"S1 with 0 cands {1 - res['q_row'].nunique() / n_q:.2%}; runtime "
          f"{time.perf_counter() - t_start:.0f}s (fixed {clock['fixed']:.0f}s); peak RAM {peak_ram_gb():.1f} GB",
          flush=True)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=50_000, help="val S1 sample size (0 = all val)")
    ap.add_argument("--caps", type=int, nargs="+", default=[200, 500, 2000])
    ap.add_argument("--jobs", type=int, default=8, help="processes for token hashing")
    ap.add_argument("--produce", choices=["test", "val50k", "fit200k"],
                    help="write artifacts/cand_<name>.parquet with the frozen config and exit")
    ap.add_argument("--from-pool", help="with --produce val50k: derive from a saved eval pool parquet")
    args = ap.parse_args()
    utf8_stdout()
    if args.produce == "val50k" and args.from_pool:
        select_from_pool(args.from_pool)
        return
    if args.produce:
        produce(args.produce, args.jobs)
        return
    t_start = time.perf_counter()

    s1 = load("S1")
    is_val = make_split(s1["entity_id"])
    keep_ids = set(sample_ids(s1.loc[is_val, "entity_id"].to_numpy(), args.sample))
    q_all = s1[s1["entity_id"].isin(keep_ids)].reset_index(drop=True)
    del s1
    q_country = q_all["country"].to_numpy()
    gt = gt_pairs()
    gt = gt[gt["s1_id"].isin(keep_ids)]
    gt_set = set(zip(gt["s1_id"], gt["cand_id"]))
    gt_country = gt["s1_id"].map(q_all.set_index("entity_id")["country"])
    n_gt_by = {c: int((gt_country == c).sum()) for c in sorted(pd.unique(q_country))}
    n_gt_by["overall"] = len(gt)
    print(f"sample {len(q_all):,} val S1, GT pairs {len(gt):,}", flush=True)

    clock = {"fixed": 0.0}
    with Parallel(n_jobs=args.jobs) as jpool:
        pools = build_pools(q_all, "train", args.caps, jpool, clock)
    t_fixed = clock["fixed"]
    t_query = {cap: clock[cap] for cap in args.caps}

    rows = []
    for cap in args.caps:
        res = pools[cap]
        is_gt = np.fromiter((p in gt_set for p in zip(q_all["entity_id"].to_numpy()[res["q_row"]],
                                                      res["cand_id"])), bool, len(res))
        res["is_gt"] = is_gt
        rows += evaluate(res, is_gt, q_country, n_gt_by, cap)
    df = pd.DataFrame(rows)

    print("\nRecall @ avg candidates per S1 (S2+S3; K per source). alpha = rerank weight of "
          "name token_set_ratio (0 = token score only)")
    for country in list(n_gt_by):
        print(f"\n[{country}]")
        print(f"{'cap':>6}{'alpha':>7}" + "".join(f"{'K=' + str(k):>18}" for k in KS))
        for (cap, alpha), g in df[df["country"] == country].groupby(["cap", "alpha"]):
            g = g.set_index("K")
            print(f"{cap:>6}{alpha:>7.1f}" + "".join(
                f"{g.at[k, 'recall']:>10.4f} @{g.at[k, 'cands_per_s1']:>5.1f}" for k in KS))

    print(f"\npool recall (all {POOL} per source, i.e. max achievable by reranking):")
    for cap in args.caps:
        res = pools[cap]
        print(f"  cap {cap}: {res['is_gt'].sum() / len(gt):.4f} @ {len(res) / len(q_all):.1f} cands/S1")

    scale = TEST_S1 / len(q_all)
    print(f"\nTiming: target-side fixed {t_fixed:.0f}s (all caps); per cap query+sim on sample and "
          f"extrapolated full TEST ({TEST_S1:,} S1; fixed cost counted once per cap run):")
    fixed_per_cap = t_fixed / len(args.caps)
    for cap in args.caps:
        print(f"  cap {cap:<5}: sample {t_query[cap]:6.0f}s -> test ~{(fixed_per_cap + t_query[cap] * scale) / 60:5.1f} min")

    ok = df[(df["country"] == "overall") & (df["recall"] >= TARGET_RECALL)]
    if len(ok):
        best = ok.sort_values("cands_per_s1").iloc[0]
        print(f"\nsmallest config with recall >= {TARGET_RECALL}: cap {best['cap']}, alpha {best['alpha']}, "
              f"K {best['K']} -> recall {best['recall']:.4f} @ {best['cands_per_s1']:.1f} cands/S1")
        cap = int(best["cap"])
    else:
        cap = int(df[df["country"] == "overall"].sort_values("recall").iloc[-1]["cap"])
        print(f"\nno config reaches recall {TARGET_RECALL}; saving pool of cap {cap}")
    res = pools[cap]
    pq.write_table(pa.table({
        "s1_id": q_all["entity_id"].to_numpy()[res["q_row"]], "cand_id": res["cand_id"].to_numpy(),
        "tok_score": res["score"].to_numpy(), "name_sim": res["name_sim"].to_numpy(),
        "tok_rank": res["tok_rank"].to_numpy()}), ART / f"val_blocking_pool_cap{cap}.parquet",
        compression="zstd")
    print(f"\nruntime {time.perf_counter() - t_start:.0f}s total; peak RAM {peak_ram_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
