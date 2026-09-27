"""Exploratory data analysis for the business entity resolution task.

Loads all train/test source TSVs and the train ground truth with pyarrow
(tab-separated, no quoting, all columns as strings) and prints a concise report:

  1. row counts per source and country
  2. null / blank rates per column
  3. share of S1 singletons in the ground truth
  4. distribution of matches per S1 entity, split S2 vs S3
  5. whether any S2/S3 id is matched to more than one S1 entity
  6. share of non-Latin script in names/addresses per source and country
  7. 15 random matched pairs (train) and 10 France rows (test)

Country is treated as an open set: nothing below assumes {US, India}.

Usage: python src/eda.py [--data-dir dataset] [--seed 42]
"""
import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

TEXT_COLS = ["business_name", "business_address"]
# Scripts to break non-Latin text down by; anything else is reported as "other".
SCRIPTS = ["Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu",
           "Kannada", "Malayalam", "Arabic", "Cyrillic", "Greek", "Han", "Hebrew", "Thai"]
# RE2 class: a letter (\p{L}) that is not Latin -> "not (non-letter or Latin)".
NON_LATIN_RE = r"[^\P{L}\p{Latin}]"


def read_tsv(path: Path) -> pa.Table:
    """Read a TSV as an Arrow table with every column typed as string.

    Quotes are treated literally (quote_char=False) and empty fields stay ''.
    """
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
    return pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in header},
                                             strings_can_be_null=False),
    )


def blank_mask(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """True where a value is null or empty after stripping whitespace."""
    return pc.or_(pc.is_null(col), pc.equal(pc.utf8_trim_whitespace(col), ""))


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def trunc(s, n: int = 55) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def load_all(data_dir: Path) -> tuple[dict, pa.Table]:
    """Load the six source tables (keyed 'train/S1' etc.) and the ground truth."""
    tables = {}
    for split in ("train", "test"):
        for s in (1, 2, 3):
            p = data_dir / split / f"{split}_source{s}.tsv"
            tables[f"{split}/S{s}"] = read_tsv(p)
            print(f"loaded {p} ({tables[f'{split}/S{s}'].num_rows:,} rows, "
                  f"{tables[f'{split}/S{s}'].nbytes / 2**20:,.0f} MiB in Arrow)", file=sys.stderr)
    gt = read_tsv(data_dir / "train" / "train_ground_truth.tsv")
    return tables, gt


def report_counts(tables: dict) -> None:
    """Rows per source x country, plus countries that appear only in test."""
    section("1. Row counts per source and country")
    countries = {k: dict(zip(*[v.to_pylist() for v in
                              pc.value_counts(t["country"]).flatten()]))
                 for k, t in tables.items()}
    all_c = sorted({c for d in countries.values() for c in d},
                   key=lambda c: -sum(d.get(c, 0) for d in countries.values()))
    print(f"{'source':<10}{'total':>11}" + "".join(f"{trunc(c, 10) or '<blank>':>12}" for c in all_c))
    for k, d in countries.items():
        print(f"{k:<10}{sum(d.values()):>11,}" + "".join(f"{d.get(c, 0):>12,}" for c in all_c))
    train_c = {c for k, d in countries.items() if k.startswith("train") for c in d}
    test_c = {c for k, d in countries.items() if k.startswith("test") for c in d}
    print(f"countries only in test: {sorted(test_c - train_c) or 'none'}; "
          f"only in train: {sorted(train_c - test_c) or 'none'}")


def report_nulls(tables: dict) -> None:
    """Null/blank rate per column per source; also duplicate-id check."""
    section("2. Null / blank rates (null or whitespace-only)")
    cols = ["entity_id", "business_name", "business_address", "country"]
    print(f"{'source':<10}" + "".join(f"{c:>18}" for c in cols) + f"{'dup ids':>10}")
    for k, t in tables.items():
        rates = [pc.mean(pc.cast(blank_mask(t[c]), pa.int8())).as_py() for c in cols]
        dups = t.num_rows - pc.count_distinct(t["entity_id"]).as_py()
        print(f"{k:<10}" + "".join(f"{pct(r):>18}" for r in rates) + f"{dups:>10,}")


def explode_gt(gt: pa.Table) -> pa.Table:
    """Ground truth as (s1_id, match_id) pairs; singletons produce no rows."""
    lists = pc.split_pattern(pc.utf8_trim_whitespace(gt["matched_entity_ids"]), ",")
    s1 = pc.take(gt["source1_entity_id"], pc.list_parent_indices(lists))
    m = pc.utf8_trim_whitespace(pc.list_flatten(lists))
    pairs = pa.table({"s1": s1, "match": m})
    pairs = pairs.filter(pc.not_equal(pairs["match"], ""))
    return pairs


def report_gt(gt: pa.Table, tables: dict) -> pa.Table:
    """Singleton share, per-S1 match-count distribution split by S2/S3, 1:many checks."""
    section("3-5. Ground truth")
    pairs = explode_gt(gt)
    n_gt = gt.num_rows
    n_s1 = tables["train/S1"].num_rows
    covered = pc.sum(pc.is_in(tables["train/S1"]["entity_id"], gt["source1_entity_id"])).as_py()
    print(f"GT rows: {n_gt:,} | train S1 rows: {n_s1:,} | S1 ids present in GT: {covered:,} "
          f"| total match pairs: {pairs.num_rows:,}")

    # per-S1 counts of S2 and S3 matches (0 for singletons) via a group-by on pairs
    src = pc.utf8_slice_codeunits(pairs["match"], 0, 2)
    pairs = pairs.append_column("src", src)
    g = pairs.group_by(["s1", "src"]).aggregate([("match", "count")])
    import pandas as pd  # small (one row per S1 x source) -> pandas is fine here
    per = (g.to_pandas().pivot_table(index="s1", columns="src", values="match_count", fill_value=0)
           .reindex(gt["source1_entity_id"].to_pandas(), fill_value=0))
    for c in ("S2", "S3"):
        if c not in per:
            per[c] = 0
    per = per[["S2", "S3"]].astype(int)
    per["total"] = per["S2"] + per["S3"]

    single = (per["total"] == 0).mean()
    print(f"\n(3) singleton S1 (no matches): {pct(single)}  ({(per['total'] == 0).sum():,} of {len(per):,})")
    unknown = sorted(set(pc.unique(src).to_pylist()) - {"S2", "S3"})
    if unknown:
        print(f"    WARNING: match ids with unexpected prefixes: {unknown}")

    print("\n(4) matches per S1 entity (share of S1 rows)")
    bins = [0, 1, 2, 3, 4, 5, 10]
    labels = ["0", "1", "2", "3", "4", "5-9", "10+"]
    print(f"{'':<8}" + "".join(f"{l:>8}" for l in labels) + f"{'mean':>8}{'max':>6}")
    for c in ("S2", "S3", "total"):
        b = pd.cut(per[c], bins=bins + [10**9], right=False, labels=labels)
        dist = b.value_counts(normalize=True).reindex(labels)
        print(f"{c:<8}" + "".join(f"{pct(v):>8}" for v in dist) + f"{per[c].mean():>8.2f}{per[c].max():>6}")
    nz = per[per["total"] > 0]
    print(f"among non-singletons: has S2 {pct((nz['S2'] > 0).mean())}, has S3 {pct((nz['S3'] > 0).mean())}, "
          f"both {pct(((nz['S2'] > 0) & (nz['S3'] > 0)).mean())}, mean total {nz['total'].mean():.2f}")

    # (5) is the mapping S2/S3 -> S1 functional?
    print("\n(5) S2/S3 ids linked to more than one S1")
    mc = pairs.group_by("match").aggregate([("s1", "count_distinct")])
    multi = mc.filter(pc.greater(mc["s1_count_distinct"], 1))
    dup_pairs = pairs.num_rows - pairs.group_by(["s1", "match"]).aggregate([]).num_rows
    print(f"    distinct matched ids: {mc.num_rows:,} | linked to >1 S1: {multi.num_rows:,} "
          f"| duplicate (S1, match) pairs: {dup_pairs:,}")
    if multi.num_rows:
        ex = multi.slice(0, 3).to_pylist()
        print(f"    examples: {ex}")
    for s in ("S2", "S3"):
        ids = tables[f"train/{s}"]["entity_id"]
        linked = pc.sum(pc.is_in(ids, mc["match"])).as_py()
        print(f"    train {s}: {linked:,}/{len(ids):,} rows ({pct(linked / len(ids))}) are matched to some S1")
    orphan = pc.sum(pc.invert(pc.is_in(
        mc["match"], pa.chunked_array(tables["train/S2"]["entity_id"].chunks
                                      + tables["train/S3"]["entity_id"].chunks)))).as_py()
    print(f"    GT match ids missing from train S2/S3: {orphan:,}")
    return pairs


def report_scripts(tables: dict) -> None:
    """Share of values containing non-Latin letters, per source x country x column,
    with a breakdown of which scripts occur (computed on the non-Latin subset only)."""
    section("6. Non-Latin script share (value contains >=1 non-Latin letter)")
    print(f"{'source':<10}{'country':<12}{'rows':>10}{'name':>9}{'address':>9}  scripts (name | address)")
    for k, t in tables.items():
        for c in pc.unique(t["country"]).to_pylist():
            sub = t.filter(pc.equal(t["country"], c))
            shares, mixes = [], []
            for col in TEXT_COLS:
                has = pc.match_substring_regex(sub[col], NON_LATIN_RE)
                shares.append(pc.mean(pc.cast(has, pa.int8())).as_py() or 0.0)
                nl = sub[col].filter(has)
                mix = []
                if len(nl):
                    for s in SCRIPTS:
                        n = pc.sum(pc.match_substring_regex(nl, rf"\p{{{s}}}")).as_py()
                        if n:
                            mix.append((s, n))
                    mix.sort(key=lambda x: -x[1])
                mixes.append(",".join(f"{s[:4]}{100 * n / len(sub):.1f}" for s, n in mix[:3]) or "-")
            print(f"{k:<10}{trunc(c, 11) or '<blank>':<12}{sub.num_rows:>10,}{pct(shares[0]):>9}"
                  f"{pct(shares[1]):>9}  {mixes[0]} | {mixes[1]}")
    print("(script columns: share of that country's rows containing each script, top 3)")


def lookup(tables: dict, split: str, ids: list) -> dict:
    """entity_id -> (name, address, country) for a small list of ids across all sources."""
    out = {}
    idset = pa.array(ids)
    for s in (1, 2, 3):
        t = tables[f"{split}/S{s}"]
        for r in t.filter(pc.is_in(t["entity_id"], idset)).to_pylist():
            out[r["entity_id"]] = (r["business_name"], r["business_address"], r["country"])
    return out


def report_samples(tables: dict, pairs: pa.Table, seed: int) -> None:
    """Print 15 random GT matched pairs and 10 random France test rows."""
    import random
    rng = random.Random(seed)

    section("7a. 15 random matched pairs (train)  S1 -> match")
    idx = rng.sample(range(pairs.num_rows), min(15, pairs.num_rows))
    sample = pairs.take(pa.array(idx)).to_pylist()
    info = lookup(tables, "train", [r["s1"] for r in sample] + [r["match"] for r in sample])
    for r in sample:
        a = info.get(r["s1"], ("?", "?", "?"))
        b = info.get(r["match"], ("?", "?", "?"))
        print(f"[{a[2]}] {r['s1']:<14} {trunc(a[0], 38):<38} | {trunc(a[1], 60)}")
        print(f"{'':>{len(a[2]) + 3}}{r['match']:<14} {trunc(b[0], 38):<38} | {trunc(b[1], 60)}"
              + (f"  [{b[2]}]" if b[2] != a[2] else ""))

    section("7b. 10 random France rows (test, any source)")
    fr = []
    for s in (1, 2, 3):
        t = tables[f"test/S{s}"]
        fr += t.filter(pc.equal(t["country"], "France")).to_pylist()
    if not fr:
        print("no rows with country == 'France' in test")
        return
    for r in rng.sample(fr, min(10, len(fr))):
        print(f"{r['entity_id']:<14} {trunc(r['business_name'], 40):<40} | {trunc(r['business_address'], 70)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # non-Latin text on Windows consoles

    tables, gt = load_all(args.data_dir)
    report_counts(tables)
    report_nulls(tables)
    pairs = report_gt(gt, tables)
    report_scripts(tables)
    report_samples(tables, pairs, args.seed)


if __name__ == "__main__":
    main()
