"""Stage 1 — normalize names and addresses for every source; cache to artifacts/*.parquet.

Pipeline per text column (vectorized with pyarrow compute; Python only touches the
distinct non-ASCII values):
  1. non-ASCII values -> ftfy (mojibake fix + NFKC) -> anyascii (script-agnostic
     transliteration, also strips accents). ASCII values pass through untouched.
  2. lowercase, '&' -> 'and', drop . ' ` (so "L.L.C." -> "llc"), other non-alnum -> space.
  3. token dictionaries: legal suffixes (names), street/state abbreviations (addresses).

Name outputs : name_norm (full cleaned), name_core (main name, legal suffixes removed),
               name_alt_core (part after dba / fka / aka, '' if none), name_acronym,
               name_sorted (core tokens sorted), name_non_latin (bool).
Address outputs: addr_norm, addr_nums (numeric tokens), addr_postal (5-6 digit tokens),
               addr_blank, addr_non_latin.

No rule depends on the country value; dictionaries are small and hand-written.

Usage:
  python src/normalize.py --split train          # build + cache train parquets
  python src/normalize.py --split train --check  # transliteration check (needs GT)
"""
import argparse
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import ART, DATA, SEED, SOURCES, norm_path, read_tsv, source_path, timed, utf8_stdout

NON_LATIN_RE = r"[^\P{L}\p{Latin}]"  # RE2: a letter that is not Latin script

# ---------------------------------------------------------------- dictionaries
# Legal-form / filler tokens removed from name_core (incl. anyascii renderings of
# Devanagari "प्राइवेट लिमिटेड" -> "praivet limited").
LEGAL = """pvt private praivet praibhet piraivet praivrr pvtltd ltd limited limitid limitet
limtid limirrd limited ltdd llc llp lp inc incorporated corp corporation co company cos
sarl sas sasu eurl sa sci snc gmbh plc pllc pc the""".split()
STOP = {"and", "of", "the", "for", "a", "an", "de", "la", "le", "des", "du", "et"}
# dba-style markers (applied after punctuation cleaning, so "d/b/a" -> "d b a").
DBA_RE = r"\b(?:dba|d b a|fka|f k a|aka|a k a|formerly known as|formerly|doing business as)\b"

STREET = {
    "st": "street", "str": "street", "rd": "road", "dr": "drive", "drv": "drive",
    "ave": "avenue", "av": "avenue", "avn": "avenue", "ln": "lane", "blvd": "boulevard",
    "bd": "boulevard", "bvd": "boulevard", "crt": "court", "cir": "circle",
    "hwy": "highway", "pkwy": "parkway", "pky": "parkway", "pl": "place", "sq": "square",
    "ter": "terrace", "trl": "trail", "cres": "crescent", "expy": "expressway",
    "ste": "suite", "apt": "apartment", "flr": "floor", "bldg": "building",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "nw": "northwest", "se": "southeast", "sw": "southwest",
    "ft": "fort", "hts": "heights",
    # India
    "ngr": "nagar", "mkt": "market", "opp": "opposite", "nr": "near", "dist": "district",
    "distt": "district", "sec": "sector",
    # France (R. De Dieppe, Imp., Bd, Pl.)
    "r": "rue", "imp": "impasse", "rte": "route",
    # NB: ct/fl/ne/mt/ch deliberately absent — they collide with state codes (CT, FL, NE, MT, CH).
}
# Full state / territory names -> abbreviation (US + India). Name -> code is safer than
# code -> name because short codes collide with ordinary words.
STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "dilli": "dl", "jammu and kashmir": "jk", "ladakh": "la",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}


# ---------------------------------------------------------------- primitives
def translit(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """ftfy + anyascii on distinct non-ASCII values only; ASCII values are returned as-is."""
    from anyascii import anyascii
    from ftfy import fix_text
    col = col.combine_chunks()
    ascii_ = pc.string_is_ascii(col)
    uniq = pc.unique(col.filter(pc.invert(ascii_)))
    if len(uniq) == 0:
        return col
    fixed = pa.array([anyascii(fix_text(s, normalization="NFKC")) for s in uniq.to_pylist()],
                     type=pa.string())
    idx = pc.index_in(col, value_set=uniq)
    return pc.if_else(ascii_, col, pc.take(fixed, idx))


def clean(col) -> pa.Array:
    """Lowercase ASCII text; '&'->and; drop . ' `; other non-alnum -> space; squeeze spaces."""
    x = pc.ascii_lower(col)
    x = pc.replace_substring(x, "&", " and ")
    x = pc.replace_substring_regex(x, r"[.'`]", "")
    x = pc.replace_substring_regex(x, r"[^a-z0-9]+", " ")
    return pc.utf8_trim_whitespace(x)


def squeeze(col) -> pa.Array:
    """Collapse runs of spaces and trim."""
    return pc.utf8_trim_whitespace(pc.replace_substring_regex(col, r" {2,}", " "))


def map_tokens(col, mapping: dict) -> pa.Array:
    """Replace whole space-separated tokens via `mapping` (value '' deletes the token)."""
    lists = pc.split_pattern(col, " ")
    if isinstance(lists, pa.ChunkedArray):
        lists = lists.combine_chunks()
    flat = lists.flatten()
    keys = pa.array(list(mapping.keys()), type=pa.string())
    vals = pa.array(list(mapping.values()), type=pa.string())
    idx = pc.index_in(flat, value_set=keys)
    new = pc.if_else(pc.is_null(idx), flat, pc.take(vals, idx))
    rebuilt = pa.ListArray.from_arrays(lists.offsets, new)
    return squeeze(pc.binary_join(rebuilt, " "))


def replace_phrases(col, mapping: dict) -> pa.Array:
    """Replace multi-word phrases (e.g. 'north carolina' -> 'nc') with word boundaries."""
    for k, v in mapping.items():
        if " " in k:
            col = pc.replace_substring_regex(col, rf"\b{k}\b", v)
    return col


def phonetic(col) -> pa.Array:
    """Script-agnostic phonetic skeleton of cleaned ASCII text, applied to BOTH sides.

    Designed for English words written in an Indic script and romanized by anyascii
    ("finance" ~ "phainans", "industries" ~ "imdstrij", "logistics" ~ "lojistiks"):
    digraphs ph/bh/kh/gh/th/dh/sh -> single letter, x->ks, q->k, w->v, z->s, y->i,
    soft c/g before e/i -> s/j, other c -> k, m before a consonant -> n (anusvara),
    word-final j -> s, drop non-initial vowels and h, collapse repeated letters.
    Digits are kept. Pure string rules; nothing language- or country-specific.
    """
    x = col
    for a, b in (("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"),
                 ("dh", "d"), ("sh", "s"), ("ch", "c"), ("x", "ks"), ("q", "k"),
                 ("w", "v"), ("z", "s"), ("y", "i")):
        x = pc.replace_substring(x, a, b)
    x = pc.replace_substring_regex(x, r"c([ei])", r"s\1")
    x = pc.replace_substring_regex(x, r"g([ei])", r"j\1")
    x = pc.replace_substring(x, "c", "k")
    x = pc.replace_substring_regex(x, r"m([bdfgjklnpqrstvz])", r"n\1")
    x = pc.replace_substring_regex(x, r"j\b", "s")
    x = pc.replace_substring_regex(x, r"\B[aeiouh]", "")
    for ch in "bdfgjklmnprstv":
        x = pc.replace_substring_regex(x, ch + "{2,}", ch)
    return x


def acronym(core) -> pa.Array:
    """First letter of each non-stopword token when the name has >= 2 such tokens."""
    no_stop = map_tokens(core, {w: "" for w in STOP})
    lists = pc.split_pattern(no_stop, " ")
    first = pc.utf8_slice_codeunits(lists.flatten(), 0, 1)
    acr = pc.binary_join(pa.ListArray.from_arrays(lists.offsets, first), "")
    n_tok = pc.list_value_length(lists)
    ok = pc.and_(pc.greater_equal(n_tok, 2), pc.greater(pc.utf8_length(no_stop), 0))
    return pc.if_else(ok, acr, "")


# ---------------------------------------------------------------- per-column pipelines
def normalize_names(raw: pa.ChunkedArray) -> dict:
    """Name columns: name_norm, name_core, name_alt_core, name_acronym, name_sorted, name_non_latin."""
    non_latin = pc.match_substring_regex(raw, NON_LATIN_RE)
    norm = clean(translit(raw))
    # split on the first dba/fka/aka marker: main name before, alternate name after
    parts = pc.split_pattern_regex(norm, DBA_RE, max_splits=1)
    main = squeeze(pc.list_element(parts, 0))
    alt = squeeze(pc.binary_join(pc.list_slice(parts, 1, 2), ""))  # '' when no marker
    legal = {w: "" for w in LEGAL}

    def core_of(x):
        c = map_tokens(x, legal)
        return pc.if_else(pc.equal(c, ""), x, c)  # never empty out a name

    core = core_of(main)
    alt_core = pc.if_else(pc.equal(alt, ""), "", core_of(alt))
    # if the main part is empty (name *starts* with dba), promote the alternate
    core = pc.if_else(pc.equal(core, ""), alt_core, core)
    sorted_ = pa.array([" ".join(sorted(s.split())) for s in core.to_pylist()], type=pa.string())
    return {"name_norm": norm, "name_core": core, "name_alt_core": alt_core,
            "name_acronym": acronym(core), "name_sorted": sorted_,
            "name_phon": phonetic(core), "name_alt_phon": phonetic(alt_core),
            "name_non_latin": non_latin}


def normalize_addresses(raw: pa.ChunkedArray) -> dict:
    """Address columns: addr_norm, addr_nums, addr_postal, addr_blank, addr_non_latin."""
    non_latin = pc.match_substring_regex(raw, NON_LATIN_RE)
    x = clean(translit(raw))
    x = replace_phrases(x, STATES)
    x = map_tokens(x, {**STREET, **{k: v for k, v in STATES.items() if " " not in k}})
    # numeric tokens / postal-like tokens: Python regex over distinct values (cheap)
    uniq = pc.unique(x)
    nums_u, post_u = [], []
    num_re, post_re = re.compile(r"\b\d+\b"), re.compile(r"\b\d{5,6}\b")
    for s in uniq.to_pylist():
        nums_u.append(" ".join(dict.fromkeys(num_re.findall(s))))
        post_u.append(" ".join(dict.fromkeys(post_re.findall(s))))
    idx = pc.index_in(x, value_set=uniq)
    return {"addr_norm": x, "addr_phon": phonetic(x),
            "addr_nums": pc.take(pa.array(nums_u, type=pa.string()), idx),
            "addr_postal": pc.take(pa.array(post_u, type=pa.string()), idx),
            "addr_blank": pc.equal(x, ""), "addr_non_latin": non_latin}


def normalize_source(split: str, src: str) -> pa.Table:
    """Normalize one raw source file and cache it as parquet."""
    raw = read_tsv(source_path(split, src))
    with timed(f"{split}/{src} names ({raw.num_rows:,})"):
        names = normalize_names(raw["business_name"])
    with timed(f"{split}/{src} addresses"):
        addrs = normalize_addresses(raw["business_address"])
    out = pa.table({"entity_id": raw["entity_id"],
                    "country": pc.utf8_trim_whitespace(raw["country"]),
                    **names, **addrs})
    ART.mkdir(exist_ok=True)
    pq.write_table(out, norm_path(split, src), compression="zstd")
    return out


# ---------------------------------------------------------------- transliteration check
def translit_check(n_show: int = 30) -> None:
    """Compare S1 names with their non-Latin S2/S3 matches (raw vs normalized) and
    contrast fuzzy similarity with Latin-Latin matched pairs."""
    import pandas as pd
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    utf8_stdout()
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv").to_pandas()
    pairs = (gt.assign(m=gt["matched_entity_ids"].str.split(","))
               .explode("m").query("m.notna() and m != ''")[["source1_entity_id", "m"]]
               .rename(columns={"source1_entity_id": "s1"}))
    cols = ["entity_id", "country", "name_norm", "name_core", "name_phon", "name_non_latin"]
    s1 = pq.read_table(norm_path("train", "S1"), columns=cols).to_pandas().set_index("entity_id")
    other = pd.concat([pq.read_table(norm_path("train", s), columns=cols).to_pandas()
                       for s in ("S2", "S3")]).set_index("entity_id")
    raw_names = pd.concat([read_tsv(source_path("train", s)).select(["entity_id", "business_name"])
                           .to_pandas() for s in ("S2", "S3")]).set_index("entity_id")["business_name"]
    raw_s1 = read_tsv(source_path("train", "S1")).select(["entity_id", "business_name"]) \
        .to_pandas().set_index("entity_id")["business_name"]

    p = pairs.join(s1, on="s1").join(other, on="m", rsuffix="_m")
    p["script"] = np.where(p["name_non_latin_m"], "non-Latin", "Latin")
    ind = p["country"] == "India"

    rng = np.random.default_rng(SEED)
    nl = p[p["name_non_latin_m"]]
    show = nl.iloc[rng.choice(len(nl), size=min(n_show, len(nl)), replace=False)]
    print(f"\n{n_show} matched pairs whose S2/S3 name is non-Latin "
          f"({len(nl):,} such pairs = {len(nl) / ind.sum():.1%} of India pairs)")
    print(f"{'S1 name':<34} | {'original S2/S3 name':<30} | normalized (name_norm)")
    for _, r in show.iterrows():
        print(f"{raw_s1[r['s1']][:34]:<34} | {raw_names[r['m']][:30]:<30} | {r['name_norm_m'][:45]}")

    print("\nMean rapidfuzz token_set_ratio, S1 vs matched S2/S3 (0-100):")
    print(f"{'group':<26}{'pairs':>10}{'name_norm':>11}{'name_core':>11}{'ratio(core)':>12}{'name_phon':>11}")
    groups = [("India non-Latin", ind & p["name_non_latin_m"]),
              ("India Latin-Latin", ind & ~p["name_non_latin_m"]),
              ("all Latin-Latin", ~p["name_non_latin_m"])]
    for label, m in groups:
        g = p[m]
        if len(g) > 300_000:
            g = g.sample(300_000, random_state=SEED)
        ts_n = cpdist(g["name_norm"].tolist(), g["name_norm_m"].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
        ts_c = cpdist(g["name_core"].tolist(), g["name_core_m"].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
        r_c = cpdist(g["name_core"].tolist(), g["name_core_m"].tolist(), scorer=fuzz.ratio, workers=-1)
        ts_p = cpdist(g["name_phon"].tolist(), g["name_phon_m"].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
        print(f"{label:<26}{int(m.sum()):>10,}{ts_n.mean():>11.1f}{ts_c.mean():>11.1f}{r_c.mean():>12.1f}"
              f"{ts_p.mean():>11.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--check", action="store_true", help="run the transliteration check only")
    args = ap.parse_args()
    if args.check:
        translit_check()
        return
    for src in args.sources:
        with timed(f"normalize {args.split}/{src}"):
            normalize_source(args.split, src)


if __name__ == "__main__":
    sys.exit(main())
