"""B's features_for() (two aligned column->list dicts) and the train-side feature contract. Run: python tests/test_features_contract.py"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd, pyarrow as pa
import features as F
from sklearn.feature_extraction.text import TfidfVectorizer
from train import attach_labels, feature_columns

def side(rows):
    """{column: [values]} in the layout features.columns_of produces."""
    base = dict(entity_id="", name_norm="", name_core="", name_phon="", name_acronym="", name_alt_core="",
                name_alt_phon="", name_non_latin=False, addr_norm="", addr_phon="", addr_nums="",
                addr_postal="", addr_blank=True, addr_non_latin=False)
    recs = [{**base, **r} for r in rows]
    return {c: [r[c] for r in recs] for c in F.REC_COLS}

s1 = dict(entity_id="S1-1", name_norm="tata motors private limited", name_core="tata motors", name_phon="tt mtrs",
          name_acronym="tm", addr_norm="12 mg road pune", addr_phon="12 mg rd pn", addr_nums="12", addr_blank=False)
a = side([s1, s1, s1])
b = side([dict(s1, entity_id="S2-1", name_non_latin=True), dict(entity_id="S3-9", name_norm="blank co", name_core="blank", name_phon="blnk"),
          dict(entity_id="S3-8")])
texts = [x for x in a["addr_norm"] + b["addr_norm"] if x]; ph = [x for x in a["addr_phon"] + b["addr_phon"] if x]
vecs = (TfidfVectorizer(**{**F.TFIDF_PARAMS, "min_df": 1}).fit(texts), TfidfVectorizer(**{**F.TFIDF_PARAMS, "min_df": 1}).fit(ph))
f = F.features_for(a, b, vecs)
assert list(f) == F.BASE_FEATURES and len(f) == 31, (len(f), set(f) ^ set(F.BASE_FEATURES))
assert all(v.dtype == np.float32 and v.shape == (3,) and np.isfinite(v).all() for v in f.values())
assert f["name_core_ratio"][0] == 1.0 and f["name_missing"][2] == 1.0 and f["addr_blank"][1] == 1.0 and f["is_s3"].tolist() == [0.0, 1.0, 1.0]

# train-side contract: keys/label/provenance never features, int8 label included; nothing hard-coded
tbl = pd.DataFrame({"s1_id": ["a", "b"], "cand_id": ["c", "d"], "label": np.array([1, 0], dtype="int8"), "neg_kind": ["pos", "overlap"],
                    **{k: v[:2] for k, v in f.items()}})
cols = feature_columns(tbl)
assert cols == F.BASE_FEATURES and "label" not in cols and "neg_kind" not in cols
assert attach_labels(tbl)["y"].tolist() == [1, 0]
extra = tbl.assign(retr_score=[0.5, 0.2], retr_rank=[1, 3])
assert feature_columns(extra) == F.BASE_FEATURES + ["retr_score", "retr_rank"]   # new columns flow through automatically
try:
    attach_labels(pd.concat([tbl, tbl])); raise SystemExit("duplicates not rejected")
except ValueError:
    pass
print("features contract OK")
