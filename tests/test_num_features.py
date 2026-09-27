"""Leading-zero fix for house-number features. Run: python tests/test_num_features.py"""
import sys
sys.path.insert(0, "src")
from features import num_features, token_stats

f = num_features(["5705", "12 005", "0", "", "007 12a", "68"], ["005705", "5", "00", "5", "7 99", "681"])
import numpy as np
assert np.allclose(f["num_agree"], [1.0, 0.5, 1.0, 0.0, 1 / 3, 0.0]), f["num_agree"].tolist()
assert f["num_conflict"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
assert f["num_missing"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
assert all(v.dtype.name == "float32" for v in f.values())

# postal tokens keep leading zeros: "01234" and "1234" are different postal codes
agree, conflict, _ = token_stats(["01234"], ["1234"])
assert agree[0] == 0.0 and conflict[0] == 1.0
print("num features OK")
