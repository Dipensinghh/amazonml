"""Tests for fallback.macro_f05 and fallback.combine."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fallback import combine, macro_f05  # noqa: E402


def _df(pairs):
    """(s1_id, cand_id) frame from tuples."""
    return pd.DataFrame(pairs, columns=["s1_id", "cand_id"])


def test_macro_f05_cases():
    """Singleton rules, partial overlap, and S1s absent from both frames."""
    gt = _df([("S1-a", "S2-1"), ("S1-a", "S3-1"), ("S1-b", "S2-2")])
    pred = _df([("S1-a", "S2-1"), ("S1-c", "S2-9")])
    # a: p=1, r=0.5 -> 1.25*0.5/(0.25+0.5)=0.8333 | b: pred empty, truth not -> 0
    # c: truth empty, pred not -> 0 | d: both empty -> 1
    f = macro_f05(pred, gt, ["S1-a", "S1-b", "S1-c", "S1-d"])
    assert f == pytest.approx((1.25 * 0.5 / 0.75 + 0 + 0 + 1) / 4)
    assert macro_f05(gt, gt, ["S1-a", "S1-b"]) == 1.0
    assert macro_f05(_df([]), _df([]), ["S1-x"]) == 1.0
    # tp = 0 with both non-empty -> 0
    assert macro_f05(_df([("S1-a", "S2-7")]), gt, ["S1-a"]) == 0.0


def test_combine_blank_address_renormalizes():
    """Blank address drops the address term instead of scoring it as 0."""
    sims = {"name_core": np.array([80, 80], np.float32), "addr_norm": np.array([0, 0], np.float32),
            "name_phon": np.array([60, 60], np.float32), "addr_blank": np.array([True, False])}
    s = combine(sims, (1, 1, 1))
    assert s[0] == pytest.approx(0.7) and s[1] == pytest.approx(140 / 300)
