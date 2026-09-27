"""The submission package (code/business_entity_resolution/) must be an exact copy of the repo.

src/, the shipped model bundle and the shared address TF-IDF are copied into the package; this
test fails if a later edit to src/ (or a model swap) is not copied too. To resync:
  cp src/*.py src/fallback_params.json code/business_entity_resolution/src/
  cp models/<MODEL_DIR>/* code/business_entity_resolution/models/<MODEL_DIR>/
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "code" / "business_entity_resolution"


def _model_dir() -> str:
    """MODEL_DIR as configured in the package's run.py."""
    m = re.search(r'^MODEL_DIR = "([^"]+)"', (PKG / "run.py").read_text(encoding="utf-8"), re.M)
    assert m, "run.py must define MODEL_DIR"
    return m.group(1)


def _src_files():
    """Every pipeline source file in the repo's src/."""
    return sorted(p.name for p in (REPO / "src").iterdir() if p.suffix in (".py", ".json"))


def test_package_src_matches_repo():
    """Same file set, byte-identical content."""
    pkg = sorted(p.name for p in (PKG / "src").iterdir() if p.suffix in (".py", ".json"))
    assert pkg == _src_files()
    stale = [n for n in pkg if (PKG / "src" / n).read_bytes() != (REPO / "src" / n).read_bytes()]
    assert not stale, f"package src/ out of date: {stale}"


@pytest.mark.parametrize("name", ["lgbm.txt", "bundle.json", "threshold.json", "val_report.json"])
def test_package_model_matches_repo(name):
    """The shipped bundle is the repo's MODEL_DIR bundle."""
    md = _model_dir()
    assert (PKG / md / name).read_bytes() == (REPO / md / name).read_bytes()


def test_package_tfidf_matches_repo():
    """Train and test features must share the tracked address TF-IDF."""
    assert (PKG / "artifacts" / "addr_tfidf.joblib").read_bytes() == \
        (REPO / "artifacts" / "addr_tfidf.joblib").read_bytes()
