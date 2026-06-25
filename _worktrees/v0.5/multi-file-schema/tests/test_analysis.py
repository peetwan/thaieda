"""ทดสอบ thaieda.analysis — การวิเคราะห์ความสัมพันธ์ของคอลัมน์กับ target."""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import pytest

from thaieda.analysis import TargetAssociation, analyze_target


def _scipy_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("scipy") is not None


# ----------------------------------------------------------- numeric × numeric
def test_numeric_numeric_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    y = 2 * x + rng.normal(0, 0.1, 100)  # สัมพันธ์เชิงบวกแรง
    df = pd.DataFrame({"x": x, "y": y})
    results = analyze_target(df, "y")
    assert len(results) == 1
    assoc = results[0]
    assert isinstance(assoc, TargetAssociation)
    assert assoc.association_type == "correlation"
    assert assoc.column == "x"
    assert assoc.target == "y"
    assert assoc.score > 0.9


# ------------------------------------------------------ numeric × categorical
def test_numeric_target_categorical_column_anova():
    # กลุ่ม A ค่าต่ำ, กลุ่ม B ค่าสูง -> ANOVA F สูง
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "value": list(range(10)) + list(range(100, 110)),
        }
    )
    results = analyze_target(df, "value")
    assoc = next(a for a in results if a.column == "group")
    assert assoc.association_type == "anova"
    assert assoc.score > 0


def test_categorical_target_numeric_column_anova():
    df = pd.DataFrame(
        {
            "label": ["yes"] * 10 + ["no"] * 10,
            "score": list(range(100, 110)) + list(range(10)),
        }
    )
    results = analyze_target(df, "label")
    assoc = next(a for a in results if a.column == "score")
    assert assoc.association_type == "anova"


# ---------------------------------------------------- categorical × categorical
def test_categorical_categorical_chi_square():
    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F", "M", "F", "M", "F"] * 3,
            "pass": ["y", "y", "n", "n", "y", "n", "y", "n"] * 3,
        }
    )
    results = analyze_target(df, "pass")
    assoc = next(a for a in results if a.column == "sex")
    assert assoc.association_type == "chi_square"
    assert assoc.score >= 0


# ------------------------------------------------------------------- to_dict
def test_target_association_to_dict():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12]})
    results = analyze_target(df, "y")
    d = results[0].to_dict()
    for key in ("column", "target", "association_type", "score", "p_value", "description_th"):
        assert key in d
    assert isinstance(d["description_th"], str)


# ------------------------------------------------------------------- errors
def test_analyze_target_missing_column_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(KeyError):
        analyze_target(df, "nope")


def test_analyze_target_unsupported_target_raises():
    # คอลัมน์ข้อความ cardinality สูง -> ใช้เป็น target ไม่ได้
    df = pd.DataFrame(
        {
            "text": [f"ข้อความยาว ๆ ที่ไม่ซ้ำกัน {i}" for i in range(20)],
            "x": list(range(20)),
        }
    )
    with pytest.raises(ValueError):
        analyze_target(df, "text")


# ------------------------------------------------------------- scipy fallback
def test_analyze_target_without_scipy_still_computes_score(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12]})
    results = analyze_target(df, "y")
    assert len(results) == 1
    assoc = results[0]
    assert assoc.association_type == "correlation"
    assert assoc.score > 0.99  # ยังคำนวณ r ได้
    assert math.isnan(assoc.p_value)  # แต่ p-value เป็น NaN เมื่อไม่มี scipy


@pytest.mark.skipif(not _scipy_installed(), reason="scipy not installed")
def test_analyze_target_with_scipy_computes_pvalue():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12]})
    results = analyze_target(df, "y")
    assert not math.isnan(results[0].p_value)
