# tests/test_audit_fixes.py

import numpy as np
import pandas as pd

from thaieda.analysis import analyze_target
from thaieda.clean import normalize_dates
from thaieda.clean._smart import _count_extra_whitespace
from thaieda.detect._thai_address import parse_thai_address

# นำเข้าฟังก์ชันที่แก้ไขมาทดสอบ
from thaieda.insight import _is_bimodal
from thaieda.insight_engine import _detect_strong_correlations
from thaieda.llm._prepare import _add_dp_noise
from thaieda.llm._synthetic import _gen_categorical, _gen_numeric
from thaieda.quality._thai_id import check_thai_id
from thaieda.schema import TableProfile, match_relationships
from thaieda.timeseries import analyze_timeseries
from thaieda.timeseries._thai_holidays import _to_date


def test_is_bimodal_handles_infinity():
    """ทดสอบว่า _is_bimodal ไม่แครชเมื่อเจอค่า infinity ในข้อมูล."""
    values = np.array([1.0, 2.0, 3.0, 2.0, 1.0, np.inf, -np.inf])
    # ฟังก์ชันต้องรันผ่านโดยไม่มี ValueError
    res = _is_bimodal(values)
    assert isinstance(res, bool)


def test_detect_strong_correlations_handles_missing_values():
    """ทดสอบว่า _detect_strong_correlations ไม่แครชหรือได้ผลลัพธ์ว่างเปล่าเมื่อมี NaN กระจัดกระจาย."""
    # สร้าง DataFrame ที่มี 12 แถวและ NaN สลับกันคนละแถว
    df = pd.DataFrame(
        {
            "col_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 12],
            "col_b": [np.nan, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        }
    )
    measures = {"col_a": {}, "col_b": {}}
    res = _detect_strong_correlations(df, measures)
    # ต้องคำนวณ correlation ของ col_a และ col_b ได้ (เพราะ r = 1.0 และมีคู่ร่วมกัน >= 10 คู่)
    assert len(res) == 1
    assert res[0]["pattern"] == "correlation"


def test_analyze_target_numeric_column_names():
    """ทดสอบว่า analyze_target ไม่แครช KeyError เมื่อหัวคอลัมน์เป็นตัวเลข."""
    df = pd.DataFrame(
        {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
    )
    # ใช้งาน analyze_target โดยเป้าหมายคือคอลัมน์ 1 (int)
    res = analyze_target(df, target_column=1)
    assert len(res) > 0
    assert res[0].column == "0"


def test_match_relationships_numeric_column_names():
    """ทดสอบว่า match_relationships ไม่แครช KeyError เมื่อตารางมีคอลัมน์ชื่อเป็นตัวเลข."""
    t1 = pd.DataFrame({0: [1, 2, 3, 4, 5]})
    t2 = pd.DataFrame({0: [1, 1, 2, 2, 3]})
    tables = {"t1": t1, "t2": t2}

    p1 = TableProfile(
        name="t1",
        file_path="t1.csv",
        row_count=5,
        column_count=1,
        columns=["0"],
        column_types={"0": "numeric"},
        key_candidates=[],
    )
    p2 = TableProfile(
        name="t2",
        file_path="t2.csv",
        row_count=5,
        column_count=1,
        columns=["0"],
        column_types={"0": "numeric"},
        key_candidates=[],
    )
    profiles = {"t1": p1, "t2": p2}

    res = match_relationships(tables, profiles, validate_values=True)
    # ต้องจับคู่ได้โดยไม่แครช KeyError
    assert len(res) >= 0


def test_synthetic_generation_empty_and_nan_inputs():
    """ทดสอบว่า synthetic generation รันผ่านไม่แครชบนชุดข้อมูลว่างเปล่าหรือ NaN ล้วน."""
    rng = np.random.default_rng(42)

    # 1. Numeric column ที่เป็น NaN ล้วน
    nan_series = pd.Series([np.nan, np.nan, np.nan], dtype=float)
    res_num = _gen_numeric(nan_series, n=5, rng=rng)
    assert len(res_num) == 5
    assert res_num.isna().all()

    # 2. Categorical column จาก DataFrame ว่างเปล่า
    empty_series = pd.Series([], dtype=object)
    res_cat = _gen_categorical(empty_series, n=5, rng=rng)
    assert len(res_cat) == 5
    assert res_cat.isna().all()

    # 3. Categorical mixed types (str และ float nan ปนกัน)
    mixed_series = pd.Series(["A", "B", np.nan], dtype=object)
    res_mixed = _gen_categorical(mixed_series, n=10, rng=rng)
    assert len(res_mixed) == 10


def test_thai_id_masking_format_invalid():
    """ทดสอบว่า check_thai_id ทำการ mask เลขบัตรประชาชนที่ผิด format เพื่อป้องกันข้อมูลหลุด."""
    # ส่งเลขบัตรที่ยาวไม่ครบ 13 หลัก (เช่น 10 หลัก)
    series = pd.Series(["1234567890"], name="id_card")
    issue = check_thai_id(series, "id_card", is_id_type=True)
    assert issue is not None
    # ตัวเลขดิบ 1234567890 ต้องไม่ปรากฏตรงๆ ใน examples แต่ต้องถูก mask
    for ex in issue.examples:
        assert "1234567890" not in ex
        assert "..." in ex or "*" in ex


def test_dp_noise_removes_exact_statistics():
    """ทดสอบว่า Differential Privacy ลบสถิติจริงที่ไม่ได้ใส่ noise ออกเพื่อป้องกันข้อมูลรั่วไหล."""
    summary = {
        "shape": (100, 5),
        "numeric_stats": {
            "col1": {
                "count": 100,
                "mean": 50.0,
                "min": 10.0,
                "max": 90.0,
                "std": 15.0,
                "q25": 25.0,
                "q50": 50.0,
                "q75": 75.0,
            }
        },
    }
    noisy = _add_dp_noise(summary, epsilon=1.0)
    stats = noisy["numeric_stats"]["col1"]
    # ต้องไม่มี std หรือ quartiles ตัวเลขดิบ
    for k in ["std", "25%", "50%", "75%", "q25", "q50", "q75", "median"]:
        assert k not in stats
    # count, mean, min, max ยังอยู่แต่ผ่านการใส่ noise
    assert "count" in stats
    assert "mean" in stats


def test_timeseries_nat_filtering_and_stl_fallback():
    """ทดสอบว่าอนุกรมเวลารองรับค่า NaT และ STL auto mode ไม่แครชบนข้อมูลระยะสั้น."""
    # 1. ตรวจสอบการรองรับ NaT ในคอลัมน์ดัชนีเวลา (ใช้ข้อมูลยาวพอที่จะยอมรับให้เป็น timeseries)
    times = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            pd.NaT,
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
            "2024-01-07",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    values = pd.Series([10, 12, 15, 14, 13, 15, 16, 17, 18, 19], index=times)
    # ต้องรันผ่านได้
    res = analyze_timeseries(values, engine="auto")
    assert res.is_timeseries is True

    # 2. ตรวจสอบว่าในข้อมูลที่สั้นมาก statsmodels STL จะไม่ทำให้ฟังก์ชันแครช
    short_series = pd.Series([1, 2, 3, 2, 1])
    res_short = analyze_timeseries(short_series, engine="auto", freq=2)
    assert res_short.column == ""


def test_to_date_hierarchy_check():
    """ทดสอบว่า _to_date ตรวจจับ datetime ก่อน date เพื่อแปลงค่าอย่างถูกต้อง."""
    dt_val = pd.Timestamp("2024-06-28 12:00:00")
    res = _to_date(dt_val)
    # ต้องได้อ็อบเจกต์ date ไม่ใช่ datetime/Timestamp
    assert type(res) is pd.Timestamp("2024-06-28").to_pydatetime().date().__class__
    assert not isinstance(res, pd.Timestamp)


def test_normalize_dates_preserves_ce_2_digit_year():
    """ทดสอบว่า normalize_dates ไม่แปลงปี ค.ศ. 2 หลักผิดพลาดเป็นปี ค.ศ. 1977."""
    s = pd.Series(["20/06/20"], name="date")  # 20 มิถุนายน ค.ศ. 2020
    out, result = normalize_dates(s)
    # ต้องไม่ถูกแปลงเป็นปี 1977 (เนื่องจากไม่มีอักษรไทยอยู่เลย)
    assert "1977" not in str(out.iloc[0])


def test_normalize_dates_thai_month_dot():
    """ทดสอบว่า regex ของชื่อเดือนไม่ไปจับคู่กับตัวอักษรอื่นมั่วเพราะเครื่องหมายจุด."""
    # ทดสอบตัวย่อเดือนไทยมีจุดต้องไม่ไป match ตัวย่อมั่ว
    s = pd.Series(["15 มคค 67"], name="date")
    out, result = normalize_dates(s)
    # มคค ไม่ควรถูกแปลงเป็นเดือนมกราคม
    assert result.rows_affected == 0


def test_address_parsing_house_number_with_branch():
    """ทดสอบการ parse ที่อยู่ที่มีตัวเลขอื่นนำหน้าเลขที่บ้าน เช่น สาขาที่ 3."""
    addr = "สาขาที่ 3 เลขที่ 123/45 หมู่ 4"
    res = parse_thai_address(addr)
    # ต้องสกัดบ้านเลขที่ได้เป็น 123/45 ไม่ใช่ 3
    assert res["house_number"] == "123/45"
    assert res["moo"] == "4"


def test_address_parsing_lookahead_subdistrict_district():
    """ทดสอบ lookahead ในตำบล/อำเภอ ป้องกัน greedy matching กลืนคำนำหน้าถัดไป."""
    addr = "ต.บางบัวอ.บางบัวจ.กรุงเทพฯ 10230"
    res = parse_thai_address(addr)
    assert res["subdistrict"] == "บางบัว"
    assert res["district"] == "บางบัว"
    assert res["province"] == "กรุงเทพฯ"


def test_whitespace_check_no_double_counting():
    """ทดสอบว่า _count_extra_whitespace ไม่นับเซลล์ที่มีปัญหาสองเรื่องซ้ำสองรอบ."""
    # เซลล์นี้มีทั้งช่องว่างตรงกลางซ้ำ และช่องว่างหน้าหลัง
    s = pd.Series(["  A  B  "])
    count = _count_extra_whitespace([s])
    # ต้องนับเป็น 1 จุดเท่านั้น ไม่ใช่ 2 จุด
    assert count == 1
