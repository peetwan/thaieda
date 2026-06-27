"""ทดสอบโมดูลวันหยุดนักขัตฤๅที่ไทย (v1.1).

ทดสอบ:
  - วันหยุดคงที่ (วันปีใหม่, วันพ่อ, วันรัฐธรรมนูญ)
  - วันหยุดเคลื่อนที่แบบประมาณ (สงกรานต์, วิสาขบูชา)
  - วันเสาร์-อาทิตย์
  - วันธรรมดา (ไม่ใช่วันหยุด)
  - flag_holiday_spikes ระบุ spike ที่ตรงกับวันหยุด
  - input format ต่าง ๆ (string, date, datetime)
"""

from __future__ import annotations

from datetime import date, datetime

from thaieda.timeseries._thai_holidays import (
    flag_holiday_spikes,
    holiday_name,
    is_thai_holiday,
)


class TestFixedHolidays:
    """ทดสอบวันหยุดคงที่ — ตรงตามเดือน+วันทุกปี."""

    def test_new_year_day(self):
        """วันที่ 1 มกราคม = วันขึ้นปีใหม่."""
        assert is_thai_holiday("2025-01-01")
        assert holiday_name("2025-01-01") == "วันขึ้นปีใหม่"

    def test_fathers_day(self):
        """วันที่ 5 ธันวาคม = วันพ่อ."""
        assert is_thai_holiday("2025-12-05")
        assert holiday_name("2025-12-05") == "วันพ่อ (วันเฉลิมพระชนมพรรษา รัชกาลที่ ๙)"

    def test_constitution_day(self):
        """วันที่ 10 ธันวาคม = วันรัฐธรรมนูญ."""
        assert is_thai_holiday("2025-12-10")
        assert holiday_name("2025-12-10") == "วันรัฐธรรมนูญ"

    def test_queen_mother_day(self):
        """วันที่ 12 สิงหาคม = วันสมเด็จพระบรมราชชนนี."""
        assert is_thai_holiday("2025-08-12")
        assert holiday_name("2025-08-12") == "วันสมเด็จพระบรมราชชนนี"


class TestMovableHolidays:
    """ทดสอบวันหยุดเคลื่อนที่แบบประมาณ."""

    def test_songkran_approximate(self):
        """สงกรานต์ ≈ 13-15 เมษายน."""
        assert is_thai_holiday("2025-04-13")
        assert is_thai_holiday("2025-04-15")
        assert "สงกรานต์" in holiday_name("2025-04-13")

    def test_visakha_approximate(self):
        """วิสาขบูชา ≅ ปลายเดือนพฤษภาคม."""
        # ตรวจว่าอย่างน้อย 1 วันในช่วง 20-31 พ.ค. เป็นวันหยุด
        found = any(is_thai_holiday(f"2025-05-{d:02d}") for d in range(20, 32))
        assert found, "ควรตรวจพบวันวิสาขบูชาในช่วงปลายพฤษภาคม"


class TestWeekend:
    """ทดสอบวันเสาร์-อาทิตย์."""

    def test_saturday(self):
        """วันเสาร์ = วันหยุด."""
        # 2025-01-04 เป็นวันเสาร์
        assert is_thai_holiday("2025-01-04")
        assert holiday_name("2025-01-04") == "วันเสาร์"

    def test_sunday(self):
        """วันอาทิตย์ = วันหยุด."""
        # 2025-01-05 เป็นวันอาทิตย์
        assert is_thai_holiday("2025-01-05")
        assert holiday_name("2025-01-05") == "วันอาทิตย์"

    def test_saturday_no_weekends(self):
        """วันเสาร์ = ไม่ใช่ตัวแทนวันหยุดเมื่อตั้ง include_weekends=False."""
        assert not is_thai_holiday("2025-01-04", include_weekends=False)

    def test_sunday_no_weekends(self):
        """วันอาทิตย์ = ไม่ใช่ตัวแทนวันหยุดเมื่อตั้ง include_weekends=False."""
        assert not is_thai_holiday("2025-01-05", include_weekends=False)


class TestNormalDays:
    """ทดสอบวันธรรมดา — ไม่ใช่วันหยุด."""

    def test_weekday_not_holiday(self):
        """วันธรรมดา = ไม่ใช่วันหยุด."""
        # 2025-01-06 เป็นวันจันทร์
        assert not is_thai_holiday("2025-01-06")
        assert holiday_name("2025-01-06") == ""

    def test_random_weekday(self):
        """วันอังคารธรรมดา = ไม่ใช่วันหยุด."""
        # 2025-01-07 เป็นวันอังคาร
        assert not is_thai_holiday("2025-01-07")


class TestInputFormats:
    """ทดสอบรูปแบบ input ต่าง ๆ."""

    def test_date_object(self):
        """รับ date object ได้."""
        d = date(2025, 12, 5)
        assert is_thai_holiday(d)
        assert holiday_name(d) == "วันพ่อ (วันเฉลิมพระชนมพรรษา รัชกาลที่ ๙)"

    def test_datetime_object(self):
        """รับ datetime object ได้."""
        dt = datetime(2025, 1, 1, 10, 30)
        assert is_thai_holiday(dt)
        assert holiday_name(dt) == "วันขึ้นปีใหม่"

    def test_invalid_string(self):
        """string ที่ไม่ใช่ ISO format → ไม่ใช่วันหยุด."""
        assert not is_thai_holiday("not-a-date")
        assert holiday_name("not-a-date") == ""

    def test_none_input(self):
        """None → ไม่ใช่วันหยุด."""
        assert not is_thai_holiday(None)
        assert holiday_name(None) == ""


class TestFlagHolidaySpikes:
    """ทดสอบ flag_holiday_spikes — ระบุ spike ที่ตรงกับวันหยุด."""

    def test_flags_holiday_spikes(self):
        """spike ที่ตรงวันหยุดถูกระบุชื่อวันหยุด."""
        dates = [
            "2025-01-01",  # วันขึ้นปีใหม่ (index 0)
            "2025-01-02",  # วันพฤหัส (index 1)
            "2025-01-04",  # วันเสาร์ (index 2)
            "2025-01-06",  # วันจันทร์ธรรมดา (index 3)
        ]
        anomalies = [0, 2, 3]
        result = flag_holiday_spikes(dates, anomalies)
        # index 0 = วันขึ้นปีใหม่, index 2 = วันเสาร์, index 3 = ไม่ใช่วันหยุด
        assert 0 in result
        assert 2 in result
        assert 3 not in result
        assert "ขึ้นปีใหม่" in result[0]
        assert "เสาร์" in result[2]

    def test_empty_anomalies(self):
        """anomalies ว่าง → ผลลัพธ์ว่าง."""
        dates = ["2025-01-01"]
        result = flag_holiday_spikes(dates, [])
        assert result == {}

    def test_index_out_of_range(self):
        """anomaly index เกินขอบเขต → ข้าม."""
        dates = ["2025-01-01"]
        result = flag_holiday_spikes(dates, [0, 5, 10])
        assert 0 in result
        assert 5 not in result
        assert 10 not in result
