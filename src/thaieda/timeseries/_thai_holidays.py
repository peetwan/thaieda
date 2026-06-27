"""วันหยุดนักขัตฤๅที่ไทย — สำหรับระบุ spike ที่เกิดจากวันหยุดใน timeseries (v1.1).

รวมวันหยุดที่ตรงตามวันที่ (fixed-date holidays) และวันหยุดที่เปลี่ยนวันที่ทุกปี
(movable holidays — อิงปฏิทินจันทรคติหรือประกาศแต่ละปี)

หลักการ:
  * วันหยุดคงที่ (fixed-date) — ตรวจได้จากเดือน+วัน โดยไม่ต้องรู้ปี
  * วันหยุดเคลื่อนที่ (movable) — ประมาณจากปี พ.ศ. (ใช้สูตรจันทรคติแบบง่าย)
  * ไม่ต้องติดตั้ง package เพิ่ม — ใช้ pure Python + dict lookup

การใช้งาน::

    from thaieda.timeseries._thai_holidays import is_thai_holiday, holiday_name

    if is_thai_holiday("2025-12-05"):
        print(holiday_name("2025-12-05"))  # "วันพ่อ"
"""

from __future__ import annotations

from datetime import date, datetime

# ----------------------------------------------------------------------------
# วันหยุดคงที่ — (เดือน, วัน) → ชื่อวันหยุด
# ----------------------------------------------------------------------------
_FIXED_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 1): "วันขึ้นปีใหม่",
    (4, 6): "วันจักรีรัชกาล (รัชกาลที่ ๑)",
    (4, 13): "วันสงกรานต์ (Songkran)",
    (4, 14): "วันสงกรานต์ (Songkran)",
    (4, 15): "วันสงกรานต์ (Songkran)",
    (5, 1): "วันแรงงานแห่งชาติ",
    (5, 5): "วันฉัตรมงคล (รัชกาลที่ ๙)",
    (6, 3): "วันสมเด็จพระนางเจ้าฯ",
    (7, 28): "วันเฉลิมพระชนมพรรษา รัชกาลที่ ๑๐",
    (8, 12): "วันสมเด็จพระบรมราชชนนี",
    (10, 13): "วันนวมินทรมหาราช",
    (10, 23): "วันปิยมหาราช (รัชกาลที่ ๕)",
    (12, 5): "วันพ่อ (วันเฉลิมพระชนมพรรษา รัชกาลที่ ๙)",
    (12, 10): "วันรัฐธรรมนูญ",
    (12, 31): "วันสิ้นปี",
}

# วันหยุดเคลื่อนที่โดยประมาณ (ประกาศแต่ละปี — ค่าประมาณจากปี พ.ศ.)
# คีย์: (เดือนเริ่มต้น, ช่วงวันเริ่มต้น, ช่วงวันสิ้นสุด) → ชื่อ
_MOVABLE_HOLIDAY_RANGES: list[dict[str, any]] = [
    {"name": "วันมาฆบูชา", "month": 2, "day_start": 10, "day_end": 20},
    {"name": "วันจักรีรัชกาล (รัชกาลที่ ๖)", "month": 1, "day_start": 1, "day_end": 1},
    {"name": "วันวิสาขบูชา", "month": 5, "day_start": 20, "day_end": 31},
]


def is_thai_holiday(dt: str | date | datetime, include_weekends: bool = True) -> bool:
    """ตรวจว่าวันที่ระบุเป็นวันหยุดราชการหรือวันหยุดนักขัตฤกษ์ของไทยหรือไม่.

    Args:
        dt: วันที่ — string (ISO format เช่น "2025-12-05"), date, หรือ datetime.
        include_weekends: รวมวันหยุดสุดสัปดาห์ (วันเสาร์-อาทิตย์) หรือไม่ (ค่าเริ่มต้นคือ True).

    Returns:
        True ถ้าเป็นวันหยุด, False ถ้าไม่ใช่.
    """
    d = _to_date(dt)
    if d is None:
        return False

    # ตรวจวันหยุดคงที่
    if (d.month, d.day) in _FIXED_HOLIDAYS:
        return True

    # ตรวจวันหยุดเคลื่อนที่แบบประมาณ
    for holiday in _MOVABLE_HOLIDAY_RANGES:
        if d.month == holiday["month"] and holiday["day_start"] <= d.day <= holiday["day_end"]:
            return True

    # ตรวจวันเสาร์-อาทิตย์ (weekend)
    if include_weekends:
        return d.weekday() >= 5  # 5=เสาร์, 6=อาทิตย์
    return False


def holiday_name(dt: str | date | datetime) -> str:
    """คืนชื่อวันหยุดของวันที่ระบุ — ถ้าไม่ใช่วันหยุดคืนสตริงว่าง.

    Args:
        dt: วันที่ — string (ISO format), date, หรือ datetime.

    Returns:
        ชื่อวันหยุดภาษาไทย หรือ "" ถ้าไม่ใช่วันหยุด.
    """
    d = _to_date(dt)
    if d is None:
        return ""

    # วันหยุดคงที่
    name = _FIXED_HOLIDAYS.get((d.month, d.day), "")
    if name:
        return name

    # วันหยุดเคลื่อนที่แบบประมาณ
    for holiday in _MOVABLE_HOLIDAY_RANGES:
        if d.month == holiday["month"] and holiday["day_start"] <= d.day <= holiday["day_end"]:
            return holiday["name"]

    # วันเสาร์-อาทิตย์
    if d.weekday() == 5:
        return "วันเสาร์"
    if d.weekday() == 6:
        return "วันอาทิตย์"

    return ""


def flag_holiday_spikes(
    dates: list[str | date | datetime],
    anomalies: list[int],
) -> dict[int, str]:
    """ระบุ spike ที่ตรงกับวันหยุด — คืน dict {anomaly_index: holiday_name}.

    ใช้ร่วมกับ TimeseriesResult.anomalies เพื่ออธิบายว่า spike ที่ตำแหน่ง N
    เกิดในวันหยุดอะไร

    Args:
        dates: รายการวันที่ของ timeseries (เรียงตามลำดับเดียวกับ anomalies).
        anomalies: ตำแหน่ง (0-based) ของ spike/ค่าผิดปกติ.

    Returns:
        dict {anomaly_index: holiday_name} — เฉพาะ spike ที่ตรงกับวันหยุด.
    """
    result: dict[int, str] = {}
    for idx in anomalies:
        if 0 <= idx < len(dates):
            name = holiday_name(dates[idx])
            if name:
                result[idx] = name
    return result


def _to_date(dt: str | date | datetime) -> date | None:
    """แปลง input เป็น date — รองรับ str/date/datetime."""
    if isinstance(dt, date):
        return dt
    if isinstance(dt, datetime):
        return dt.date()
    if isinstance(dt, str):
        try:
            return datetime.fromisoformat(dt).date()
        except ValueError:
            return None
    return None


__all__ = ["is_thai_holiday", "holiday_name", "flag_holiday_spikes"]
