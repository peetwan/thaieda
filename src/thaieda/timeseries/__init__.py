"""Timeseries analysis — วิเคราะห์ข้อมูลอนุกรมเวลา (timeseries) แบบอัตโนมัติ.

โมดูลนี้ตรวจหาคอลัมน์วันที่/เวลา แล้ววิเคราะห์คอลัมน์ตัวเลขเทียบกับแกนเวลานั้น —
หาความถี่ (รายวัน/สัปดาห์/เดือน/ชั่วโมง), แนวโน้ม (trend), รูปแบบตามฤดูกาล (seasonality),
แยกองค์ประกอบ (decomposition), ค่าผิดปกติเฉพาะช่วง (spike), และช่องว่างข้อมูล (time gap)

ออกแบบให้ "ทำงานได้โดยไม่ต้องมี statsmodels" — การวิเคราะห์พื้นฐาน (trend/seasonality/gap)
ใช้ numpy ล้วน ส่วน STL decomposition ที่แม่นกว่าจะใช้ statsmodels เมื่อมี (thaieda[timeseries])
ตามหลักการของแพ็กเกจ: ไม่มี fallback แบบเงียบ — ถ้าผู้ใช้ "บังคับ" ใช้ statsmodels แต่ไม่ได้ติดตั้ง
จะ raise ImportError พร้อมคำสั่ง pip install ที่ชัดเจน
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thaieda.detect import ColumnType, detect_all, is_nonmeasure_numeric

# ----------------------------------------------------------------------------
# ค่าคงที่
# ----------------------------------------------------------------------------
# จำนวนจุดข้อมูลขั้นต่ำที่ทำให้การวิเคราะห์ timeseries มีความหมาย
_MIN_TS_POINTS = 5
# สัดส่วน "ช่วงเวลาไม่ซ้ำ / จำนวนแถว" ขั้นต่ำที่ยังถือว่าเป็นอนุกรมเวลาระดับแถว —
# ถ้าต่ำกว่านี้ แปลว่าหลายแถวใช้ timestamp เดียวกันมาก (ข้อมูล panel/snapshot
# เช่น ค่าตรวจวัดของหลายสถานี ณ เวลาเดียวกัน) ไม่ใช่อนุกรมเวลารายแถว จึงไม่ควรวิเคราะห์ TS
_MIN_UNIQUE_TS_RATIO = 0.5
# จำนวนจุดสูงสุดที่ยอมใช้ STL decomposition (statsmodels) ในโหมด auto — STL บนซีรีส์ยาวมาก
# (หลายแสนจุด) ช้ามาก (O(n) ต่อรอบ × หลายรอบ) เกินนี้ถอยไปใช้ decomposition พื้นฐานที่เป็น
# เวกเตอร์และเร็วกว่ามาก. โหมด engine="statsmodels" ที่ผู้ใช้ระบุเองยังบังคับใช้ STL ตามเดิม
_MAX_STL_POINTS = 200_000
# เกณฑ์สหสัมพันธ์ (|r| ระหว่างเวลา↔ค่า) ที่ถือว่า "มีแนวโน้ม"
_TREND_R_THRESHOLD = 0.30
# เกณฑ์ ACF ที่ถือว่า "มี seasonality" ที่ lag หนึ่ง
_SEASONAL_ACF_THRESHOLD = 0.30
# จำนวนเท่าของ std ที่ถือว่า residual เป็นค่าผิดปกติ (spike/level shift)
_ANOMALY_SIGMA = 3.0
# lag ที่นิยมตรวจ seasonality (รอบสัปดาห์ 7, ปีรายเดือน 12, รอบวันรายชั่วโมง 24, รอบเดือน 30, รอบปีรายสัปดาห์ 52)
_CANDIDATE_LAGS = (7, 12, 24, 30, 52, 365)
# จำนวน index ค่าผิดปกติสูงสุดที่เก็บต่อหนึ่งคอลัมน์
_MAX_ANOMALY_INDICES = 50
# ตัวคูณ median delta ที่ถือว่าเป็น "ช่องว่าง" (gap) ของเวลา
_GAP_FACTOR = 1.5
# ความถี่ -> คาบ seasonality ตามธรรมชาติ (จำนวนจุดต่อรอบ)
_FREQ_NATURAL_PERIOD: dict[str, int] = {"H": 24, "D": 7, "W": 52, "M": 12}
# ความถี่ -> คำอธิบายภาษาไทย
_FREQ_TH: dict[str, str] = {
    "H": "รายชั่วโมง",
    "D": "รายวัน",
    "W": "รายสัปดาห์",
    "M": "รายเดือน",
    "irregular": "ไม่สม่ำเสมอ",
}
# ทิศทางแนวโน้ม -> ภาษาไทย
_TREND_TH: dict[str, str] = {
    "increasing": "เพิ่มขึ้น",
    "decreasing": "ลดลง",
    "stable": "คงที่",
}


# ----------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ----------------------------------------------------------------------------
@dataclass
class TimeseriesComponent:
    """องค์ประกอบหนึ่งของ timeseries (trend/seasonal/residual) สำหรับนำไปวาดกราฟ."""

    component: str  # "trend" | "seasonal" | "residual"
    values: list[float]
    description_th: str

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "values": self.values,
            "description_th": self.description_th,
        }


@dataclass
class TimeseriesResult:
    """สรุปการวิเคราะห์ timeseries ของคอลัมน์ตัวเลขหนึ่งคอลัมน์เทียบกับแกนเวลา."""

    column: str
    is_timeseries: bool
    frequency: str  # "D" | "W" | "M" | "H" | "irregular"
    frequency_th: str
    has_trend: bool
    trend_direction: str  # "increasing" | "decreasing" | "stable"
    trend_direction_th: str
    has_seasonality: bool
    seasonal_period: int  # จำนวนจุดต่อรอบ (เช่น 7 = รอบสัปดาห์, 12 = รอบปีรายเดือน)
    seasonal_period_th: str
    components: dict[str, TimeseriesComponent] = field(default_factory=dict)
    anomalies: list[int] = field(default_factory=list)  # ตำแหน่ง (0-based) ของ spike/ค่าผิดปกติ
    gaps: list[tuple[str, str]] = field(default_factory=list)  # ช่องว่างเวลา (เริ่ม, จบ)
    gap_count: int = 0
    stats: dict[str, float] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)  # ข้อค้นพบเป็นภาษาไทย
    engine_used: str = "basic"  # "statsmodels" | "basic"

    def to_dict(self) -> dict:
        """ส่งออกแบบกระชับ — ไม่รวมอาเรย์ของ components (ใช้สำหรับวาดกราฟในหน่วยความจำ)."""
        return {
            "column": self.column,
            "is_timeseries": self.is_timeseries,
            "frequency": self.frequency,
            "frequency_th": self.frequency_th,
            "has_trend": self.has_trend,
            "trend_direction": self.trend_direction,
            "trend_direction_th": self.trend_direction_th,
            "has_seasonality": self.has_seasonality,
            "seasonal_period": self.seasonal_period,
            "seasonal_period_th": self.seasonal_period_th,
            "anomaly_count": len(self.anomalies),
            "anomalies": self.anomalies,
            "gap_count": self.gap_count,
            "gaps": [list(g) for g in self.gaps],
            "stats": self.stats,
            "insights": self.insights,
            "engine_used": self.engine_used,
        }


# ----------------------------------------------------------------------------
# helper — ความถี่ของเวลา
# ----------------------------------------------------------------------------
def _datetime_index(series: pd.Series) -> pd.DatetimeIndex | None:
    """คืน DatetimeIndex ของ series ถ้ามี (ไม่งั้น None — ใช้ตำแหน่งแทนเวลา)."""
    idx = series.index
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    return None


def _classify_frequency(median_seconds: float) -> str:
    """แมป median delta (วินาที) ของแกนเวลาเป็นรหัสความถี่ D/W/M/H/irregular."""
    day = 86400.0
    if 3000 <= median_seconds <= 5400:  # ~1 ชั่วโมง (50–90 นาที)
        return "H"
    if 0.8 * day <= median_seconds <= 1.2 * day:  # ~1 วัน
        return "D"
    if 6 * day <= median_seconds <= 8 * day:  # ~1 สัปดาห์
        return "W"
    if 27 * day <= median_seconds <= 32 * day:  # ~1 เดือน
        return "M"
    return "irregular"


def _detect_frequency(dt_index: pd.DatetimeIndex) -> tuple[str, float]:
    """ตรวจความถี่จาก DatetimeIndex — คืน (รหัสความถี่, median delta เป็นวินาที).

    median delta = 0.0 ถ้ามีจุดเดียวหรือเรียงเวลาไม่ได้
    """
    ordered = dt_index.sort_values()
    if len(ordered) < 2:
        return "irregular", 0.0
    # หาร timedelta64 ด้วย 1 วินาที -> float วินาที (ทนต่อ unit ns/us/ms ของ index)
    deltas = np.diff(ordered.values) / np.timedelta64(1, "s")
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return "irregular", 0.0
    median_seconds = float(np.median(deltas))
    return _classify_frequency(median_seconds), median_seconds


def _detect_time_gaps(dt_index: pd.DatetimeIndex, median_seconds: float) -> list[tuple[str, str]]:
    """หาช่องว่างของเวลา (ช่วงที่ delta ใหญ่ผิดปกติ) — คืนรายการ (เริ่ม, จบ) เป็นสตริง.

    ใช้วิธีเทียบ delta กับ median delta (ไม่พึ่ง freq string ของ pandas เพื่อเลี่ยงปัญหาเวอร์ชัน)
    """
    if median_seconds <= 0:
        return []
    ordered = dt_index.sort_values().unique()
    if len(ordered) < 2:
        return []
    ordered = pd.DatetimeIndex(ordered)
    threshold = median_seconds * _GAP_FACTOR
    gaps: list[tuple[str, str]] = []
    secs = np.diff(ordered.values) / np.timedelta64(1, "s")
    for i, gap_seconds in enumerate(secs):
        if gap_seconds > threshold:
            gaps.append((str(ordered[i]), str(ordered[i + 1])))
    return gaps


# ----------------------------------------------------------------------------
# helper — แนวโน้ม (trend) และ seasonality
# ----------------------------------------------------------------------------
def _linear_trend(values: np.ndarray) -> tuple[float, float, float]:
    """ปรับเส้นตรงให้กับค่า (เทียบตำแหน่ง 0..n-1) — คืน (slope, intercept, r).

    r = สหสัมพันธ์ระหว่างตำแหน่งเวลากับค่า (ใช้ตัดสินว่ามีแนวโน้มชัดหรือไม่)
    """
    n = values.size
    x = np.arange(n, dtype="float64")
    slope, intercept = np.polyfit(x, values, 1)
    if np.std(values) == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(x, values)[0, 1])
        if math.isnan(r):
            r = 0.0
    return float(slope), float(intercept), r


def _acf_at(values: np.ndarray, lag: int) -> float:
    """คำนวณ autocorrelation ที่ lag หนึ่ง (numpy ล้วน) — คืน 0 ถ้าคำนวณไม่ได้."""
    if lag <= 0 or lag >= values.size:
        return 0.0
    x = values - values.mean()
    denom = float(np.sum(x * x))
    if denom == 0:
        return 0.0
    return float(np.sum(x[lag:] * x[:-lag]) / denom)


def _detect_seasonality(
    detrended: np.ndarray, freq: str, max_period: int
) -> tuple[bool, int, float]:
    """หา seasonality จาก ACF ของค่าที่ตัดแนวโน้มแล้ว — คืน (มี/ไม่มี, คาบที่ดีที่สุด, ACF).

    ลอง lag ตามรายการ _CANDIDATE_LAGS โดยให้คาบตามธรรมชาติของความถี่มาก่อน
    เลือกเฉพาะ lag ที่มีข้อมูลครบอย่างน้อย 2 รอบ (n >= 2*lag) และไม่เกิน max_period
    """
    n = detrended.size
    natural = _FREQ_NATURAL_PERIOD.get(freq)
    candidates: list[int] = []
    if natural is not None:
        candidates.append(natural)
    candidates.extend(lag for lag in _CANDIDATE_LAGS if lag not in candidates)

    best_lag = 0
    best_acf = 0.0
    for lag in candidates:
        if lag > max_period or 2 * lag > n:
            continue
        acf = _acf_at(detrended, lag)
        if acf > best_acf:
            best_acf = acf
            best_lag = lag

    has_seasonality = best_acf >= _SEASONAL_ACF_THRESHOLD and best_lag > 0
    return has_seasonality, (best_lag if has_seasonality else 0), best_acf


# ----------------------------------------------------------------------------
# helper — decomposition (STL หรือ moving-average พื้นฐาน)
# ----------------------------------------------------------------------------
def _statsmodels_available() -> bool:
    """True ถ้าติดตั้ง statsmodels (thaieda[timeseries])."""
    import importlib.util

    return importlib.util.find_spec("statsmodels") is not None


def _stl_decompose(values: np.ndarray, period: int) -> dict[str, np.ndarray]:
    """แยกองค์ประกอบด้วย statsmodels STL — คืน dict trend/seasonal/residual.

    Raises:
        ImportError: ถ้าไม่ได้ติดตั้ง statsmodels (แนะนำ pip install thaieda[timeseries]).
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError as exc:  # pragma: no cover - ขึ้นกับสภาพแวดล้อม
        raise ImportError(
            "STL decomposition requires pip install thaieda[timeseries] "
            "(the 'statsmodels' package)."
        ) from exc

    period = max(2, int(period))
    result = STL(values, period=period, robust=True).fit()
    return {
        "trend": np.asarray(result.trend, dtype="float64"),
        "seasonal": np.asarray(result.seasonal, dtype="float64"),
        "residual": np.asarray(result.resid, dtype="float64"),
    }


def _basic_decompose(values: np.ndarray, period: int) -> dict[str, np.ndarray]:
    """แยกองค์ประกอบแบบพื้นฐาน (ไม่พึ่ง statsmodels) — moving average + เฉลี่ยตามเฟส.

    trend = ค่าเฉลี่ยเคลื่อนที่ (window = คาบ หรือหน้าต่างเล็กถ้าไม่มีคาบ)
    seasonal = ค่าเฉลี่ยของ (ค่า - trend) จัดกลุ่มตามตำแหน่งในรอบ (index % period)
    residual = ค่า - trend - seasonal
    """
    n = values.size
    s = pd.Series(values)

    window = period if period >= 2 else min(7, n)
    window = max(2, min(window, n))
    trend = s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    detrended = values - trend
    seasonal = np.zeros(n, dtype="float64")
    if period >= 2:
        phase = np.arange(n) % period
        for p in range(period):
            mask = phase == p
            if mask.any():
                seasonal[mask] = np.nanmean(detrended[mask])
        # ทำให้ค่าเฉลี่ยของ seasonal ≈ 0 (ย้ายส่วนเฉลี่ยไปรวมกับ trend)
        seasonal -= np.nanmean(seasonal)

    residual = values - trend - seasonal
    return {
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
    }


def _clean_floats(arr: np.ndarray) -> list[float]:
    """แปลงอาเรย์เป็น list[float] โดยแทน NaN/inf ด้วย 0.0 (ปลอดภัยต่อ JSON/กราฟ)."""
    out = np.asarray(arr, dtype="float64")
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return [float(v) for v in out]


def _residual_anomalies(residual: np.ndarray) -> list[int]:
    """หา index ของค่าผิดปกติจาก residual — |resid - mean| > sigma*std."""
    resid = np.asarray(residual, dtype="float64")
    finite = resid[np.isfinite(resid)]
    if finite.size < _MIN_TS_POINTS:
        return []
    std = float(np.std(finite))
    if std == 0:
        return []
    mean = float(np.mean(finite))
    threshold = _ANOMALY_SIGMA * std
    idx = [int(i) for i, v in enumerate(resid) if np.isfinite(v) and abs(v - mean) > threshold]
    return idx[:_MAX_ANOMALY_INDICES]


# ----------------------------------------------------------------------------
# การสร้าง insight ภาษาไทย
# ----------------------------------------------------------------------------
def _build_ts_insights(
    has_trend: bool,
    trend_direction: str,
    slope: float,
    has_seasonality: bool,
    seasonal_period: int,
    freq: str,
    n_anomalies: int,
    gap_count: int,
) -> list[str]:
    """สร้างข้อความ insight ภาษาไทยจากผลการวิเคราะห์ timeseries."""
    insights: list[str] = []

    if has_trend and trend_direction != "stable":
        insights.append(
            f"พบแนวโน้ม{_TREND_TH[trend_direction]} (ค่าเปลี่ยนเฉลี่ย {slope:+.4g} ต่อช่วงเวลา)"
        )
    if has_seasonality and seasonal_period > 0:
        insights.append(f"พบรูปแบบตามฤดูกาล (seasonality) รอบ {seasonal_period} จุด")
    if gap_count > 0:
        insights.append(f"พบช่องว่างข้อมูล {gap_count} ช่วง (เวลาที่ขาดหาย)")
    if n_anomalies > 0:
        insights.append(f"พบค่าผิดปกติเฉพาะช่วง (spike) {n_anomalies} จุด")
    if not has_trend and not has_seasonality:
        insights.append("ไม่พบแนวโน้มหรือ seasonality ชัดเจน — อาจเป็น random walk หรือข้อมูลนิ่ง")

    return insights


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def detect_timeseries_columns(df: pd.DataFrame) -> dict[str, ColumnType]:
    """ตรวจหาคอลัมน์ที่ใช้เป็นแกนเวลาได้ (datetime ที่มีจุดข้อมูลพอ).

    คืน mapping ชื่อคอลัมน์ -> ColumnType.DATETIME เฉพาะคอลัมน์ที่ parse เป็นวันที่ได้
    และมีค่าไม่ว่างอย่างน้อย _MIN_TS_POINTS ค่า (พอจะวิเคราะห์เป็น timeseries)
    """
    types = detect_all(df)
    out: dict[str, ColumnType] = {}
    for col, ctype in types.items():
        if ctype != ColumnType.DATETIME:
            continue
        # format="mixed": parse แต่ละค่าแยกกัน เลี่ยง UserWarning "Could not infer format" (U1)
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if int(parsed.notna().sum()) >= _MIN_TS_POINTS:
            out[col] = ColumnType.DATETIME
    return out


def is_panel_time_axis(timestamps: pd.Series | pd.DatetimeIndex) -> bool:
    """แกนเวลานี้ดูเป็นข้อมูล panel/snapshot (หลายแถวใช้ timestamp เดียวกัน) หรือไม่.

    อนุกรมเวลาระดับแถวที่แท้จริง timestamp ควร "ไม่ซ้ำ" เป็นส่วนใหญ่ (เรียงตามเวลา 1 แถว/จุด)
    ถ้าจำนวน timestamp ที่ไม่ซ้ำมีน้อยเมื่อเทียบกับจำนวนแถว (เช่น ค่าตรวจวัดของหลายสถานี
    ณ เวลาเดียวกัน) แปลว่าเป็นข้อมูล panel/ภาพรวม ณ ช่วงเวลาเดียว ไม่ควรวิเคราะห์เป็น TS รายแถว
    เพราะจะได้ trend/spike ปลอมจากการเรียงแถวที่ timestamp เท่ากัน
    """
    idx = pd.Index(timestamps)
    n = int(idx.notna().sum())
    if n == 0:
        return False
    n_unique = int(idx.dropna().nunique())
    if n_unique < _MIN_TS_POINTS:
        return True
    return (n_unique / n) < _MIN_UNIQUE_TS_RATIO


def analyze_timeseries(
    series: pd.Series,
    freq: str = "auto",
    max_period: int = 365,
    engine: str = "auto",
) -> TimeseriesResult:
    """วิเคราะห์ timeseries ของคอลัมน์ตัวเลขหนึ่งคอลัมน์.

    ครอบคลุม: ความถี่, แนวโน้ม (trend), seasonality, การแยกองค์ประกอบ (decomposition),
    ค่าผิดปกติเฉพาะช่วง (spike), ช่องว่างของเวลา (gap) และ autocorrelation

    Args:
        series: คอลัมน์ตัวเลข — ถ้า index เป็น DatetimeIndex จะใช้ตรวจความถี่/ช่องว่าง
            ถ้าไม่ใช่ จะถือว่าจุดเรียงห่างเท่ากัน (วิเคราะห์ trend/seasonality ตามตำแหน่ง)
        freq: "auto" (ตรวจจากข้อมูล) หรือระบุรหัสความถี่ "D"/"W"/"M"/"H" เอง
        max_period: คาบ seasonality สูงสุดที่ยอมตรวจ (กันการจับคาบที่ยาวเกินไป)
        engine: "auto" (ใช้ statsmodels STL ถ้ามี ไม่งั้นพื้นฐาน), "statsmodels" (บังคับใช้ STL —
            raise ImportError ถ้าไม่ได้ติดตั้ง), หรือ "basic" (ใช้ moving-average พื้นฐานเสมอ)

    Returns:
        TimeseriesResult พร้อมองค์ประกอบ trend/seasonal/residual และ insight ภาษาไทย

    Raises:
        ImportError: ถ้า engine="statsmodels" แต่ไม่ได้ติดตั้ง statsmodels.
    """
    if engine not in ("auto", "statsmodels", "basic"):
        raise ValueError(f"engine {engine!r} ไม่ถูกต้อง — ใช้ 'auto', 'statsmodels' หรือ 'basic'")

    col_name = str(series.name) if series.name is not None else ""

    # --- เตรียมค่าตัวเลข + แกนเวลา (เรียงตามเวลาถ้ามี DatetimeIndex) ---
    dt_index = _datetime_index(series)
    work = series
    if dt_index is not None:
        order = dt_index.argsort()
        work = series.iloc[order]
        dt_index = dt_index[order]

    numeric = pd.to_numeric(work, errors="coerce")
    mask = numeric.notna().to_numpy()
    values = numeric.to_numpy(dtype="float64")[mask]
    if dt_index is not None:
        dt_index = dt_index[mask]

    n = values.size

    # --- ความถี่ + ช่องว่างเวลา ---
    median_seconds = 0.0
    if dt_index is not None:
        detected_freq, median_seconds = _detect_frequency(dt_index)
    else:
        detected_freq = "irregular"
    used_freq = freq if freq != "auto" else detected_freq
    gaps = _detect_time_gaps(dt_index, median_seconds) if dt_index is not None else []

    # --- ข้อมูลน้อยเกินไป — คืนผลแบบจำกัด ---
    if n < _MIN_TS_POINTS:
        return TimeseriesResult(
            column=col_name,
            is_timeseries=False,
            frequency=used_freq,
            frequency_th=_FREQ_TH.get(used_freq, used_freq),
            has_trend=False,
            trend_direction="stable",
            trend_direction_th=_TREND_TH["stable"],
            has_seasonality=False,
            seasonal_period=0,
            seasonal_period_th="",
            gaps=gaps,
            gap_count=len(gaps),
            stats={},
            insights=["ข้อมูลน้อยเกินไปสำหรับการวิเคราะห์ timeseries"],
            engine_used="basic",
        )

    # --- แนวโน้ม (trend) ---
    slope, intercept, r = _linear_trend(values)
    has_trend = abs(r) >= _TREND_R_THRESHOLD
    if not has_trend:
        trend_direction = "stable"
    elif slope > 0:
        trend_direction = "increasing"
    else:
        trend_direction = "decreasing"

    # --- seasonality (จาก ACF ของค่าที่ตัดแนวโน้มแล้ว) ---
    x = np.arange(n, dtype="float64")
    detrended = values - (slope * x + intercept)
    has_seasonality, seasonal_period, _best_acf = _detect_seasonality(
        detrended, used_freq, max_period
    )

    # --- decomposition (STL หรือพื้นฐาน) ---
    # คาบสำหรับแยกองค์ประกอบ: ใช้คาบ seasonality ที่ตรวจพบ ไม่งั้นใช้คาบตามธรรมชาติของความถี่
    period_for_decompose = (
        seasonal_period if seasonal_period >= 2 else _FREQ_NATURAL_PERIOD.get(used_freq, 0)
    )

    def _stl_period() -> int:
        """เลือกคาบที่ STL ใช้งานได้จริง (>=2 และข้อมูลครบ 2 รอบ) — มีคาบสำรองเมื่อไม่พบ seasonality."""
        p = period_for_decompose
        if p < 2 or n < 2 * p:
            p = max(2, min(n // 2, 7))
        return p

    engine_used = "basic"
    parts: dict[str, np.ndarray]
    if engine == "statsmodels":
        # ผู้ใช้บังคับ STL — เรียกเสมอ (raise ImportError ชัดเจนถ้าไม่ได้ติดตั้ง statsmodels)
        parts = _stl_decompose(values, _stl_period())
        engine_used = "statsmodels"
    elif (
        engine == "auto"
        and _statsmodels_available()
        and period_for_decompose >= 2
        and n >= 2 * period_for_decompose
        and n <= _MAX_STL_POINTS
    ):
        parts = _stl_decompose(values, period_for_decompose)
        engine_used = "statsmodels"
    else:
        parts = _basic_decompose(values, period_for_decompose)

    components = {
        "trend": TimeseriesComponent("trend", _clean_floats(parts["trend"]), "แนวโน้มระยะยาว"),
        "seasonal": TimeseriesComponent(
            "seasonal", _clean_floats(parts["seasonal"]), "รูปแบบตามฤดูกาล (ซ้ำเป็นรอบ)"
        ),
        "residual": TimeseriesComponent(
            "residual", _clean_floats(parts["residual"]), "ส่วนที่เหลือ (random/noise)"
        ),
    }

    # --- ค่าผิดปกติเฉพาะช่วง (spike) จาก residual ---
    anomalies = _residual_anomalies(parts["residual"])

    # --- สถิติ ---
    autocorr_lag1 = _acf_at(values, 1)
    stats = {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
        "autocorr_lag1": round(autocorr_lag1, 4),
    }

    insights = _build_ts_insights(
        has_trend,
        trend_direction,
        slope,
        has_seasonality,
        seasonal_period,
        used_freq,
        len(anomalies),
        len(gaps),
    )

    return TimeseriesResult(
        column=col_name,
        is_timeseries=True,
        frequency=used_freq,
        frequency_th=_FREQ_TH.get(used_freq, used_freq),
        has_trend=has_trend,
        trend_direction=trend_direction,
        trend_direction_th=_TREND_TH[trend_direction],
        has_seasonality=has_seasonality,
        seasonal_period=seasonal_period,
        seasonal_period_th=(f"รอบ {seasonal_period} จุด" if seasonal_period > 0 else ""),
        components=components,
        anomalies=anomalies,
        gaps=gaps,
        gap_count=len(gaps),
        stats=stats,
        insights=insights,
        engine_used=engine_used,
    )


def analyze_dataframe_timeseries(
    df: pd.DataFrame,
    time_col: str = "auto",
    max_columns: int = 20,
    engine: str = "auto",
) -> dict[str, TimeseriesResult]:
    """หา datetime column แล้ววิเคราะห์ทุกคอลัมน์ตัวเลขเทียบกับแกนเวลานั้น — แบบอัตโนมัติ.

    Args:
        df: ข้อมูลที่จะวิเคราะห์.
        time_col: "auto" (เลือก datetime column แรกที่เหมาะ) หรือชื่อคอลัมน์เวลาที่ต้องการ.
        max_columns: จำนวนคอลัมน์ตัวเลขสูงสุดที่จะวิเคราะห์ (กันช้าบนตารางกว้าง).
        engine: ส่งต่อให้ analyze_timeseries ("auto"/"statsmodels"/"basic").

    Returns:
        dict {ชื่อคอลัมน์ตัวเลข: TimeseriesResult} — ว่างถ้าไม่มีแกนเวลาหรือไม่มีคอลัมน์ตัวเลข
    """
    # --- เลือกคอลัมน์เวลา ---
    if time_col == "auto":
        ts_cols = detect_timeseries_columns(df)
        if not ts_cols:
            return {}
        chosen = next(iter(ts_cols))
    else:
        if time_col not in df.columns:
            raise KeyError(f"ไม่พบคอลัมน์เวลา {time_col!r} ในข้อมูล")
        chosen = time_col

    # --- สร้าง DataFrame ที่มี DatetimeIndex (เรียงตามเวลา) ---
    # format="mixed": parse แต่ละค่าแยกกัน เลี่ยง UserWarning "Could not infer format" (U1)
    time_values = pd.to_datetime(df[chosen], errors="coerce", format="mixed")
    valid = time_values.notna()
    if int(valid.sum()) < _MIN_TS_POINTS:
        return {}
    # ข้อมูล panel/snapshot (หลายแถวใช้ timestamp เดียวกัน) ไม่ใช่อนุกรมเวลารายแถว → ไม่วิเคราะห์
    if is_panel_time_axis(time_values[valid]):
        return {}
    indexed = df.loc[valid].copy()
    indexed.index = pd.DatetimeIndex(time_values[valid])
    indexed = indexed.sort_index()

    # --- เลือกคอลัมน์ตัวเลข (ไม่รวมคอลัมน์เวลา) ---
    types = detect_all(df)
    # ข้าม identifier/รหัส/พิกัด (id, *_id, lat/long, zip) — ไม่ใช่อนุกรมเวลา
    # กัน false positive เช่น "zip_code มีแนวโน้มเพิ่มตามเวลา" / spike บน lat/long
    numeric_cols = [
        str(c)
        for c in df.columns
        if str(c) != str(chosen)
        and types.get(str(c)) == ColumnType.NUMERIC
        and not is_nonmeasure_numeric(df[c], types.get(str(c)))
    ]

    results: dict[str, TimeseriesResult] = {}
    for col in numeric_cols[:max_columns]:
        result = analyze_timeseries(indexed[col], engine=engine)
        if result.is_timeseries:
            results[col] = result
    return results


__all__ = [
    "TimeseriesComponent",
    "TimeseriesResult",
    "detect_timeseries_columns",
    "is_panel_time_axis",
    "analyze_timeseries",
    "analyze_dataframe_timeseries",
]
