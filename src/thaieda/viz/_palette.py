"""Color palette สำหรับ ThaiEDA — colorblind-safe (v1.1).

ใช้ชุดสี Okabe-Ito (ออกแบบสำหรับคนตาบอดสี) ปรับให้เข้ากับ EDA
ใช้ได้ทั้งใน matplotlib และ Plotly

การใช้งาน::

    from thaieda.viz._palette import PALETTE, get_color, get_cmap

    color = get_color(0)        # สีแรกของ palette
    colors = get_cmap(5)        # 5 สีแรกของ palette
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Okabe-Ito palette — colorblind-safe (เหมาะกับ EDA ทุกประเภท)
# อ้างอิง: Okabe & Ito (2008) "Color Universal Design"
# ----------------------------------------------------------------------------
PALETTE: list[str] = [
    "#0072B2",  # น้ำเงิน — ค่าหลัก/ปกติ
    "#E69F00",  # ส้ม — เตือน/highlight
    "#009E73",  # เขียว — ผ่าน/ดี
    "#CC79A7",  # ชมพู — หมวดหมู่
    "#56B4E9",  # ฟ้าอ่อน — ข้อมูลทุติยภูมิ
    "#D55E00",  # แดงเข้ม — วิกฤต/error
    "#F0E442",  # เหลือง — เน้นพิเศษ
]

# สีหลักสำหรับ theme (dark/light)
ACCENT = PALETTE[0]  # น้ำเงิน
WARNING = PALETTE[1]  # ส้ม
CRITICAL = PALETTE[5]  # แดงเข้ม
OK = PALETTE[2]  # เขียว

# สีพื้นหลัง + ตัวอักษร (dark theme)
BG_DARK = "#1a1a2e"
BG_LIGHT = "#ffffff"
TEXT_DARK = "#e0e0e0"
TEXT_LIGHT = "#333333"
GRID_COLOR = "#333355"

# CSS font stack สำหรับ Plotly (browser จัดการ font เอง)
PLOTLY_FONT_FAMILY = "'Sarabun', 'Noto Sans Thai', 'Tahoma', sans-serif"

# Google Fonts import สำหรับ Plotly HTML
PLOTLY_FONT_CSS = (
    "@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap');"
)


# ----------------------------------------------------------------------------
# helper
# ----------------------------------------------------------------------------
def get_color(index: int) -> str:
    """คืนสีจาก palette ตาม index (วนรอบถ้าเกิน)."""
    return PALETTE[index % len(PALETTE)]


def get_cmap(n: int) -> list[str]:
    """คืน n สีจาก palette (วนรอบถ้า n > len(PALETTE))."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def apply_matplotlib_style(dark: bool = True) -> None:
    """ตั้งค่า matplotlib style ให้ใช้ palette นี้ + dark/light theme.

    Args:
        dark: True = dark theme (default), False = light theme.
    """
    bg = BG_DARK if dark else BG_LIGHT
    text = TEXT_DARK if dark else TEXT_LIGHT
    plt.rcParams.update(
        {
            "figure.facecolor": bg,
            "axes.facecolor": bg,
            "axes.edgecolor": text,
            "axes.labelcolor": text,
            "xtick.color": text,
            "ytick.color": text,
            "text.color": text,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.3,
            "axes.grid": True,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )


def plotly_layout_template(dark: bool = True) -> dict:
    """คืน layout template dict สำหรับ Plotly.

    ใช้กับ ``fig.update_layout(**plotly_layout_template())``

    Args:
        dark: True = dark theme (default), False = light theme.
    """
    bg = BG_DARK if dark else BG_LIGHT
    text = TEXT_DARK if dark else TEXT_LIGHT
    return {
        "font": {"family": PLOTLY_FONT_FAMILY, "color": text, "size": 13},
        "paper_bgcolor": bg,
        "plot_bgcolor": bg,
        "colorway": PALETTE,
        "margin": {"l": 60, "r": 30, "t": 40, "b": 50},
        "xaxis": {"gridcolor": GRID_COLOR, "zerolinecolor": GRID_COLOR},
        "yaxis": {"gridcolor": GRID_COLOR, "zerolinecolor": GRID_COLOR},
    }


__all__ = [
    "PALETTE",
    "ACCENT",
    "WARNING",
    "CRITICAL",
    "OK",
    "PLOTLY_FONT_FAMILY",
    "PLOTLY_FONT_CSS",
    "get_color",
    "get_cmap",
    "apply_matplotlib_style",
    "plotly_layout_template",
]
