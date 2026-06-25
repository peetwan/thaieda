"""ชนิดกราฟเพิ่มเติม (matplotlib) — pair plot, KDE, QQ-plot, sunburst.

ทุกฟังก์ชันคืนค่าเป็นสตริง base64 PNG เพื่อฝังใน HTML report.
โมดูลนี้ใช้ matplotlib ล้วน (ไม่ต้องติดตั้ง dependency เพิ่ม) ยกเว้น sunburst
ที่จะลองใช้ plotly (lazy) แต่ถ้าไม่มี plotly จะใช้ matplotlib pie chart แทน
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thaieda.viz._palette import PALETTE, apply_matplotlib_style, get_cmap

__all__ = [
    "create_pair_plot",
    "create_kde_plot",
    "create_qq_plot",
    "create_sunburst_chart",
]

# จำกัดขนาดกราฟ
_MAX_PAIR_COLS = 5
_MAX_PAIR_ROWS = 300
_MAX_SUNBURST_SLICES = 20

# ลายเซ็นไฟล์ PNG สำหรับตรวจสอบผลลัพธ์
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """แปลง matplotlib figure -> base64 PNG string แล้วปิด figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _numeric_frame(df: pd.DataFrame, max_cols: int) -> pd.DataFrame:
    """คืนเฉพาะคอลัมน์ตัวเลข (จำกัดจำนวน)."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > max_cols:
        numeric = numeric.iloc[:, :max_cols]
    return numeric


def create_pair_plot(
    df: pd.DataFrame,
    hue_col: str | None = None,
) -> str:
    """Pair plot (scatter matrix + histogram แนวทแยง) แบบ matplotlib.

    จำกัดที่ 5 คอลัมน์ตัวเลขแรก และสุ่มตัวอย่าง 300 แถวเพื่อความเร็ว.
    ถ้ามี ``hue_col`` จะระบายสีตามหมวดหมู่ของคอลัมน์นั้น (ใช้ palette Okabe-Ito).

    Args:
        df: DataFrame ต้นทาง.
        hue_col: ชื่อคอลัมน์ที่จะใช้ระบายสี (optional — ปกติเป็นหมวดหมู่).

    Returns:
        สตริง base64 PNG หรือสตริงว่างถ้ามีคอลัมน์ตัวเลขน้อยกว่า 2 คอลัมน์.
    """
    apply_matplotlib_style(dark=True)

    numeric = _numeric_frame(df, _MAX_PAIR_COLS)
    if numeric.shape[1] < 2:
        return ""

    data = numeric.copy()
    if hue_col is not None and hue_col in df.columns:
        data[hue_col] = df[hue_col]
    data = data.dropna()
    if len(data) < 2:
        return ""
    if len(data) > _MAX_PAIR_ROWS:
        data = data.sample(_MAX_PAIR_ROWS, random_state=42)

    cols = [c for c in numeric.columns if c in data.columns]
    n = len(cols)
    labels = [str(c) for c in cols]

    if hue_col is not None and hue_col in data.columns:
        cats = data[hue_col].dropna().astype(str).unique().tolist()
        colors = get_cmap(len(cats))
        cat_color = {c: colors[i % len(colors)] for i, c in enumerate(cats)}
    else:
        cats = None
        cat_color = {}

    fig, axes = plt.subplots(
        n, n, figsize=(2.4 * n + 1, 2.4 * n + 1), squeeze=False, facecolor="#1a1a2e"
    )
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ax.set_facecolor("#1a1a2e")
            if i == j:
                vals = data[cols[i]].to_numpy(dtype="float64")
                ax.hist(vals, bins=15, color=PALETTE[0], alpha=0.85, edgecolor="#1a1a2e")
            else:
                if cats is not None:
                    for cat in cats:
                        subset = data[data[hue_col].astype(str) == cat]
                        ax.scatter(
                            subset[cols[j]].to_numpy(dtype="float64"),
                            subset[cols[i]].to_numpy(dtype="float64"),
                            s=8,
                            color=cat_color[cat],
                            alpha=0.6,
                            edgecolors="none",
                            label=str(cat),
                        )
                    if j == 0 and i == 1:
                        ax.legend(
                            fontsize=6,
                            loc="upper left",
                            framealpha=0.5,
                            facecolor="#1a1a2e",
                            edgecolor="#333355",
                        )
                else:
                    ax.scatter(
                        data[cols[j]].to_numpy(dtype="float64"),
                        data[cols[i]].to_numpy(dtype="float64"),
                        s=8,
                        color=PALETTE[0],
                        alpha=0.6,
                        edgecolors="none",
                    )
            ax.tick_params(labelsize=6, colors="#e0e0e0")
            for spine in ax.spines.values():
                spine.set_color("#333355")
            if i == n - 1:
                ax.set_xlabel(labels[j], color="#e0e0e0", fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], color="#e0e0e0", fontsize=8)
    fig.suptitle("Pair plot", color="#e0e0e0", fontsize=11)
    fig.tight_layout()
    return _fig_to_base64(fig)


def create_kde_plot(df: pd.DataFrame, col: str) -> str:
    """กราฟ Kernel Density Estimation (KDE) ของคอลัมน์ตัวเลข.

    ใช้ Gaussian kernel แบบง่าย (ไม่พึ่ง scipy) ในการประมาณความหนาแน่น
    คืนสตริงว่างถ้าคอลัมน์ไม่ใช่ตัวเลขหรือมีค่าน้อยกว่า 2 ค่า.

    Args:
        df: DataFrame.
        col: ชื่อคอลัมน์ตัวเลข.

    Returns:
        สตริง base64 PNG หรือสตริงว่างถ้าข้อมูลไม่พอ.
    """
    apply_matplotlib_style(dark=True)

    if col not in df.columns:
        return ""
    values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype="float64")
    if values.size < 2:
        return ""

    # KDE แบบ Gaussian (silverman's rule สำหรับ bandwidth)
    n = values.size
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    # ค่าคงที่ (std == 0) ใช้ bandwidth ตายตัวเล็กน้อยเพื่อให้วาดได้
    h = 0.1 if std == 0 else 1.06 * std * (n ** (-1 / 5))  # Silverman

    x_grid = np.linspace(values.min() - 3 * h, values.max() + 3 * h, 200)
    density = np.zeros_like(x_grid)
    for v in values:
        density += np.exp(-0.5 * ((x_grid - v) / h) ** 2)
    density /= n * h * np.sqrt(2 * np.pi)

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.plot(x_grid, density, color=PALETTE[0], linewidth=1.8)
    ax.fill_between(x_grid, density, alpha=0.25, color=PALETTE[0])
    ax.set_title(f"KDE: {col}", color="#e0e0e0", fontsize=11)
    ax.set_xlabel(col, color="#e0e0e0")
    ax.set_ylabel("density", color="#e0e0e0")
    ax.tick_params(colors="#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _fig_to_base64(fig)


def create_qq_plot(df: pd.DataFrame, col: str) -> str:
    """Q-Q plot เปรียบเทียบการแจกแจงคอลัมน์กับ normal distribution.

    ถ้าข้อมูลเป็น normal จุดจะเรียงเป็นเส้นตรงตามเส้นอ้างอิง.
    คืนสตริงว่างถ้าคอลัมน์ไม่ใช่ตัวเลขหรือมีค่าน้อยกว่า 2 ค่า.

    Args:
        df: DataFrame.
        col: ชื่อคอลัมน์ตัวเลข.

    Returns:
        สตริง base64 PNG หรือสตริงว่างถ้าข้อมูลไม่พอ.
    """
    apply_matplotlib_style(dark=True)

    if col not in df.columns:
        return ""
    values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype="float64")
    if values.size < 2:
        return ""

    from statistics import NormalDist

    sorted_vals = np.sort(values)
    n = sorted_vals.size
    # Theoretical quantiles (standard normal) ด้วย stdlib ไม่พึ่ง scipy
    normal = NormalDist()
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.array([normal.inv_cdf(float(p)) for p in probs], dtype="float64")

    # matplotlib warning: edgecolors='none' อาจเตือนกับ marker บางชนิด — ใช้ marker default ได้ปลอดภัย
    # จุดข้อมูลยังคงเป็นสีจาก palette หลักเสมอ.

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.scatter(
        theoretical,
        sorted_vals,
        s=14,
        color=PALETTE[0],
        alpha=0.7,
        edgecolors="none",
    )
    # เส้นอ้างอิง y = mean + std * quantile
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if std > 0:
        ref_line = mean + std * theoretical
        ax.plot(
            theoretical,
            ref_line,
            color=PALETTE[5],
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
        )
    ax.set_title(f"Q-Q plot: {col}", color="#e0e0e0", fontsize=11)
    ax.set_xlabel("theoretical quantiles", color="#e0e0e0")
    ax.set_ylabel("sample quantiles", color="#e0e0e0")
    ax.tick_params(colors="#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _fig_to_base64(fig)


def create_sunburst_chart(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str | None = None,
) -> str:
    """Sunburst chart ของคอลัมน์หมวดหมู่ — คืน base64 PNG.

    ถ้ามี plotly ใช้ Plotly sunburst; ถ้าไม่มีจะ fallback เป็น matplotlib pie chart.
    คืนสตริงว่างถ้า cat_col ไม่มี หรือไม่มีค่าที่ไม่ว่าง.

    Args:
        df: DataFrame.
        cat_col: ชื่อคอลัมน์หมวดหมู่.
        val_col: ถ้าระบุจะใช้ค่าจากคอลัมน์นี้ (aggregate ด้วย sum)
                 ถ้าไม่ระบุจะใช้ค่า count (จำนวนที่พบ).

    Returns:
        สตริง base64 PNG หรือสตริงว่างถ้าไม่มีข้อมูล.
    """
    if cat_col not in df.columns:
        return ""

    if val_col is not None and val_col in df.columns:
        # aggregate ผลรวมตามหมวดหมู่ (coerce เป็นตัวเลขเพื่อกันคอลัมน์ object)
        measure = pd.to_numeric(df[val_col], errors="coerce")
        agg = measure.groupby(df[cat_col]).sum().dropna()
        values: list[float] = [float(v) for v in agg.to_numpy()]
        labels: list[str] = [str(k) for k in agg.index]
    else:
        vc = df[cat_col].dropna().astype(str).value_counts()
        values = [int(v) for v in vc.to_numpy()]
        labels = [str(k) for k in vc.index]

    if not values:
        return ""

    # จำกัดจำนวน slice
    if len(values) > _MAX_SUNBURST_SLICES:
        # เก็บ top N และรวมที่เหลือเป็น "อื่น ๆ"
        paired = sorted(zip(labels, values, strict=False), key=lambda x: x[1], reverse=True)
        top = paired[: _MAX_SUNBURST_SLICES - 1]
        other_sum = sum(v for _, v in paired[_MAX_SUNBURST_SLICES - 1 :])
        if other_sum > 0:
            top.append(("อื่น ๆ", other_sum))
        labels = [k for k, _ in top]
        values = [v for _, v in top]

    try:
        import plotly.graph_objects as go
    except ImportError:
        return _sunburst_pie_fallback(labels, values, cat_col)

    return _sunburst_plotly(labels, values, cat_col, go)


def _sunburst_plotly(
    labels: list[str],
    values: list[float],
    cat_col: str,
    go: Any,
) -> str:
    """สร้าง plotly sunburst + แปลงเป็น base64 PNG (ไม่เปิดหน้าต่าง)."""
    from thaieda.viz._palette import (
        BG_DARK,
        PLOTLY_FONT_FAMILY,
        TEXT_DARK,
    )

    fig = go.Figure(
        data=go.Sunburst(
            labels=labels,
            parents=[""] * len(labels),
            values=values,
            marker={"colors": [PALETTE[i % len(PALETTE)] for i in range(len(labels))]},
            hovertemplate="%{label}<br>value = %{value}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": f"Sunburst: {cat_col}", "x": 0.5},
        font={"family": PLOTLY_FONT_FAMILY, "color": TEXT_DARK, "size": 13},
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    # แปลงเป็น PNG bytes
    try:
        png_bytes = fig.to_image(format="png", width=600, height=500, scale=1)
    except (ImportError, ValueError, RuntimeError):
        # Plotly ต้องใช้ static image engine (kaleido) ในการ export PNG
        # ถ้า engine ไม่พร้อมให้ใช้ fallback ที่ระบุไว้ใน requirement แทน
        return _sunburst_pie_fallback(labels, values, cat_col)
    return base64.b64encode(png_bytes).decode("ascii")


def _sunburst_pie_fallback(labels: list[str], values: list[float], cat_col: str) -> str:
    """Fallback: pie chart ด้วย matplotlib ถ้า plotly ไม่พร้อมใช้."""
    apply_matplotlib_style(dark=True)
    colors = get_cmap(len(labels))
    explode = [0.02] * len(labels)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        explode=explode,
        textprops={"color": "#e0e0e0", "fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_color("#1a1a2e")
        autotext.set_fontsize(8)
    ax.set_title(f"Sunburst: {cat_col} (pie)", color="#e0e0e0", fontsize=11)
    return _fig_to_base64(fig)
