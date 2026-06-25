"""กราฟโต้ตอบได้ (interactive) ด้วย Plotly — คืนค่าเป็นสตริง HTML <div>.

แต่ละฟังก์ชันคืน HTML div ที่ฝังได้ใน Jinja2 report (เรียก ``fig.to_html`` ด้วย
``include_plotlyjs='cdn'`` เพื่อไม่ให้ report ใหญ่เกินไป)

การใช้งาน::

    from thaieda.viz._interactive import create_correlation_heatmap_interactive

    html_div = create_correlation_heatmap_interactive(df)

Plotly เป็น optional dependency — lazy import ทุกฟังก์ชัน และแจ้ง ImportError
ที่อ่านง่ายถ้าผู้ใช้ยังไม่ติดตั้ง plotly
"""

from __future__ import annotations

import pandas as pd

from thaieda.viz._palette import (
    PALETTE,
    PLOTLY_FONT_FAMILY,
    plotly_layout_template,
)

__all__ = [
    "create_correlation_heatmap_interactive",
    "create_distribution_interactive",
    "create_missing_matrix_interactive",
    "create_scatter_interactive",
]

# จำกัดขนาดกราฟเพื่อความเร็ว (เหมือนโมดูล matplotlib)
_MAX_NUMERIC_COLS = 30
_MISSING_MATRIX_MAX_ROWS = 1000


def _require_plotly():
    """Lazy import plotly พร้อมข้อความ ImportError ที่อ่านง่าย."""
    try:
        import plotly.express as px
        import plotly.figure_factory as ff
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "กราฟ interactive ของ thaieda ต้องใช้ plotly "
            "ติดตั้งด้วย: pip install plotly>=5.18.0 "
            "(หรือ pip install thaieda ซึ่งรวม plotly แล้วตั้งแต่ v1.1)"
        ) from exc
    return go, ff, px


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """คืนเฉพาะคอลัมน์ตัวเลข (จำกัดจำนวนคอลัมน์ให้กราฟอ่านได้)."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > _MAX_NUMERIC_COLS:
        numeric = numeric.iloc[:, :_MAX_NUMERIC_COLS]
    return numeric


def _wrap_labels(labels: list[str], limit: int = 18) -> list[str]:
    """ตัดชื่อคอลัมน์ที่ยาวเกินไป."""
    return [lbl if len(lbl) <= limit else lbl[: limit - 1] + "…" for lbl in labels]


def _layout_template() -> dict:
    """คืน Plotly layout ที่บังคับใช้ font family ของ ThaiEDA."""
    layout = plotly_layout_template()
    font = dict(layout.get("font", {}))
    font["family"] = PLOTLY_FONT_FAMILY
    layout["font"] = font
    return layout


def _fig_to_html(fig) -> str:
    """แปลง plotly figure เป็น HTML div (include_plotlyjs='cdn')."""
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_correlation_heatmap_interactive(df: pd.DataFrame) -> str:
    """แผนภาพความร้อนสหสัมพันธ์แบบ interactive ด้วย Plotly.

    คืนสตริง HTML <div> ที่ฝังใน report ได้.
    คืนสตริงว่างถ้ามีคอลัมน์ตัวเลขน้อยกว่า 2 คอลัมน์.

    Args:
        df: DataFrame ที่ต้องการคำนวณสหสัมพันธ์.

    Returns:
        สตริง HTML ของ plotly figure (div + script).

    Raises:
        ImportError: ถ้าไม่ได้ติดตั้ง plotly.
    """
    go, _ff, _px = _require_plotly()

    numeric = _numeric_frame(df)
    if numeric.shape[1] < 2:
        return ""

    corr = numeric.corr(numeric_only=True)
    labels = _wrap_labels([str(c) for c in corr.columns])
    z = corr.to_numpy(dtype="float64").tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{y} × %{x}<br>r = %{z:.4f}<extra></extra>",
            colorbar={"title": "r", "ticksuffix": ""},
        )
    )
    fig.update_layout(
        title={"text": "Correlation", "x": 0.5},
        **_layout_template(),
    )
    fig.update_xaxes(side="bottom")
    return _fig_to_html(fig)


def create_distribution_interactive(df: pd.DataFrame, col: str) -> str:
    """ฮิสโทแกรม + box marginal แบบ interactive ของคอลัมน์ตัวเลข.

    คืนสตริง HTML <div> ถ้าคอลัมน์ไม่ใช่ตัวเลขหรือไม่มีค่า จะคืนสตริงว่าง.

    Args:
        df: DataFrame.
        col: ชื่อคอลัมน์ตัวเลขที่ต้องการดูการแจกแจง.

    Returns:
        สตริง HTML ของ plotly figure.

    Raises:
        ImportError: ถ้าไม่ได้ติดตั้ง plotly.
    """
    _go, _ff, px = _require_plotly()

    if col not in df.columns:
        return ""
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return ""

    plot_df = pd.DataFrame({col: series.to_numpy(dtype="float64")})
    n_bins = min(40, max(10, int(series.nunique())))
    fig = px.histogram(
        plot_df,
        x=col,
        nbins=n_bins,
        marginal="box",
        color_discrete_sequence=[PALETTE[0]],
        title=f"Distribution: {col}",
    )
    fig.update_layout(**_layout_template())
    fig.update_traces(hovertemplate=f"{col} = %{{x}}<br>count = %{{y}}<extra></extra>")
    return _fig_to_html(fig)


def create_missing_matrix_interactive(df: pd.DataFrame) -> str:
    """เมทริกซ์ค่าว่างแบบ interactive (heatmap แบบ binary).

    มืด = ค่าว่าง, สว่าง = มีค่า. สุ่มตัวอย่างแถวถ้ามากกว่า 1000 แถว.

    Args:
        df: DataFrame.

    Returns:
        สตริง HTML ของ plotly figure หรือสตริงว่างถ้า df ว่าง.

    Raises:
        ImportError: ถ้าไม่ได้ติดตั้ง plotly.
    """
    go, _ff, _px = _require_plotly()

    if df.shape[0] == 0 or df.shape[1] == 0:
        return ""

    sample = df
    note = ""
    if len(df) > _MISSING_MATRIX_MAX_ROWS:
        sample = df.sample(_MISSING_MATRIX_MAX_ROWS, random_state=42).sort_index()
        note = f" (sampled {_MISSING_MATRIX_MAX_ROWS:,}/{len(df):,} rows)"

    labels = _wrap_labels([str(c) for c in sample.columns])
    present = sample.notna().to_numpy(dtype="float64")
    y_labels = [f"row {i}" for i in sample.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=present,
            x=labels,
            y=y_labels,
            colorscale=[[0.0, "#2b3038"], [1.0, PALETTE[0]]],
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            hovertemplate="row %{y}<br>col %{x}<br>%{z:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "Missing value matrix" + note, "x": 0.5},
        yaxis={"title": "rows", "autorange": "reversed", "showticklabels": False},
        **{k: v for k, v in _layout_template().items() if k != "yaxis"},
    )
    fig.update_xaxes(side="bottom", tickangle=-45)
    return _fig_to_html(fig)


def create_scatter_interactive(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str | None = None,
) -> str:
    """Scatter plot แบบ interactive ของสองคอลัมน์ตัวเลข.

    คืนสตริง HTML <div> ถ้าคอลัมน์ไม่ใช่ตัวเลขหรือไม่มีค่า จะคืนสตริงว่าง.

    Args:
        df: DataFrame.
        x: ชื่อคอลัมน์แกน x (ต้องเป็นตัวเลข).
        y: ชื่อคอลัมน์แกน y (ต้องเป็นตัวเลข).
        color_col: ชื่อคอลัมน์ที่จะใช้ระบายสีจุด (optional — รับได้ทั้งตัวเลข/หมวดหมู่).

    Returns:
        สตริง HTML ของ plotly figure.

    Raises:
        ImportError: ถ้าไม่ได้ติดตั้ง plotly.
    """
    _go, _ff, px = _require_plotly()

    if x not in df.columns or y not in df.columns:
        return ""
    sub = df[[x, y] + ([color_col] if color_col and color_col in df.columns else [])]
    sub = sub.dropna(subset=[x, y])
    if sub.empty:
        return ""

    sub = sub.copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna(subset=[x, y])
    if sub.empty:
        return ""

    color_kw: dict = {}
    if color_col and color_col in sub.columns:
        color_kw = {"color": color_col}
        if sub[color_col].dtype.kind in "biufc":
            color_kw["color_continuous_scale"] = [PALETTE[0], PALETTE[5]]
        else:
            color_kw["color_discrete_sequence"] = PALETTE

    fig = px.scatter(
        sub,
        x=x,
        y=y,
        title=f"{y} vs {x}",
        opacity=0.7,
        **color_kw,
    )
    fig.update_layout(**_layout_template())
    fig.update_traces(
        hovertemplate=f"{x} = %{{x}}<br>{y} = %{{y}}<extra></extra>",
        marker={"size": 6, "line": {"width": 0.5, "color": "#999999"}},
    )
    return _fig_to_html(fig)
