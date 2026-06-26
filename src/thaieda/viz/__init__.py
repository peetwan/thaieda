"""Visualization + font handling — กราฟทั้งหมดคืนค่าเป็น base64 PNG เพื่อฝังใน HTML.

ใช้ matplotlib แบบ Agg (ไม่มี GUI) และจัดการฟอนต์ไทยให้ word cloud/กราฟแสดงผลได้
"""

from __future__ import annotations

import base64
import io
import os
import warnings
from pathlib import Path

import matplotlib

# ใช้ Agg backend เสมอ — ไม่เปิดหน้าต่าง GUI
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thaieda.tokenize import Tokenizer  # noqa: E402

# ----------------------------------------------------------------------------
# การค้นหาฟอนต์ไทย
# ----------------------------------------------------------------------------
# โฟลเดอร์ฟอนต์ที่ bundle มากับแพ็กเกจ (ถ้ามี .ttf วางไว้)
_BUNDLED_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

# ฟอนต์ไทยยอดนิยมและพาธมาตรฐานบนแต่ละ OS
_COMMON_THAI_FONTS = [
    # Windows
    r"C:\Windows\Fonts\THSarabunNew.ttf",
    r"C:\Windows\Fonts\Sarabun-Regular.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\leelawui.ttf",
    r"C:\Windows\Fonts\leelawad.ttf",
    # Linux (Noto / TLWG)
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/tlwg/Sarabun.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "/usr/share/fonts/truetype/thai/Garuda.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Ayuthaya.ttf",
    "/Library/Fonts/Thonburi.ttf",
    "/System/Library/Fonts/Thonburi.ttc",
]

_cached_font_path: str | None = None
_font_search_done = False


def get_thai_font_path() -> str | None:
    """ค้นหาพาธฟอนต์ไทย ตามลำดับ: ฟอนต์ bundle -> ฟอนต์ระบบ -> None.

    คืน None ถ้าไม่พบฟอนต์ใด ๆ (ผู้เรียกต้องจัดการกรณีนี้เอง)
    """
    global _cached_font_path, _font_search_done
    if _font_search_done:
        return _cached_font_path

    _font_search_done = True

    # 1) ฟอนต์ที่ bundle มา
    if _BUNDLED_FONT_DIR.is_dir():
        for ttf in sorted(_BUNDLED_FONT_DIR.glob("*.ttf")) + sorted(
            _BUNDLED_FONT_DIR.glob("*.otf")
        ):
            _cached_font_path = str(ttf)
            return _cached_font_path

    # 2) ฟอนต์ระบบที่พบบ่อย
    for path in _COMMON_THAI_FONTS:
        if os.path.isfile(path):
            _cached_font_path = path
            return _cached_font_path

    # 3) ลองค้นใน matplotlib font manager หาฟอนต์ที่ชื่อเข้าข่ายไทย
    try:
        from matplotlib import font_manager

        thai_keywords = ("sarabun", "thai", "loma", "garuda", "kinnari", "norasi", "tlwg")
        for font in font_manager.fontManager.ttflist:
            name = font.name.lower()
            if any(k in name for k in thai_keywords):
                _cached_font_path = font.fname
                return _cached_font_path
    except Exception:
        pass

    _cached_font_path = None
    return None


_matplotlib_configured = False


def setup_matplotlib_thai_font() -> str | None:
    """ตั้งค่า matplotlib ให้ใช้ฟอนต์ไทย และปิด warning เรื่องฟอนต์.

    คืนพาธฟอนต์ที่ใช้ หรือ None ถ้าไม่พบ (กราฟยังทำงานได้ แต่อักษรไทยอาจเป็นกล่อง)
    """
    global _matplotlib_configured
    font_path = get_thai_font_path()

    # ปิด warning เรื่องหา glyph ไม่เจอ ไม่ให้รก log
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    logging_filter()

    if font_path:
        try:
            from matplotlib import font_manager

            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            family = prop.get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["font.sans-serif"] = [family, *plt.rcParams.get("font.sans-serif", [])]
        except Exception:
            pass

    plt.rcParams["axes.unicode_minus"] = False
    _matplotlib_configured = True
    return font_path


def logging_filter() -> None:
    """ลดเสียง log ของ matplotlib font manager."""
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def _fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """แปลง matplotlib figure เป็นสตริง base64 PNG แล้วปิด figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _font_prop(font_path: str | None):
    """คืน FontProperties สำหรับฟอนต์ที่ระบุ หรือ None."""
    if font_path is None:
        font_path = get_thai_font_path()
    if font_path is None:
        return None
    try:
        from matplotlib import font_manager

        return font_manager.FontProperties(fname=font_path)
    except Exception:
        return None


# ธีมเข้มให้เข้ากับรายงาน
_DARK_BG = "#1a1d23"
_DARK_FG = "#e6e6e6"
_ACCENT = "#4dabf7"
_CRITICAL = "#ff6b6b"  # สีจุด outlier ใน box plot (ให้เข้ากับธีมรายงาน)


def create_wordcloud(
    text: str,
    tokenizer: Tokenizer,
    font_path: str | None = None,
    width: int = 800,
    height: int = 400,
) -> str:
    """สร้าง word cloud จากข้อความไทย คืนค่าเป็น base64 PNG.

    ต้องตัดคำก่อน (ภาษาไทยไม่มีช่องว่างระหว่างคำ) แล้วป้อนให้ wordcloud แบบคั่นช่องว่าง

    Raises:
        ImportError: ถ้าไม่ได้ติดตั้ง wordcloud (แนะนำ pip install thaieda[viz]).
    """
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise ImportError(
            "Word cloud requires pip install thaieda[viz] (the 'wordcloud' package)."
        ) from exc

    if font_path is None:
        font_path = get_thai_font_path()

    # ตัดคำก่อน แล้วต่อด้วยช่องว่างให้ wordcloud นับเป็นคำ
    tokens = [t for t in tokenizer.tokenize(text) if t.strip()]
    joined = " ".join(tokens) if tokens else text

    wc_kwargs = dict(
        width=width,
        height=height,
        background_color=_DARK_BG,
        colormap="Blues_r",
        regexp=r"\S+",  # ใช้ token ที่เราตัดมาแล้ว (คั่นด้วยช่องว่าง)
        collocations=False,
    )
    if font_path:
        wc_kwargs["font_path"] = font_path

    wc = WordCloud(**wc_kwargs).generate(joined)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), facecolor=_DARK_BG)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return _fig_to_base64(fig)


def create_length_histogram(
    lengths: list[int], title: str = "", font_path: str | None = None
) -> str:
    """ฮิสโทแกรมการกระจายความยาวข้อความ คืนค่าเป็น base64 PNG."""
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)

    if lengths:
        n_bins = min(40, max(10, len(set(lengths))))
        ax.hist(lengths, bins=n_bins, color=_ACCENT, edgecolor=_DARK_BG, alpha=0.85)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=_DARK_FG)

    if title:
        ax.set_title(title, color=_DARK_FG, fontproperties=prop, fontsize=12)
    ax.set_xlabel("characters", color=_DARK_FG, fontproperties=prop)
    ax.set_ylabel("count", color=_DARK_FG, fontproperties=prop)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def create_top_tokens_chart(
    top_tokens: list[tuple[str, int]], title: str = "", font_path: str | None = None
) -> str:
    """กราฟแท่งแนวนอนของคำที่พบบ่อย คืนค่าเป็น base64 PNG."""
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)

    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.35 * len(top_tokens) + 1)), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)

    if top_tokens:
        # เรียงจากมากไปน้อย แต่ใน barh ให้บนสุดคือมากสุด
        tokens = [t for t, _ in top_tokens][::-1]
        counts = [c for _, c in top_tokens][::-1]
        positions = range(len(tokens))
        ax.barh(list(positions), counts, color=_ACCENT, alpha=0.9)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(tokens, fontproperties=prop, color=_DARK_FG)
    else:
        ax.text(0.5, 0.5, "no tokens", ha="center", va="center", color=_DARK_FG)

    if title:
        ax.set_title(title, color=_DARK_FG, fontproperties=prop, fontsize=12)
    ax.set_xlabel("frequency", color=_DARK_FG, fontproperties=prop)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def _style_dark_axes(ax: plt.Axes) -> None:
    """จัดสไตล์แกนให้เข้ากับธีมเข้ม."""
    ax.tick_params(colors=_DARK_FG)
    for spine in ax.spines.values():
        spine.set_color("#3a3f47")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ----------------------------------------------------------------------------
# helper สำหรับกราฟชุดใหม่ (ตัวเลข/ค่าว่าง)
# ----------------------------------------------------------------------------
# จำนวนแถวสูงสุดที่นำมาวาดเมทริกซ์ค่าว่าง (เกินนี้สุ่มตัวอย่างเพื่อความเร็ว)
_MISSING_MATRIX_MAX_ROWS = 1000
# คอลัมน์ตัวเลขสูงสุดที่นำมาวาด (กันกราฟล้นเมื่อมีคอลัมน์ตัวเลขจำนวนมาก)
_MAX_NUMERIC_COLS = 30

# เกณฑ์จำนวนคอลัมน์ตัวเลข "กว้างเกินไป" — ข้ามกราฟ O(n^2) เพื่อความเร็ว/หน่วยความจำ (P3)
# correlation heatmap: O(n^2) ของหน่วยความจำ + การวาดช่อง; scatter matrix: O(n^2) ของกราฟย่อย
_CORR_HEATMAP_MAX_COLS = 30
_SCATTER_MATRIX_MAX_COLS = 50

# จำนวนจุดสูงสุดที่นำมา "วาด" ในกราฟเส้น/scatter — เกินนี้ลดจำนวนจุดลงเพื่อความเร็ว
# (กราฟกว้าง ~9 นิ้วแสดงจุดที่แตกต่างได้ไม่เกินไม่กี่พันจุดอยู่แล้ว การลดจุดจึงไม่เสียรายละเอียดที่มองเห็น)
# กราฟเส้น/decomposition: สุ่มแบบเว้นระยะเท่ากัน (รักษารูปร่างตามเวลา); scatter: สุ่มแถว (รักษาการกระจาย)
_PLOT_MAX_POINTS = 5000


def _stride_subsample(n: int, max_points: int = _PLOT_MAX_POINTS) -> slice | np.ndarray:
    """คืน index แบบเว้นระยะเท่ากันเพื่อลดจำนวนจุดที่วาดให้ไม่เกิน max_points (รักษารูปร่างตามลำดับ)."""
    if n <= max_points:
        return slice(None)
    step = (n + max_points - 1) // max_points  # ceil(n / max_points)
    return np.arange(0, n, step)


def _numeric_col_count(df: pd.DataFrame) -> int:
    """จำนวนคอลัมน์ตัวเลข "จริง" ก่อนถูก cap โดย _numeric_frame (ใช้ตัดสินว่ากราฟกว้างเกินไปไหม)."""
    return int(df.select_dtypes(include="number").shape[1])


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """คืนเฉพาะคอลัมน์ตัวเลข (จำกัดจำนวนคอลัมน์เพื่อให้กราฟอ่านได้)."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > _MAX_NUMERIC_COLS:
        numeric = numeric.iloc[:, :_MAX_NUMERIC_COLS]
    return numeric


def _wrap_labels(labels: list[str], limit: int = 14) -> list[str]:
    """ตัดชื่อคอลัมน์ที่ยาวเกินไปเพื่อไม่ให้ทับกันบนแกน."""
    return [lbl if len(lbl) <= limit else lbl[: limit - 1] + "…" for lbl in labels]


def create_correlation_heatmap(df: pd.DataFrame, font_path: str | None = None) -> str:
    """แผนภาพความร้อนสหสัมพันธ์ (correlation) ของคอลัมน์ตัวเลข คืนค่าเป็น base64 PNG.

    คืนสตริงว่างถ้ามีคอลัมน์ตัวเลขน้อยกว่า 2 คอลัมน์ (คำนวณสหสัมพันธ์ไม่ได้)
    ใช้สเกลสีแบบ diverging (แดง–น้ำเงิน) ด้วย matplotlib ล้วน ไม่พึ่ง seaborn
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)
    # P3: ข้าม correlation heatmap เมื่อคอลัมน์ตัวเลขมากเกินไป (O(n^2) memory บนตารางกว้าง)
    if _numeric_col_count(df) > _CORR_HEATMAP_MAX_COLS:
        return ""
    numeric = _numeric_frame(df)
    if numeric.shape[1] < 2:
        return ""

    corr = numeric.corr(numeric_only=True)
    labels = [str(c) for c in corr.columns]
    n = len(labels)
    matrix = corr.to_numpy(dtype="float64")

    fig, ax = plt.subplots(
        figsize=(max(4.5, 0.8 * n + 2), max(4.0, 0.8 * n + 1.5)), facecolor=_DARK_BG
    )
    ax.set_facecolor(_DARK_BG)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(
        _wrap_labels(labels), rotation=45, ha="right", color=_DARK_FG, fontproperties=prop
    )
    ax.set_yticklabels(_wrap_labels(labels), color=_DARK_FG, fontproperties=prop)

    # ใส่ค่าสหสัมพันธ์ในแต่ละช่อง (สีตัวอักษรตัดกับพื้นหลังของช่อง)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            txt_color = "#ffffff" if abs(val) > 0.5 else _DARK_FG
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=_DARK_FG)
    cbar.outline.set_edgecolor("#3a3f47")
    ax.set_title("Correlation", color=_DARK_FG, fontproperties=prop, fontsize=12)
    for spine in ax.spines.values():
        spine.set_color("#3a3f47")
    return _fig_to_base64(fig)


def create_boxplot(df: pd.DataFrame, font_path: str | None = None) -> str:
    """กราฟกล่อง (box plot) ของคอลัมน์ตัวเลขทั้งหมดเรียงเคียงกัน คืนค่าเป็น base64 PNG.

    แสดง outlier เป็นจุดเดี่ยว ๆ คืนสตริงว่างถ้าไม่มีคอลัมน์ตัวเลข
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)
    numeric = _numeric_frame(df)
    if numeric.shape[1] < 1:
        return ""

    labels = [str(c) for c in numeric.columns]
    data = [numeric[c].dropna().to_numpy(dtype="float64") for c in numeric.columns]
    if all(len(d) == 0 for d in data):
        return ""

    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(labels) + 2), 4.0), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)

    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor=_CRITICAL,
            markeredgecolor=_CRITICAL,
            alpha=0.6,
        ),
        medianprops=dict(color="#ffffff"),
        whiskerprops=dict(color=_DARK_FG),
        capprops=dict(color=_DARK_FG),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_ACCENT)
        patch.set_alpha(0.55)
        patch.set_edgecolor(_DARK_FG)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(
        _wrap_labels(labels), rotation=45, ha="right", color=_DARK_FG, fontproperties=prop
    )
    ax.set_title("Box plot", color=_DARK_FG, fontproperties=prop, fontsize=12)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def create_violinplot(df: pd.DataFrame, font_path: str | None = None) -> str:
    """กราฟไวโอลิน (violin plot) ของคอลัมน์ตัวเลข — แสดงรูปร่างการแจกแจง คืนค่าเป็น base64 PNG.

    ต่างจาก box plot ตรงที่เห็นรูปทรงความหนาแน่น ไม่ใช่แค่ค่าสรุป คืนสตริงว่างถ้าไม่มีคอลัมน์ตัวเลข
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)
    numeric = _numeric_frame(df)
    if numeric.shape[1] < 1:
        return ""

    labels: list[str] = []
    data: list[np.ndarray] = []
    for c in numeric.columns:
        arr = numeric[c].dropna().to_numpy(dtype="float64")
        # violinplot ต้องมีอย่างน้อย 2 ค่าเพื่อประมาณความหนาแน่น
        if arr.size >= 2:
            labels.append(str(c))
            data.append(arr)
    if not data:
        return ""

    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(labels) + 2), 4.0), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)

    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor(_ACCENT)
        body.set_edgecolor(_DARK_FG)
        body.set_alpha(0.5)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(_DARK_FG)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(
        _wrap_labels(labels), rotation=45, ha="right", color=_DARK_FG, fontproperties=prop
    )
    ax.set_title("Violin plot", color=_DARK_FG, fontproperties=prop, fontsize=12)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def create_distribution_histogram(
    series: pd.Series, title: str = "", font_path: str | None = None
) -> str:
    """ฮิสโทแกรมการแจกแจงค่าของคอลัมน์ตัวเลข คืนค่าเป็น base64 PNG (ว่างถ้าไม่มีค่าตัวเลข)."""
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return ""

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    n_bins = min(40, max(10, int(np.unique(values).size)))
    ax.hist(values, bins=n_bins, color=_ACCENT, edgecolor=_DARK_BG, alpha=0.85)
    if title:
        ax.set_title(title, color=_DARK_FG, fontproperties=prop, fontsize=12)
    ax.set_xlabel("value", color=_DARK_FG, fontproperties=prop)
    ax.set_ylabel("count", color=_DARK_FG, fontproperties=prop)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def create_missing_matrix(df: pd.DataFrame, font_path: str | None = None) -> str:
    """เมทริกซ์ค่าว่างแบบ missingno: แถว = ข้อมูล, คอลัมน์ = คอลัมน์ คืนค่าเป็น base64 PNG.

    มืด = ค่าว่าง, สว่าง = มีค่า สุ่มตัวอย่างแถวถ้ามากกว่า 1000 แถวเพื่อความเร็ว
    คืนสตริงว่างถ้า DataFrame ไม่มีคอลัมน์
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)
    if df.shape[1] == 0 or df.shape[0] == 0:
        return ""

    sample = df
    note = ""
    if len(df) > _MISSING_MATRIX_MAX_ROWS:
        sample = df.sample(_MISSING_MATRIX_MAX_ROWS, random_state=42).sort_index()
        note = f" (sampled {_MISSING_MATRIX_MAX_ROWS:,}/{len(df):,} rows)"

    # 1 = มีค่า (สว่าง), 0 = ว่าง (มืด)
    present = sample.notna().to_numpy(dtype="float64")
    labels = [str(c) for c in sample.columns]
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#2b3038", _ACCENT])  # [missing(มืด), present(สว่าง)]

    fig, ax = plt.subplots(figsize=(max(4.5, 0.7 * len(labels) + 2), 4.5), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    ax.imshow(present, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        _wrap_labels(labels), rotation=45, ha="right", color=_DARK_FG, fontproperties=prop
    )
    ax.set_ylabel("rows" + note, color=_DARK_FG, fontproperties=prop)
    ax.set_title("Missing value matrix", color=_DARK_FG, fontproperties=prop, fontsize=12)
    ax.tick_params(colors=_DARK_FG)
    for spine in ax.spines.values():
        spine.set_color("#3a3f47")
    return _fig_to_base64(fig)


def create_missing_heatmap(df: pd.DataFrame, font_path: str | None = None) -> str:
    """แผนภาพความร้อนสหสัมพันธ์ของรูปแบบค่าว่าง (nullity correlation) คืนค่าเป็น base64 PNG.

    แสดงว่าค่าว่างของคอลัมน์ต่าง ๆ "หายไปพร้อมกัน" แค่ไหน (เหมือน missingno.heatmap)
    คืนสตริงว่างถ้ามีคอลัมน์ที่มีค่าว่าง (และค่าว่างไม่คงที่) น้อยกว่า 2 คอลัมน์
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)
    if df.shape[1] == 0:
        return ""

    null_df = df.isna()
    # เลือกเฉพาะคอลัมน์ที่ "มีทั้งค่าว่างและไม่ว่าง" (nullity ไม่คงที่ ถึงจะคำนวณ correlation ได้)
    varying = [c for c in null_df.columns if 0 < int(null_df[c].sum()) < len(null_df)]
    if len(varying) < 2:
        return ""

    corr = null_df[varying].astype(float).corr()
    labels = [str(c) for c in corr.columns]
    n = len(labels)
    matrix = corr.to_numpy(dtype="float64")
    matrix = np.nan_to_num(matrix, nan=0.0)

    fig, ax = plt.subplots(
        figsize=(max(4.5, 0.8 * n + 2), max(4.0, 0.8 * n + 1.5)), facecolor=_DARK_BG
    )
    ax.set_facecolor(_DARK_BG)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(
        _wrap_labels(labels), rotation=45, ha="right", color=_DARK_FG, fontproperties=prop
    )
    ax.set_yticklabels(_wrap_labels(labels), color=_DARK_FG, fontproperties=prop)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            txt_color = "#ffffff" if abs(val) > 0.5 else _DARK_FG
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=_DARK_FG)
    cbar.outline.set_edgecolor("#3a3f47")
    ax.set_title("Nullity correlation", color=_DARK_FG, fontproperties=prop, fontsize=12)
    for spine in ax.spines.values():
        spine.set_color("#3a3f47")
    return _fig_to_base64(fig)


# จำนวนคอลัมน์ตัวเลขสูงสุดที่นำมาวาด scatter matrix (เกินนี้กราฟอ่านไม่ออก)
_MAX_SCATTER_COLS = 5


def create_scatter_matrix(df: pd.DataFrame, font_path: str | None = None) -> str:
    """scatter matrix (pairwise) ของคอลัมน์ตัวเลข — เส้นทแยงเป็นฮิสโทแกรม คืนค่าเป็น base64 PNG.

    คืนสตริงว่างถ้ามีคอลัมน์ตัวเลขน้อยกว่า 2 คอลัมน์ หรือไม่มีแถวที่ครบทุกคอลัมน์
    จำกัดที่ 5 คอลัมน์แรก (_MAX_SCATTER_COLS) เพื่อให้กราฟยังอ่านได้
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)
    # P3: ข้าม scatter matrix เมื่อคอลัมน์ตัวเลขมากเกินไป (เลือกแค่ไม่กี่คอลัมน์จากหลายร้อยไม่สื่อความหมาย)
    if _numeric_col_count(df) > _SCATTER_MATRIX_MAX_COLS:
        return ""
    numeric = _numeric_frame(df)
    if numeric.shape[1] < 2:
        return ""

    cols = list(numeric.columns)[:_MAX_SCATTER_COLS]
    data = numeric[cols].dropna()
    if len(data) < 2:
        return ""
    n = len(cols)
    labels = _wrap_labels([str(c) for c in cols])

    fig, axes = plt.subplots(
        n, n, figsize=(2.2 * n + 1, 2.2 * n + 1), facecolor=_DARK_BG, squeeze=False
    )
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ax.set_facecolor(_DARK_BG)
            if i == j:
                ax.hist(data[cols[i]].to_numpy(dtype="float64"), bins=20, color=_ACCENT, alpha=0.85)
            else:
                ax.scatter(
                    data[cols[j]].to_numpy(dtype="float64"),
                    data[cols[i]].to_numpy(dtype="float64"),
                    s=8,
                    color=_ACCENT,
                    alpha=0.5,
                    edgecolors="none",
                )
            _style_dark_axes(ax)
            ax.tick_params(labelsize=7)
            if i == n - 1:
                ax.set_xlabel(labels[j], color=_DARK_FG, fontproperties=prop, fontsize=9)
            if j == 0:
                ax.set_ylabel(labels[i], color=_DARK_FG, fontproperties=prop, fontsize=9)
    fig.suptitle("Scatter matrix", color=_DARK_FG, fontproperties=prop, fontsize=12)
    return _fig_to_base64(fig)


def create_category_bar(
    series: pd.Series, title: str = "", font_path: str | None = None, top_n: int = 20
) -> str:
    """กราฟแท่งแนวนอนของความถี่ค่าหมวดหมู่ (value counts) คืนค่าเป็น base64 PNG.

    คืนสตริงว่างถ้าคอลัมน์ไม่มีค่าที่ไม่ว่าง
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return ""
    vc = non_null.value_counts().head(top_n)
    if vc.empty:
        return ""

    cats = [str(k) for k in vc.index][::-1]
    counts = [int(v) for v in vc.to_numpy()][::-1]
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.35 * len(cats) + 1)), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    positions = range(len(cats))
    ax.barh(list(positions), counts, color=_ACCENT, alpha=0.9)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(_wrap_labels(cats, limit=24), fontproperties=prop, color=_DARK_FG)
    if title:
        ax.set_title(title, color=_DARK_FG, fontproperties=prop, fontsize=12)
    ax.set_xlabel("count", color=_DARK_FG, fontproperties=prop)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


# ประเภทคอลัมน์ที่ถือเป็นข้อความ (ใช้สร้างฮิสโทแกรมความยาว) — เก็บเป็น string เทียบกับ ColumnType.value
_TEXTUAL_TYPE_VALUES = {"thai_text", "english_text", "mixed_text"}


def auto_visualize(
    df: pd.DataFrame, column_types: dict, font_path: str | None = None
) -> dict[str, str]:
    """เลือกและสร้างกราฟที่เหมาะสมกับข้อมูลโดยอัตโนมัติ — คืน dict ชื่อกราฟ -> base64 PNG.

    ตรรกะการเลือก:
      - มีคอลัมน์ตัวเลข >1 คอลัมน์ : correlation heatmap + box plot + violin plot
      - มีค่าว่าง                  : missing matrix (+ missing heatmap ถ้า >1 คอลัมน์ที่มีค่าว่าง)
      - คอลัมน์ตัวเลขแต่ละคอลัมน์   : ฮิสโทแกรมการแจกแจง (key 'distribution::<col>')
      - คอลัมน์ข้อความแต่ละคอลัมน์  : ฮิสโทแกรมความยาวอักขระ (key 'length_hist::<col>')

    หมายเหตุ: word cloud ต้องใช้ tokenizer จึงไม่ถูกสร้างที่นี่ (รายงานสร้างแยกต่างหาก)
    กราฟใดที่สร้างไม่ได้ (ข้อมูลไม่พอ) จะถูกข้าม ไม่ใส่ใน dict
    คีย์ที่เป็นชนิด dataset-level จะไม่ถูกใส่ถ้าได้สตริงว่าง
    """
    if font_path is None:
        font_path = get_thai_font_path()

    # แปลง column_types values เป็น string (รองรับทั้ง ColumnType enum และ str)
    type_values = {str(k): getattr(v, "value", v) for k, v in column_types.items()}

    charts: dict[str, str] = {}
    numeric = _numeric_frame(df)

    # --- กราฟระดับชุดข้อมูล (ตัวเลข) ---
    if numeric.shape[1] > 1:
        for name, fn in (
            ("correlation_heatmap", create_correlation_heatmap),
            ("boxplot", create_boxplot),
            ("violinplot", create_violinplot),
        ):
            img = fn(df, font_path=font_path)
            if img:
                charts[name] = img

    # --- ค่าว่าง ---
    if int(df.isna().sum().sum()) > 0:
        img = create_missing_matrix(df, font_path=font_path)
        if img:
            charts["missing_matrix"] = img
        cols_with_missing = [c for c in df.columns if int(df[c].isna().sum()) > 0]
        if len(cols_with_missing) > 1:
            img = create_missing_heatmap(df, font_path=font_path)
            if img:
                charts["missing_heatmap"] = img

    # --- ฮิสโทแกรมต่อคอลัมน์ตัวเลข ---
    for col in numeric.columns:
        img = create_distribution_histogram(df[col], title=str(col), font_path=font_path)
        if img:
            charts[f"distribution::{col}"] = img

    # --- ฮิสโทแกรมความยาวต่อคอลัมน์ข้อความ ---
    for col in df.columns:
        if type_values.get(str(col)) in _TEXTUAL_TYPE_VALUES:
            lengths = [len(s) for s in df[col].dropna().astype(str)]
            if lengths:
                img = create_length_histogram(lengths, title=str(col), font_path=font_path)
                if img:
                    charts[f"length_hist::{col}"] = img

    return charts


# ----------------------------------------------------------------------------
# กราฟ timeseries (เส้น/decomposition/ACF)
# ----------------------------------------------------------------------------
_SEASONAL = "#69db7c"  # สีเขียว สำหรับองค์ประกอบ seasonal
_RESIDUAL = "#ffa94d"  # สีส้ม สำหรับ residual


def create_timeseries_plot(series: pd.Series, title: str = "", font_path: str | None = None) -> str:
    """กราฟเส้น timeseries พร้อมเส้นแนวโน้ม (trend line) คืนค่าเป็น base64 PNG.

    ถ้า index เป็น DatetimeIndex จะใช้เป็นแกน x (เรียงตามเวลา) ไม่งั้นใช้ตำแหน่ง
    คืนสตริงว่างถ้าไม่มีค่าตัวเลข
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)

    numeric = pd.to_numeric(series, errors="coerce")
    if isinstance(series.index, pd.DatetimeIndex):
        numeric = numeric.sort_index()
    valid = numeric.dropna()
    if valid.empty:
        return ""

    y = valid.to_numpy(dtype="float64")
    x = valid.index if isinstance(valid.index, pd.DatetimeIndex) else np.arange(y.size)

    # เส้นแนวโน้มคำนวณบนข้อมูลเต็ม (แม่นยำ) แล้วค่อยลดจำนวนจุดตอนวาดเท่านั้น
    trend_line = None
    if y.size >= 2 and float(np.std(y)) > 0:
        xi = np.arange(y.size, dtype="float64")
        slope, intercept = np.polyfit(xi, y, 1)
        trend_line = slope * xi + intercept

    # ลดจำนวนจุดที่วาดบนซีรีส์ยาว (รักษารูปร่างตามเวลาด้วยการเว้นระยะเท่ากัน)
    idx = _stride_subsample(y.size)
    x_plot = x[idx]

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    ax.plot(x_plot, y[idx], color=_ACCENT, linewidth=1.2, alpha=0.9)
    if trend_line is not None:
        ax.plot(
            x_plot, trend_line[idx], color=_CRITICAL, linewidth=1.4, linestyle="--", alpha=0.85
        )

    if title:
        ax.set_title(title, color=_DARK_FG, fontproperties=prop, fontsize=12)
    ax.set_xlabel("time", color=_DARK_FG, fontproperties=prop)
    ax.set_ylabel("value", color=_DARK_FG, fontproperties=prop)
    if isinstance(x, pd.DatetimeIndex):
        fig.autofmt_xdate()
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def create_decomposition_plot(
    components: dict[str, list[float]], title: str = "", font_path: str | None = None
) -> str:
    """กราฟ STL decomposition 4 แผง (observed/trend/seasonal/residual) คืนค่าเป็น base64 PNG.

    components ต้องมีคีย์ 'trend', 'seasonal', 'residual' (observed = ผลรวมของทั้งสาม)
    คืนสตริงว่างถ้าไม่มีองค์ประกอบใดเลย
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)

    trend = np.asarray(components.get("trend", []), dtype="float64")
    seasonal = np.asarray(components.get("seasonal", []), dtype="float64")
    residual = np.asarray(components.get("residual", []), dtype="float64")
    if trend.size == 0 and seasonal.size == 0 and residual.size == 0:
        return ""

    n = max(trend.size, seasonal.size, residual.size)

    def _pad(arr: np.ndarray) -> np.ndarray:
        if arr.size == n:
            return arr
        out = np.zeros(n, dtype="float64")
        out[: arr.size] = arr
        return out

    trend, seasonal, residual = _pad(trend), _pad(seasonal), _pad(residual)
    observed = trend + seasonal + residual
    # ลดจำนวนจุดที่วาดบนซีรีส์ยาว (เว้นระยะเท่ากัน — รักษารูปร่างของแต่ละองค์ประกอบ)
    idx = _stride_subsample(n)
    x = np.arange(n)[idx]

    panels = [
        ("observed", observed[idx], _ACCENT),
        ("trend", trend[idx], _CRITICAL),
        ("seasonal", seasonal[idx], _SEASONAL),
        ("residual", residual[idx], _RESIDUAL),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(9, 7), facecolor=_DARK_BG, sharex=True)
    for ax, (name, data, color) in zip(axes, panels, strict=False):
        ax.set_facecolor(_DARK_BG)
        if name == "residual":
            ax.scatter(x, data, s=6, color=color, alpha=0.6, edgecolors="none")
            ax.axhline(0, color=_DARK_FG, linewidth=0.6, alpha=0.5)
        else:
            ax.plot(x, data, color=color, linewidth=1.1)
        ax.set_ylabel(name, color=_DARK_FG, fontproperties=prop, fontsize=10)
        _style_dark_axes(ax)
    if title:
        fig.suptitle(title, color=_DARK_FG, fontproperties=prop, fontsize=12)
    axes[-1].set_xlabel("time", color=_DARK_FG, fontproperties=prop)
    return _fig_to_base64(fig)


def create_acf_plot(
    series: pd.Series, lags: int = 40, title: str = "", font_path: str | None = None
) -> str:
    """กราฟ autocorrelation function (ACF) แบบ stem คืนค่าเป็น base64 PNG.

    ใช้ numpy ล้วน (ไม่พึ่ง statsmodels) คืนสตริงว่างถ้าข้อมูลน้อยกว่า 2 จุด
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)

    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype="float64")
    if values.size < 2:
        return ""

    max_lag = min(lags, values.size - 1)
    x = values - values.mean()
    denom = float(np.sum(x * x))
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        acf.append(0.0 if denom == 0 else float(np.sum(x[lag:] * x[:-lag]) / denom))

    # แถบนัยสำคัญโดยประมาณ ±1.96/sqrt(n)
    conf = 1.96 / np.sqrt(values.size)

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    positions = range(len(acf))
    ax.vlines(list(positions), 0, acf, color=_ACCENT, linewidth=1.5)
    ax.scatter(list(positions), acf, s=14, color=_ACCENT)
    ax.axhline(0, color=_DARK_FG, linewidth=0.6, alpha=0.6)
    ax.axhline(conf, color=_CRITICAL, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(-conf, color=_CRITICAL, linewidth=0.8, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, color=_DARK_FG, fontproperties=prop, fontsize=12)
    ax.set_xlabel("lag", color=_DARK_FG, fontproperties=prop)
    ax.set_ylabel("ACF", color=_DARK_FG, fontproperties=prop)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def _top_tokens_from_series(series: pd.Series, tokenizer: Tokenizer, top_n: int = 20):
    """นับคำที่พบบ่อยจากคอลัมน์ข้อความ (ตัดคำด้วย tokenizer, ทิ้ง token ที่เป็นเครื่องหมายล้วน)."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for value in series.head(2000):
        for tok in tokenizer.tokenize(value):
            t = tok.strip()
            if t and any(c.isalnum() for c in t):
                counter[t] += 1
    return counter.most_common(top_n)


def auto_select_charts(
    df: pd.DataFrame,
    tokenizer: Tokenizer | None = None,
    font_path: str | None = None,
    text_columns: list[str] | None = None,
) -> dict[str, str]:
    """เลือก chart type อัตโนมัติตามชนิดข้อมูล — คืน dict {ชื่อกราฟ: base64 PNG}.

    ตรรกะการเลือก:
      - ตัวเลข × ตัวเลข (>=2 คอลัมน์) : correlation heatmap + scatter matrix
      - การแจกแจงตัวเลข (>=1 คอลัมน์)  : box plot + violin plot + ฮิสโทแกรมต่อคอลัมน์
      - มีค่าว่าง                      : missing matrix (+ missing heatmap ถ้าหลายคอลัมน์)
      - คอลัมน์ข้อความ                 : ฮิสโทแกรมความยาว + (ถ้ามี tokenizer) top tokens + word cloud
      - คอลัมน์หมวดหมู่                : กราฟแท่งความถี่ค่า (value counts)

    text_columns: ถ้าไม่ระบุ จะตรวจหาคอลัมน์ข้อความเองด้วยโมดูล detect
    กราฟใดที่สร้างไม่ได้ (ข้อมูลไม่พอ) จะถูกข้าม ไม่ใส่ใน dict
    """
    from thaieda.detect import ColumnType, detect_all

    if font_path is None:
        font_path = get_thai_font_path()

    types = detect_all(df)
    charts: dict[str, str] = {}
    numeric = _numeric_frame(df)

    # --- ตัวเลข × ตัวเลข ---
    if numeric.shape[1] >= 2:
        for name, fn in (
            ("correlation_heatmap", create_correlation_heatmap),
            ("scatter_matrix", create_scatter_matrix),
        ):
            img = fn(df, font_path=font_path)
            if img:
                charts[name] = img

    # --- การแจกแจงตัวเลข ---
    if numeric.shape[1] >= 1:
        for name, fn in (("boxplot", create_boxplot), ("violinplot", create_violinplot)):
            img = fn(df, font_path=font_path)
            if img:
                charts[name] = img
        for col in numeric.columns:
            img = create_distribution_histogram(df[col], title=str(col), font_path=font_path)
            if img:
                charts[f"distribution::{col}"] = img

    # --- ค่าว่าง ---
    if int(df.isna().sum().sum()) > 0:
        img = create_missing_matrix(df, font_path=font_path)
        if img:
            charts["missing_matrix"] = img
        cols_with_missing = [c for c in df.columns if int(df[c].isna().sum()) > 0]
        if len(cols_with_missing) > 1:
            img = create_missing_heatmap(df, font_path=font_path)
            if img:
                charts["missing_heatmap"] = img

    # --- คอลัมน์ข้อความ ---
    text_types = {ColumnType.THAI_TEXT, ColumnType.MIXED_TEXT, ColumnType.ENGLISH_TEXT}
    if text_columns is None:
        text_columns = [str(c) for c in df.columns if types.get(str(c)) in text_types]
    for col in text_columns:
        if col not in df.columns:
            continue
        non_null = df[col].dropna().astype(str)
        if non_null.empty:
            continue
        lengths = [len(s) for s in non_null]
        if lengths:
            img = create_length_histogram(lengths, title=str(col), font_path=font_path)
            if img:
                charts[f"length_hist::{col}"] = img
        if tokenizer is not None:
            top = _top_tokens_from_series(non_null, tokenizer)
            if top:
                charts[f"top_tokens::{col}"] = create_top_tokens_chart(top, font_path=font_path)
            try:
                joined = " ".join(non_null.head(2000))
                if joined.strip():
                    charts[f"wordcloud::{col}"] = create_wordcloud(
                        joined, tokenizer, font_path=font_path
                    )
            except ImportError:
                # ไม่มี wordcloud (optional extra) — ข้ามเงียบ ๆ
                pass

    # --- คอลัมน์หมวดหมู่ ---
    for col in df.columns:
        if types.get(str(col)) in (ColumnType.CATEGORICAL, ColumnType.ID):
            img = create_category_bar(df[col], title=str(col), font_path=font_path)
            if img:
                charts[f"valuecounts::{col}"] = img

    return charts


# ------------------------------------------------------------------------------
# Insight charts (v0.7) — กราฟสำหรับ cross-column insight cards
# ------------------------------------------------------------------------------
# สีประจำแต่ละ pattern (เข้ากับธีมเข้ม)
_PATTERN_COLORS = {
    "outstanding": "#69db7c",  # เขียว — เด่น
    "attribution": "#ffd43b",  # เหลือง — สัดส่วน
    "comparison": "#ff8787",  # แดงอ่อน — ต่าง
    "trend": "#4dabf7",  # ฟ้า — แนวโน้ม
}
# จำนวน segment สูงสุดที่แสดงในกราฟ (กันรกเกินไป)
_MAX_INSIGHT_BARS = 15
# จำนวน bucket สูงสุดในกราฟ trend
_MAX_TREND_POINTS = 50


def create_insight_outstanding_chart(
    segments: list[list],
    top_segment: str,
    title: str = "",
    font_path: str | None = None,
) -> str:
    """กราฟแท่งแนวนอน — เปรียบเทียบ segment ที่โดดเด่นกับกลุ่มอื่น (outstanding pattern).

    segments: [[label, value], ...] — เรียงจากมากไปน้อย
    ไฮไลต์ segment ที่โดดเด่นด้วยสีเขียว, ที่เหลือสีจาง
    คืนสตริงว่างถ้าไม่มีข้อมูล
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)

    if not segments:
        return ""
    labels = [str(s[0]) for s in segments[:_MAX_INSIGHT_BARS]]
    values = [float(s[1]) for s in segments[:_MAX_INSIGHT_BARS]]
    # เรียง barh จากล่างขึ้น (มากสุดบน)
    labels = labels[::-1]
    values = values[::-1]
    colors = [_PATTERN_COLORS["outstanding"] if lbl == top_segment else "#3a3f47" for lbl in labels]

    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.35 * len(labels) + 1)), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    positions = range(len(labels))
    ax.barh(list(positions), values, color=colors, alpha=0.85)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(_wrap_labels(labels, limit=20), fontproperties=prop, color=_DARK_FG)
    if title:
        ax.set_title(title, color=_DARK_FG, fontproperties=prop, fontsize=11)
    ax.set_xlabel("value", color=_DARK_FG, fontproperties=prop)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def create_insight_attribution_chart(
    segments: list[list],
    top_segment: str,
    share: float,
    title: str = "",
    font_path: str | None = None,
) -> str:
    """กราฟโดนัท (donut) — แสดงสัดส่วน segment ที่ครอง vs ที่เหลือ (attribution pattern).

    segments: [[label, value], ...] เรียงจากมากไปน้อย
    แสดง top segment เป็นสีเหลือง, ที่เหลือรวมกันเป็นสีเทา
    คืนสตริงว่างถ้าไม่มีข้อมูล
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)

    if not segments:
        return ""
    top_val = float(segments[0][1])
    rest_val = sum(float(s[1]) for s in segments[1:])
    if top_val <= 0 and rest_val <= 0:
        return ""

    sizes = [top_val, rest_val] if rest_val > 0 else [top_val]
    colors = (
        [_PATTERN_COLORS["attribution"], "#3a3f47"]
        if rest_val > 0
        else [_PATTERN_COLORS["attribution"]]
    )

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor=_DARK_BG, linewidth=2),
        textprops=dict(color=_DARK_FG, fontproperties=prop, fontsize=10),
    )
    for t in autotexts:
        t.set_color("#15171c" if t.get_text() == f"{int(share)}%" else _DARK_FG)
        t.set_fontweight("bold")
    # ข้อความกลางโดนัท
    ax.text(
        0,
        0,
        f"{share:.0f}%",
        ha="center",
        va="center",
        color=_PATTERN_COLORS["attribution"],
        fontproperties=prop,
        fontsize=20,
        fontweight="bold",
    )
    if title:
        ax.set_title(title, color=_DARK_FG, fontproperties=prop, fontsize=11)
    return _fig_to_base64(fig)


def create_insight_comparison_chart(
    df: pd.DataFrame,
    breakdown: str,
    measure: str,
    top_segment: str,
    title: str = "",
    font_path: str | None = None,
) -> str:
    """Box plot ตามกลุ่ม — เปรียบเทียบการแจกแจงของ measure ระหว่าง segment เด่น vs ที่เหลือ.

    ไฮไลต์กลุ่มเด่นด้วยสีแดงอ่อน, ที่เหลือสีเทา
    จำกัดจำนวนกลุ่มที่แสดง (top 10 + "อื่น ๆ") กันกราฟรก
    คืนสตริงว่างถ้าข้อมูลไม่พอ
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)

    num = pd.to_numeric(df[measure], errors="coerce")
    cat = df[breakdown].astype(str)
    valid = num.notna() & cat.notna()
    if int(valid.sum()) < 4:
        return ""

    frame = pd.DataFrame({"_g": cat[valid], "_m": num[valid]})
    vc = frame["_g"].value_counts()
    # เก็บ top 9 กลุ่ม + รวมที่เหลือเป็น "อื่น ๆ"
    top_groups = vc.index[:9].tolist()
    frame["_g"] = frame["_g"].where(frame["_g"].isin(top_groups), "อื่น ๆ")
    # เรียง: top_segment ก่อน, แล้วตามขนาด
    groups = frame.groupby("_g", observed=True)["_m"]
    group_list = sorted(
        groups.groups.keys(),
        key=lambda g: (0 if g == top_segment else 1, -groups.size()[g]),
    )
    if len(group_list) < 2:
        return ""

    data = [groups.get_group(g).to_numpy(dtype="float64") for g in group_list]
    labels = _wrap_labels([str(g) for g in group_list], limit=14)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(group_list) + 2), 4.5), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor="#5c5c5c",
            markeredgecolor="#5c5c5c",
            alpha=0.4,
        ),
        medianprops=dict(color="#ffffff"),
        whiskerprops=dict(color=_DARK_FG),
        capprops=dict(color=_DARK_FG),
    )
    for i, patch in enumerate(bp["boxes"]):
        if group_list[i] == top_segment:
            patch.set_facecolor(_PATTERN_COLORS["comparison"])
            patch.set_alpha(0.55)
        else:
            patch.set_facecolor("#3a3f47")
            patch.set_alpha(0.45)
        patch.set_edgecolor(_DARK_FG)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", color=_DARK_FG, fontproperties=prop)
    if title:
        ax.set_title(title, color=_DARK_FG, fontproperties=prop, fontsize=11)
    ax.set_ylabel(measure, color=_DARK_FG, fontproperties=prop, fontsize=10)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def create_insight_trend_chart(
    segments: list[list],
    direction: str,
    tau: float,
    title: str = "",
    font_path: str | None = None,
) -> str:
    """กราฟเส้น — แสดงแนวโน้มตามช่วงเวลา (trend pattern).

    segments: [[label, value], ...] เรียงตามลำดับเวลา
    เส้นสีฟ้าพร้อมจุด, ลูกศรชี้ทิศทาง (เพิ่ม/ลด)
    คืนสตริงว่างถ้าไม่มีข้อมูล
    """
    setup_matplotlib_thai_font()
    prop = _font_prop(font_path)

    if not segments or len(segments) < 2:
        return ""
    labels = [str(s[0]) for s in segments[:_MAX_TREND_POINTS]]
    values = [float(s[1]) for s in segments[:_MAX_TREND_POINTS]]
    x = list(range(len(values)))

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    ax.plot(x, values, color=_PATTERN_COLORS["trend"], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(x, values, color=_PATTERN_COLORS["trend"], s=24, zorder=3)
    # เติมพื้นใต้เส้นเบา ๆ
    ax.fill_between(
        x,
        values,
        min(values) - (max(values) - min(values)) * 0.1,
        color=_PATTERN_COLORS["trend"],
        alpha=0.08,
    )

    # ลูกศรทิศทาง (ใช้ ASCII เพื่อหลีกเลี่ยง glyph ที่อาจไม่มีใน font ไทย)
    arrow_y = values[-1]
    val_range = max(values) - min(values)
    y_offset = val_range * 0.15 * (1 if direction == "up" else -1)
    ax.annotate(
        "Trend UP" if direction == "up" else "Trend DOWN",
        xy=(x[-1], arrow_y),
        xytext=(x[-1] - 1.5, arrow_y + y_offset),
        color=_PATTERN_COLORS["trend"],
        fontproperties=prop,
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=_PATTERN_COLORS["trend"], lw=1.5),
    )

    n = len(labels)
    step = max(1, n // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(
        _wrap_labels([labels[i] for i in range(0, n, step)], limit=12),
        rotation=30,
        ha="right",
        color=_DARK_FG,
        fontproperties=prop,
    )
    if title:
        ax.set_title(f"{title} (τ={tau:.2f})", color=_DARK_FG, fontproperties=prop, fontsize=11)
    ax.set_ylabel("value", color=_DARK_FG, fontproperties=prop)
    _style_dark_axes(ax)
    return _fig_to_base64(fig)


def create_insight_chart(
    card_dict: dict,
    df: pd.DataFrame | None = None,
    font_path: str | None = None,
) -> str:
    """เลือกกราฟที่เหมาะกับ insight card โดยอัตโนมัติ — คืน base64 PNG หรือสตริงว่าง.

    card_dict: InsightCard.to_dict() ที่มี pattern, perspective, evidence
    df: DataFrame ต้นฉบับ (สำหรับ comparison ที่ต้องเข้าถึงข้อมูลดิบ)
    """
    if font_path is None:
        font_path = get_thai_font_path()

    pattern = card_dict.get("pattern", "")
    evidence = card_dict.get("evidence", {})
    perspective = card_dict.get("perspective", {})
    breakdown = perspective.get("breakdown", "")
    measure = perspective.get("measure")

    if pattern == "outstanding":
        segments = evidence.get("top_segments", [])
        if not segments:
            return ""
        # เติม top_segment ถ้ามี (อาจไม่ได้อยู่ใน top_segments)
        top_seg = evidence.get("top_segment", "")
        top_val = evidence.get("top_value", 0)
        if top_seg and not any(s[0] == top_seg for s in segments):
            segments = [[top_seg, top_val]] + segments
        return create_insight_outstanding_chart(
            segments,
            top_seg,
            title=f"{breakdown} × {measure or 'count'}",
            font_path=font_path,
        )

    if pattern == "attribution":
        segments = evidence.get("top_segments", [])
        if not segments:
            return ""
        top_seg = evidence.get("top_segment", "")
        share = float(evidence.get("share", 0))
        return create_insight_attribution_chart(
            segments,
            top_seg,
            share,
            title=f"{breakdown} × {measure or 'count'}",
            font_path=font_path,
        )

    if pattern == "comparison":
        if df is None or not breakdown or not measure:
            return ""
        top_seg = evidence.get("top_segment", "")
        return create_insight_comparison_chart(
            df,
            breakdown,
            measure,
            top_seg,
            title=f"{breakdown} × {measure}",
            font_path=font_path,
        )

    if pattern == "trend":
        # v0.8: ใช้ all_buckets จาก evidence แทนการสร้างจาก first/last แค่ 2 จุด
        segments = evidence.get("all_buckets", [])
        if not segments:
            # fallback เก่า: สร้างจาก first/last ถ้าไม่มี all_buckets
            first_v = evidence.get("first_value", 0)
            last_v = evidence.get("last_value", 0)
            segments = [
                [evidence.get("first_bucket", "start"), first_v],
                [evidence.get("last_bucket", "end"), last_v],
            ]
        return create_insight_trend_chart(
            segments,
            evidence.get("direction", "up"),
            float(evidence.get("tau", 0)),
            title=f"{breakdown} × {measure or 'count'}",
            font_path=font_path,
        )

    return ""


__all__ = [
    "get_thai_font_path",
    "setup_matplotlib_thai_font",
    "create_wordcloud",
    "create_length_histogram",
    "create_top_tokens_chart",
    "create_correlation_heatmap",
    "create_boxplot",
    "create_violinplot",
    "create_distribution_histogram",
    "create_missing_matrix",
    "create_missing_heatmap",
    "create_scatter_matrix",
    "create_category_bar",
    "create_timeseries_plot",
    "create_decomposition_plot",
    "create_acf_plot",
    "create_insight_outstanding_chart",
    "create_insight_attribution_chart",
    "create_insight_comparison_chart",
    "create_insight_trend_chart",
    "create_insight_chart",
    "auto_visualize",
    "auto_select_charts",
]
