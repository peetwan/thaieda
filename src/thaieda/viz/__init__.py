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
    "auto_visualize",
]
