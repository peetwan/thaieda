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


__all__ = [
    "get_thai_font_path",
    "setup_matplotlib_thai_font",
    "create_wordcloud",
    "create_length_histogram",
    "create_top_tokens_chart",
]
