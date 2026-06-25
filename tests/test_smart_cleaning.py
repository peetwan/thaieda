"""ทดสอบ smart cleaning + color palette (v1.1).

ทดสอบ:
  - plan_cleaning ตรวจพบ zwspace, เลขไทย, whitespace, duplicates
  - plan_cleaning ไม่แนะนำ action เมื่อข้อมูลสะอาด
  - CleaningPlan.has_actions ทำงานถูก
  - PALETTE มี 7 สี colorblind-safe
  - get_color, get_cmap วนรอบถ้าเกิน
  - plotly_layout_template คืน dict ที่ถูกต้อง
"""

from __future__ import annotations

import pandas as pd

from thaieda.clean._smart import CleaningPlan, plan_cleaning
from thaieda.viz._palette import (
    PALETTE,
    PLOTLY_FONT_FAMILY,
    get_cmap,
    get_color,
    plotly_layout_template,
)


# ----------------------------------------------------------------------------
# Smart Cleaning Tests
# ----------------------------------------------------------------------------
class TestPlanCleaning:
    """ทดสอบ plan_cleaning — ตัดสินใจว่าควรทำความสะอาดอะไร."""

    def test_clean_data_no_actions(self):
        """ข้อมูลสะอาด → ไม่มี action."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        plan = plan_cleaning(df)
        assert not plan.has_actions
        assert len(plan.actions) == 0

    def test_detects_zwspace(self):
        """เจอ zero-width space → แนะนำ zwspace."""
        df = pd.DataFrame({"text": ["สม\u200bชาย", "ปกติ"]})
        plan = plan_cleaning(df)
        assert "zwspace" in plan.actions
        assert plan.details["zwspace"] > 0

    def test_detects_thai_numerals(self):
        """เจอเลขไทย ๐-๙ → แนะนำ numerals."""
        df = pd.DataFrame({"price": ["๑๒๓", "100"]})
        plan = plan_cleaning(df)
        assert "numerals" in plan.actions
        assert plan.details["numerals"] > 0

    def test_detects_duplicates(self):
        """เจอแถวซ้ำ → แนะนำ duplicates."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        plan = plan_cleaning(df)
        assert "duplicates" in plan.actions
        assert plan.details["duplicates"] > 0

    def test_detects_whitespace(self):
        """เจอ whitespace ซ้ำ → แนะนำ whitespace."""
        df = pd.DataFrame({"text": ["hello  world", "normal"]})
        plan = plan_cleaning(df)
        assert "whitespace" in plan.actions

    def test_detects_placeholders(self):
        """เจอ placeholder values → แนะนำ missing."""
        df = pd.DataFrame({"col": ["data", "-", "N/A", "ไม่มี"]})
        plan = plan_cleaning(df)
        assert "missing" in plan.actions
        assert plan.details["missing"] > 0

    def test_skipped_list_populated(self):
        """action ที่ไม่จำเป็นต้องอยู่ใน skipped."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        plan = plan_cleaning(df)
        assert "zwspace" in plan.skipped
        assert "numerals" in plan.skipped

    def test_empty_dataframe(self):
        """DataFrame ว่าง → ไม่มี action."""
        df = pd.DataFrame()
        plan = plan_cleaning(df)
        assert not plan.has_actions

    def test_multiple_issues(self):
        """เจอหลายปัญหาพร้อมกัน → แนะนำหลาย action."""
        df = pd.DataFrame(
            {
                "text": ["สม\u200bชาย", "๑๒๓", "สม\u200bชาย"],  # zwspace + numerals + แถวซ้ำ
            }
        )
        plan = plan_cleaning(df)
        assert "zwspace" in plan.actions
        assert "numerals" in plan.actions
        assert "duplicates" in plan.actions


class TestCleaningPlan:
    """ทดสอบ CleaningPlan dataclass."""

    def test_has_actions_false_when_empty(self):
        """has_actions = False เมื่อ actions ว่าง."""
        plan = CleaningPlan()
        assert not plan.has_actions

    def test_has_actions_true_when_non_empty(self):
        """has_actions = True เมื่อมี action."""
        plan = CleaningPlan(actions=["zwspace"])
        assert plan.has_actions


# ----------------------------------------------------------------------------
# Color Palette Tests
# ----------------------------------------------------------------------------
class TestPalette:
    """ทดสอบ color palette สำหรับ viz."""

    def test_palette_has_7_colors(self):
        """PALETTE ต้องมี 7 สี (Okabe-Ito)."""
        assert len(PALETTE) == 7

    def test_palette_colors_are_hex(self):
        """ทุกสีต้องเป็น hex format (#RRGGBB)."""
        for color in PALETTE:
            assert color.startswith("#")
            assert len(color) == 7

    def test_get_color_wraps_around(self):
        """get_color วนรอบถ้า index เกิน len(PALETTE)."""
        assert get_color(0) == PALETTE[0]
        assert get_color(7) == PALETTE[0]  # วนรอบ
        assert get_color(8) == PALETTE[1]

    def test_get_cmap_returns_n_colors(self):
        """get_cmap คืน n สี."""
        colors = get_cmap(3)
        assert len(colors) == 3
        assert colors[0] == PALETTE[0]

    def test_get_cmap_wraps_for_large_n(self):
        """get_cmap วนรอบถ้า n > len(PALETTE)."""
        colors = get_cmap(10)
        assert len(colors) == 10
        assert colors[7] == PALETTE[0]  # วนรอบ

    def test_plotly_font_family_has_thai(self):
        """PLOTLY_FONT_FAMILY ต้องมี font ไทย."""
        assert "Sarabun" in PLOTLY_FONT_FAMILY or "Noto Sans Thai" in PLOTLY_FONT_FAMILY

    def test_plotly_layout_template_returns_dict(self):
        """plotly_layout_template คืน dict ที่ใช้กับ Plotly ได้."""
        layout = plotly_layout_template()
        assert isinstance(layout, dict)
        assert "font" in layout
        assert "colorway" in layout
        assert "paper_bgcolor" in layout
        assert layout["colorway"] == PALETTE

    def test_plotly_layout_dark_vs_light(self):
        """dark theme ต้องต่างจาก light theme."""
        dark = plotly_layout_template(dark=True)
        light = plotly_layout_template(dark=False)
        assert dark["paper_bgcolor"] != light["paper_bgcolor"]
