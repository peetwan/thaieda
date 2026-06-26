"""ทดสอบ graceful LLM degradation + การผูก narrative เข้ากับ run() (v2.0)."""

from __future__ import annotations

import pandas as pd
import pytest

import thaieda
from thaieda.narrative import NarrativeResult


@pytest.fixture
def no_api_keys(monkeypatch):
    """ลบ API key ทุกตัวเพื่อจำลองสภาพ offline."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _small_df():
    return pd.DataFrame(
        {
            "region": ["เหนือ", "ใต้", "เหนือ", "กลาง", "ใต้", "เหนือ"],
            "sales": [100, 50, 120, 80, 60, 110],
        }
    )


def _fast_run(df, **kwargs):
    return thaieda.run(df, make_charts=False, timeseries=False, insights_engine=False, **kwargs)


class TestNarrativeWiring:
    def test_edaresult_has_narrative_attribute(self):
        result = _fast_run(_small_df())
        assert hasattr(result, "narrative")

    def test_run_populates_narrative(self):
        result = _fast_run(_small_df())
        assert isinstance(result.narrative, NarrativeResult)
        assert result.narrative.executive_summary_th

    def test_narrative_disabled(self):
        result = _fast_run(_small_df(), narrative=False)
        assert result.narrative is None


class TestGracefulLLM:
    def test_llm_no_key_does_not_crash(self, no_api_keys):
        result = _fast_run(_small_df(), llm=True, provider="openai")
        # ต้องไม่ throw และยังมี response (template narrative)
        assert result.llm_response is not None

    def test_llm_falls_back_to_template(self, no_api_keys):
        result = _fast_run(_small_df(), llm=True, provider="openai")
        assert result.narrative is not None
        assert result.llm_response == result.narrative.executive_summary_th

    def test_degradation_note_added(self, no_api_keys):
        result = _fast_run(_small_df(), llm=True, provider="openai")
        assert any("LLM" in note for note in result.notes)

    def test_provider_still_raises_without_key(self, no_api_keys):
        # ระดับ provider ยังคง fail loudly (no silent fallback)
        from thaieda.llm import call_llm

        with pytest.raises(RuntimeError):
            call_llm("hello", "openai")

    def test_no_llm_leaves_response_none(self):
        result = _fast_run(_small_df(), llm=False)
        assert result.llm_response is None
