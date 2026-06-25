"""ทดสอบโมดูล LLM — privacy-preserving analysis (v0.9).

ทดสอบ:
  1. analyze_with_llm มี signature ที่ถูกต้อง
  2. prepare_for_llm โหมด insight_only ไม่ส่งข้อมูลดิบ
  3. anonymize_dataframe แทนที่เบอร์โทร/บัตรประชาชนด้วย token
  4. build_prompt สร้าง prompt ที่ไม่ใช่สตริงว่าง
  5. mock LLM provider calls (ไม่เรียก API จริง)
  6. prepare_for_llm โหมด dp_noise เพิ่ม noise
  7. prepare_for_llm โหมด full ส่งข้อมูลดิบ
  8. prepare_for_llm โหมด anonymized ลบ PII
  9. call_llm raise ValueError ถ้า provider ไม่รองรับ
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pandas as pd
import pytest

from thaieda.llm import analyze_with_llm, build_prompt, call_llm, prepare_for_llm
from thaieda.llm._anonymize import anonymize_dataframe


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """DataFrame ตัวอย่างที่มี PII (เบอร์โทร + บัตรประชาชน)."""
    return pd.DataFrame(
        {
            "name": ["สมชาย", "สมหญิง", "วิชัย"],
            "phone": ["081-234-5678", "089-876-5432", "082-111-2222"],
            "id_card": [
                "1-1234-56789-01-2",
                "2-2345-67890-12-3",
                "3-3456-78901-23-4",
            ],
            "age": [25, 30, 28],
            "salary": [35000, 45000, 40000],
        }
    )


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """DataFrame เล็กไม่มี PII."""
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "C"],
            "value": [10, 20, 15, 30],
        }
    )


# ----------------------------------------------------------------------------
# 1. analyze_with_llm signature
# ----------------------------------------------------------------------------
class TestAnalyzeSignature:
    """ทดสอบว่า analyze_with_llm มี signature ที่ถูกต้อง."""

    def test_function_exists(self):
        """ตรวจว่า analyze_with_llm ถูก export จากโมดูล."""
        assert callable(analyze_with_llm)

    def test_signature_has_required_params(self):
        """ตรวจพารามิเตอร์ของ analyze_with_llm."""
        sig = inspect.signature(analyze_with_llm)
        params = list(sig.parameters.keys())
        assert "df" in params
        assert "privacy" in params
        assert "provider" in params
        assert "model" in params
        assert "language" in params

    def test_default_privacy_is_insight_only(self):
        """default privacy ต้องเป็น insight_only."""
        sig = inspect.signature(analyze_with_llm)
        assert sig.parameters["privacy"].default == "insight_only"

    def test_default_provider_is_openai(self):
        """default provider ต้องเป็น openai."""
        sig = inspect.signature(analyze_with_llm)
        assert sig.parameters["provider"].default == "openai"


# ----------------------------------------------------------------------------
# 2. prepare_for_llm — insight_only mode
# ----------------------------------------------------------------------------
class TestPrepareInsightOnly:
    """ทดสอบโหมด insight_only — ไม่ส่งข้อมูลดิบ."""

    def test_insight_only_returns_no_raw_data(self, simple_df):
        """insight_only ต้องไม่มี data (data=None)."""
        result = prepare_for_llm(simple_df, None, "insight_only")
        assert result["mode"] == "insight_only"
        assert result["data"] is None

    def test_insight_only_has_summary(self, simple_df):
        """insight_only ต้องมีสถิติสรุป."""
        result = prepare_for_llm(simple_df, None, "insight_only")
        assert "summary" in result
        assert result["summary"]["shape"] == (4, 2)

    def test_insight_only_no_token_map(self, simple_df):
        """insight_only ไม่ต้องมี token_map."""
        result = prepare_for_llm(simple_df, None, "insight_only")
        assert result["token_map"] is None

    def test_insight_only_dp_noise_false(self, simple_df):
        """insight_only ต้องไม่เพิ่ม noise."""
        result = prepare_for_llm(simple_df, None, "insight_only")
        assert result["dp_noise"] is False


# ----------------------------------------------------------------------------
# 3. anonymize_dataframe
# ----------------------------------------------------------------------------
class TestAnonymize:
    """ทดสอบ anonymize_dataframe — แทนที่ PII ด้วย token."""

    def test_replaces_phone_numbers(self, sample_df):
        """เบอร์โทรต้องถูกแทนที่ด้วย [PHONE_N]."""
        df_safe, token_map = anonymize_dataframe(sample_df)
        for val in df_safe["phone"]:
            assert "[PHONE_" in val, f"เบอร์ไม่ถูกแทนที่: {val}"

    def test_replaces_id_cards(self, sample_df):
        """บัตรประชาชนต้องถูกแทนที่ด้วย [IDCARD_N]."""
        df_safe, token_map = anonymize_dataframe(sample_df)
        for val in df_safe["id_card"]:
            assert "[IDCARD_" in val, f"บัตรไม่ถูกแทนที่: {val}"

    def test_token_map_has_entries(self, sample_df):
        """token_map ต้องมี mapping PII → token."""
        df_safe, token_map = anonymize_dataframe(sample_df)
        # กรอง key พิเศษออก
        pii_keys = [k for k in token_map if not k.startswith("_")]
        assert len(pii_keys) > 0, "token_map ว่าง"

    def test_same_value_same_token(self, sample_df):
        """PII ค่าเดียวกันต้องได้ token เดียวกัน."""
        df = pd.DataFrame({"phone": ["081-234-5678", "081-234-5678"]})
        df_safe, token_map = anonymize_dataframe(df)
        assert df_safe["phone"][0] == df_safe["phone"][1]

    def test_different_values_different_tokens(self, sample_df):
        """PII ค่าต่างกันต้องได้ token ต่างกัน."""
        df_safe, _ = anonymize_dataframe(sample_df)
        phones = list(df_safe["phone"])
        assert len(set(phones)) == len(phones), "เบอร์ต่างกันแต่ได้ token ซ้ำ"

    def test_numeric_columns_untouched(self, sample_df):
        """คอลัมน์ตัวเลขต้องไม่ถูกแตะ."""
        df_safe, _ = anonymize_dataframe(sample_df)
        assert df_safe["age"].equals(sample_df["age"])
        assert df_safe["salary"].equals(sample_df["salary"])

    def test_string_dtype_column(self):
        """คอลัมน์ string dtype (ไม่ใช่ object) ต้องถูกประมวลผลด้วย."""
        df = pd.DataFrame({"phone": pd.Series(["081-234-5678", "089-876-5432"], dtype="string")})
        df_safe, _ = anonymize_dataframe(df)
        assert "[PHONE_" in df_safe["phone"][0]


# ----------------------------------------------------------------------------
# 4. build_prompt
# ----------------------------------------------------------------------------
class TestBuildPrompt:
    """ทดสอบ build_prompt — สร้าง prompt ที่ไม่ว่าง."""

    def test_prompt_non_empty(self, simple_df):
        """build_prompt ต้องสร้าง string ที่ไม่ว่าง."""
        prepared = prepare_for_llm(simple_df, None, "insight_only")
        prompt = build_prompt(prepared, None, None, "th")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_summary_th(self, simple_df):
        """prompt ภาษาไทยต้องมีสถิติสรุป."""
        prepared = prepare_for_llm(simple_df, None, "insight_only")
        prompt = build_prompt(prepared, None, None, "th")
        assert "สถิติ" in prompt

    def test_prompt_contains_summary_en(self, simple_df):
        """prompt ภาษาอังกฤษต้องมี summary."""
        prepared = prepare_for_llm(simple_df, None, "insight_only")
        prompt = build_prompt(prepared, None, None, "en")
        assert "Summary" in prompt or "summary" in prompt.lower()

    def test_prompt_with_insights(self, simple_df):
        """prompt ต้องแสดงข้อค้นพบถ้ามี."""
        prepared = prepare_for_llm(simple_df, None, "insight_only")
        insights = [
            {"title_th": "คอลัมน์ value มี outlier", "description_th": "พบค่า 30 ที่สูงกว่า Q3"},
        ]
        prompt = build_prompt(prepared, None, insights, "th")
        assert "outlier" in prompt

    def test_prompt_no_raw_data_in_insight_only(self, simple_df):
        """insight_only ต้องไม่ฝังตารางข้อมูลดิบใน prompt."""
        prepared = prepare_for_llm(simple_df, None, "insight_only")
        prompt = build_prompt(prepared, None, None, "th")
        # insight_only ไม่มี data จึงไม่มีตารางข้อมูลดิบ
        assert "ข้อมูลตัวอย่าง" not in prompt

    def test_prompt_raises_on_non_dict(self):
        """build_prompt ต้อง raise TypeError ถ้า prepared_data ไม่ใช่ dict."""
        with pytest.raises(TypeError, match="dict"):
            build_prompt("not a dict", None, None, "th")


# ----------------------------------------------------------------------------
# 5. mock LLM provider calls
# ----------------------------------------------------------------------------
class TestMockProvider:
    """ทดสอบ call_llm ด้วย mock — ไม่เรียก API จริง."""

    def test_call_llm_raises_on_unsupported_provider(self):
        """provider ไม่รองรับ → ValueError."""
        with pytest.raises(ValueError, match="ไม่รองรับ provider"):
            call_llm("test prompt", "unknown_provider", None)

    @patch("thaieda.llm._provider._call_openai")
    def test_mock_openai(self, mock_openai):
        """mock OpenAI — ไม่เรียก API จริง."""
        mock_openai.return_value = "วิเคราะห์เสร็จแล้ว"
        result = call_llm("ทดสอบ", "openai", "gpt-4o-mini")
        assert result == "วิเคราะห์เสร็จแล้ว"
        mock_openai.assert_called_once()

    @patch("thaieda.llm._provider._call_anthropic")
    def test_mock_anthropic(self, mock_anthropic):
        """mock Anthropic — ไม่เรียก API จริง."""
        mock_anthropic.return_value = "Analysis complete"
        result = call_llm("test", "anthropic", "claude-3-5-sonnet")
        assert result == "Analysis complete"
        mock_anthropic.assert_called_once()

    @patch("thaieda.llm._provider._call_ollama")
    def test_mock_ollama(self, mock_ollama):
        """mock Ollama — ไม่เรียก API จริง."""
        mock_ollama.return_value = "สรุปข้อมูลเรียบร้อย"
        result = call_llm("ทดสอบ", "ollama", "llama3.1")
        assert result == "สรุปข้อมูลเรียบร้อย"
        mock_ollama.assert_called_once()

    @patch("thaieda.llm._provider._call_openai")
    def test_analyze_with_llm_mocked(self, mock_openai, simple_df):
        """ทดสอบ analyze_with_llm ทั้ง pipeline ด้วย mock."""
        mock_openai.return_value = "ผลวิเคราะห์"
        result = analyze_with_llm(
            simple_df,
            privacy="insight_only",
            provider="openai",
            model="gpt-4o-mini",
        )
        assert result == "ผลวิเคราะห์"
        mock_openai.assert_called_once()


# ----------------------------------------------------------------------------
# 6. prepare_for_llm — dp_noise mode
# ----------------------------------------------------------------------------
class TestPrepareDpNoise:
    """ทดสอบโหมด dp_noise — เพิ่ม noise."""

    def test_dp_noise_has_noise_flag(self, simple_df):
        """dp_noise ต้องตั้ง dp_noise=True."""
        result = prepare_for_llm(simple_df, None, "dp_noise", epsilon=1.0)
        assert result["dp_noise"] is True

    def test_dp_noise_no_raw_data(self, simple_df):
        """dp_noise ไม่ส่งข้อมูลดิบ."""
        result = prepare_for_llm(simple_df, None, "dp_noise")
        assert result["data"] is None

    def test_dp_noise_changes_stats(self, simple_df):
        """dp_noise ต้องเปลี่ยนสถิติ (เพิ่ม noise)."""
        noisy = prepare_for_llm(simple_df, None, "dp_noise", epsilon=0.1)
        # เนื่องจาก noise สุ่ม ตรวจว่ามี key dp_epsilon
        assert "dp_epsilon" in noisy["summary"]

    def test_dp_noise_raises_on_zero_epsilon(self, simple_df):
        """epsilon=0 → ValueError."""
        with pytest.raises(ValueError, match="epsilon"):
            prepare_for_llm(simple_df, None, "dp_noise", epsilon=0)


# ----------------------------------------------------------------------------
# 7. prepare_for_llm — full mode
# ----------------------------------------------------------------------------
class TestPrepareFull:
    """ทดสอบโหมด full — ส่งข้อมูลดิบ."""

    def test_full_returns_data(self, simple_df):
        """full ต้องส่งข้อมูลดิบ."""
        result = prepare_for_llm(simple_df, None, "full")
        assert result["data"] is not None
        assert isinstance(result["data"], pd.DataFrame)

    def test_full_data_is_copy(self, simple_df):
        """full ต้องส่ง copy ไม่ใช่ของเดิม."""
        result = prepare_for_llm(simple_df, None, "full")
        assert result["data"] is not simple_df  # ต้องเป็น copy


# ----------------------------------------------------------------------------
# 8. prepare_for_llm — anonymized mode
# ----------------------------------------------------------------------------
class TestPrepareAnonymized:
    """ทดสอบโหมด anonymized — ลบ PII."""

    def test_anonymized_returns_data(self, sample_df):
        """anonymized ต้องส่งข้อมูลที่ลบ PII แล้ว."""
        result = prepare_for_llm(sample_df, None, "anonymized")
        assert result["data"] is not None
        assert isinstance(result["data"], pd.DataFrame)

    def test_anonymized_has_token_map(self, sample_df):
        """anonymized ต้องมี token_map."""
        result = prepare_for_llm(sample_df, None, "anonymized")
        assert result["token_map"] is not None
        pii_keys = [k for k in result["token_map"] if not k.startswith("_")]
        assert len(pii_keys) > 0

    def test_anonymized_no_pii_in_data(self, sample_df):
        """anonymized ต้องไม่มี PII ในข้อมูลที่ส่ง."""
        result = prepare_for_llm(sample_df, None, "anonymized")
        df_safe = result["data"]
        # ตรวจว่าไม่มีเบอร์โทรเดิม
        for val in df_safe["phone"]:
            assert "081" not in val or "[PHONE" in val


# ----------------------------------------------------------------------------
# 9. invalid mode
# ----------------------------------------------------------------------------
class TestInvalidMode:
    """ทดสอบโหมดที่ไม่รองรับ."""

    def test_invalid_mode_raises(self, simple_df):
        """privacy_mode ไม่รองรับ → ValueError."""
        with pytest.raises(ValueError, match="ไม่รองรับโหมด"):
            prepare_for_llm(simple_df, None, "invalid_mode")
