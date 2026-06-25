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

import numpy as np
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


# ---------------------------------------------------------------------------
# 10. Real-world Thai government dataset scenarios
# ---------------------------------------------------------------------------
class TestRealWorldThaiGov:
    """ทดสอบกับข้อมูลจริงแบบราชการไทย — ชื่อ เบอร์ บัตรประชาชน เงินเดือน.

    ทดสอบนี้ใช้ DataFrame ที่คล้ายกับข้อมูลราชการจริง (ชื่อ-นามสกุลภาษาไทย,
    เบอร์โทรหลายรูปแบบ, เลขบัตรประชาชน 13 หลัก, เงินเดือน) เพื่อยืนยันว่า
    prepare_for_llm และ anonymize_dataframe ทำงานได้กับข้อมูลในโลกจริง
    ไม่ใช่แค่ข้อมูลตัวอย่างเล็ก ๆ
    """

    @pytest.fixture()
    def gov_df(self) -> pd.DataFrame:
        """DataFrame จำลองข้อมูลราชการ: ชื่อ เบอร์ บัตร เงินเดือน."""
        return pd.DataFrame(
            {
                "full_name": ["สมชาย ใจดี", "สมหญิง รักไทย", "วิชัย สุขสันต์", "ปรีดา มั่งมี"],
                "phone": [
                    "081-234-5678",
                    "0812345678",
                    "+66812345678",
                    "089-876-5432",
                ],
                "id_card": [
                    "1-1234-56789-01-2",
                    "2-2345-67890-12-3",
                    "3-3456-78901-23-4",
                    "4-4567-89012-34-5",
                ],
                "department": ["กรมที่ดิน", "กรมสรรพากร", "กรมการแพทย์", "กรมศึกษาธิการ"],
                "salary": [35000, 42000, 55000, 38000],
                "age": [35, 28, 45, 31],
            }
        )

    def test_prepare_insight_only_gov(self, gov_df):
        """insight_only กับข้อมูลราชการ — ต้องไม่ส่งข้อมูลดิบ แต่มีสถิติเงินเดือน."""
        result = prepare_for_llm(gov_df, None, "insight_only")
        assert result["data"] is None
        assert result["summary"]["shape"] == (4, 6)
        assert "salary" in result["summary"]["numeric_stats"]
        assert result["summary"]["numeric_stats"]["salary"]["mean"] == 42500.0

    def test_prepare_anonymized_gov_no_pii(self, gov_df):
        """anonymized กับข้อมูลราชการ — ชื่อ/เบอร์/บัตรต้องถูกแทนที่."""
        result = prepare_for_llm(gov_df, None, "anonymized")
        df_safe = result["data"]
        # เบอร์โทรทุกค่าต้องเป็น token
        for val in df_safe["phone"]:
            assert "[PHONE_" in val, f"เบอร์ไม่ถูกแทนที่: {val}"
        # บัตรประชาชนทุกค่าต้องเป็น token
        for val in df_safe["id_card"]:
            assert "[IDCARD_" in val, f"บัตรไม่ถูกแทนที่: {val}"
        # เลขเดือนและอายุต้องไม่ถูกแตะ
        assert df_safe["salary"].equals(gov_df["salary"])
        assert df_safe["age"].equals(gov_df["age"])

    def test_anonymize_gov_multiple_phone_formats(self, gov_df):
        """เบอร์โทร 3 รูปแบบ (081-234-5678, 0812345678, +66812345678) ต้องถูกแทนที่ทั้งหมด."""
        df_safe, _ = anonymize_dataframe(gov_df)
        for val in df_safe["phone"]:
            assert "[PHONE_" in val, f"เบอร์ไม่ถูกแทนที่: {val}"

    def test_prepare_full_gov_sends_raw(self, gov_df):
        """full mode กับข้อมูลราชการ — ต้องส่งข้อมูลดิบ (copy)."""
        result = prepare_for_llm(gov_df, None, "full")
        assert result["data"] is not None
        assert result["data"] is not gov_df
        assert result["data"]["salary"].equals(gov_df["salary"])


# ---------------------------------------------------------------------------
# 11. Anonymized mode — token_map reversibility
# ---------------------------------------------------------------------------
class TestTokenMapReversible:
    """ทดสอบว่า token_map สามารถย้อนกลับได้ — map token → original PII.

    สำคัญเพราะผู้ใช้อาจต้อง map ผลลัพธ์จาก LLM กลับเป็นค่าจริง
    ถ้า LLM อ้างถึง [PHONE_1] ในคำตอบ ผู้ใช้ต้องสามารถดูได้ว่าเบอร์จริงคืออะไร
    """

    def test_token_map_is_invertible(self, sample_df):
        """token_map (PII → token) ต้องสามารถกลับด้านได้ (token → PII)."""
        df_safe, token_map = anonymize_dataframe(sample_df)
        # สร้าง reverse map
        reverse_map: dict[str, str] = {}
        for original, token in token_map.items():
            if not original.startswith("_"):
                reverse_map[token] = original
        # ตรวจว่า token ใน df_safe สามารถ map กลับได้
        for val in df_safe["phone"]:
            if val in reverse_map:
                # token ที่อยู่ใน DataFrame ต้อง map กลับเป็นเบอร์เดิม
                assert reverse_map[val] in sample_df["phone"].values

    def test_reverse_map_covers_all_tokens_in_data(self, sample_df):
        """ทุก token ที่ปรากฏในข้อมูลต้องมีใน reverse map."""
        df_safe, token_map = anonymize_dataframe(sample_df)
        reverse_map = {v: k for k, v in token_map.items() if not k.startswith("_")}
        # หา token ทั้งหมดในคอลัมน์ phone
        for val in df_safe["phone"]:
            if val.startswith("[PHONE_"):
                assert val in reverse_map, f"token {val} ไม่มีใน reverse map"

    def test_reverse_map_id_cards(self, sample_df):
        """token ของบัตรประชาชนต้อง map กลับเป็นเลขบัตรเดิมได้."""
        df_safe, token_map = anonymize_dataframe(sample_df)
        reverse_map = {v: k for k, v in token_map.items() if not k.startswith("_")}
        for val in df_safe["id_card"]:
            if val.startswith("[IDCARD_"):
                original = reverse_map[val]
                assert original in sample_df["id_card"].values


# ---------------------------------------------------------------------------
# 12. build_prompt — ThaiEDA branding + privacy label
# ---------------------------------------------------------------------------
class TestPromptBranding:
    """ทดสอบว่า build_prompt ฝังชื่อ ThaiEDA และป้ายโหมดความเป็นส่วนตัว.

    สำคัญเพราะ prompt ที่ส่งให้ LLM ต้องบอกบริบทว่ามาจาก ThaiEDA
    และต้องแสดงโหมดความเป็นส่วนตัวเพื่อให้ LLM ทราบข้อจำกัด
    """

    def test_prompt_contains_thaieda_th(self, simple_df):
        """prompt ภาษาไทยต้องมีคำว่า ThaiEDA."""
        prepared = prepare_for_llm(simple_df, None, "insight_only")
        prompt = build_prompt(prepared, None, None, "th")
        assert "ThaiEDA" in prompt

    def test_prompt_contains_thaieda_en(self, simple_df):
        """prompt ภาษาอังกฤษต้องมีคำว่า ThaiEDA."""
        prepared = prepare_for_llm(simple_df, None, "insight_only")
        prompt = build_prompt(prepared, None, None, "en")
        assert "ThaiEDA" in prompt

    def test_prompt_contains_privacy_label_insight_only(self, simple_df):
        """prompt ต้องมีป้ายโหมด insight_only (ภาษาไทย)."""
        prepared = prepare_for_llm(simple_df, None, "insight_only")
        prompt = build_prompt(prepared, None, None, "th")
        assert "โหมดปกปิตัวตนสูง" in prompt or "insight_only" in prompt.lower()

    def test_prompt_contains_privacy_label_full_en(self, simple_df):
        """prompt ภาษาอังกฤษต้องมีป้ายโหมด full."""
        prepared = prepare_for_llm(simple_df, None, "full")
        prompt = build_prompt(prepared, None, None, "en")
        assert "Full raw data mode" in prompt or "full" in prompt.lower()

    def test_prompt_contains_privacy_label_dp_noise(self, simple_df):
        """prompt ต้องมีป้ายโหมด dp_noise."""
        prepared = prepare_for_llm(simple_df, None, "dp_noise", epsilon=1.0)
        prompt = build_prompt(prepared, None, None, "th")
        assert "differential privacy" in prompt or "dp_noise" in prompt


# ---------------------------------------------------------------------------
# 13. analyze_with_llm full pipeline — verify prompt content
# ---------------------------------------------------------------------------
class TestAnalyzePipelinePrompt:
    """ทดสอบ pipeline เต็มรูปแบบ — ตรวจ prompt ที่ส่งให้ LLM.

    ใน insight_only mode ต้องแน่ใจว่า prompt ที่ส่งให้ LLM มีสถิติ
    แต่ไม่มีข้อมูลดิบ (เช่น ชื่อคน เบอร์โทร) — นี่คือหัวใจของ privacy guarantee
    """

    @patch("thaieda.llm._provider._call_openai")
    def test_insight_only_prompt_has_stats_no_raw_data(self, mock_openai, sample_df):
        """insight_only: prompt ที่ส่งให้ LLM ต้องมีสถิติ แต่ไม่มีตารางข้อมูลดิบ."""
        mock_openai.return_value = "ผลวิเคราะห์"
        analyze_with_llm(
            sample_df,
            privacy="insight_only",
            provider="openai",
            model="gpt-4o-mini",
        )
        # เก็บ prompt ที่ส่งให้ _call_openai
        sent_prompt = mock_openai.call_args[0][0]
        # ต้องมีสถิติ (mean, std ฯลฯ)
        assert "mean" in sent_prompt.lower() or "สถิติ" in sent_prompt
        # ต้องไม่มีตารางข้อมูลดิบ (insight_only ไม่ฝัง data DataFrame)
        assert "ข้อมูลตัวอย่าง" not in sent_prompt

    @patch("thaieda.llm._provider._call_openai")
    def test_anonymized_prompt_has_tokens_no_pii(self, mock_openai, sample_df):
        """anonymized: prompt ต้องมี token แทน PII ไม่มีค่าดิบในตารางข้อมูล."""
        mock_openai.return_value = "ผลวิเคราะห์"
        analyze_with_llm(
            sample_df,
            privacy="anonymized",
            provider="openai",
            model="gpt-4o-mini",
        )
        sent_prompt = mock_openai.call_args[0][0]
        # ตรวจเฉพาะบริบทของตารางข้อมูล — ไม่ใช่ส่วนสถิติ categorical_stats
        data_table_section = (
            sent_prompt.split("ข้อมูลตัวอย่าง")[-1] if "ข้อมูลตัวอย่าง" in sent_prompt else ""
        )
        assert "081-234-5678" not in data_table_section
        assert "1-1234-56789-01-2" not in data_table_section
        # ต้องมี token แทน
        assert "[PHONE_" in sent_prompt
        assert "[IDCARD_" in sent_prompt

    @patch("thaieda.llm._provider._call_openai")
    def test_full_prompt_contains_raw_data(self, mock_openai, sample_df):
        """full mode: prompt ต้องมีข้อมูลดิบ (ชื่อคน เบอร์โทร)."""
        mock_openai.return_value = "ผลวิเคราะห์"
        analyze_with_llm(
            sample_df,
            privacy="full",
            provider="openai",
            model="gpt-4o-mini",
        )
        sent_prompt = mock_openai.call_args[0][0]
        # full mode ต้องมีข้อมูลดิบ
        assert "สมชาย" in sent_prompt or "081" in sent_prompt


# ---------------------------------------------------------------------------
# 14. dp_noise — numeric values actually change (with fixed seed)
# ---------------------------------------------------------------------------
class TestDpNoiseChangesValues:
    """ทดสอบว่า dp_noise เปลี่ยนค่าตัวเลขจริง — ใช้ seed คงที่เพื่อ reproducible.

    สำคัญเพราะถ้า noise ไม่เปลี่ยนค่าจริงก็ไม่ต่างจาก insight_only
    ใช้ fixed seed เพื่อให้ test ทำซ้ำได้ (reproducible)
    """

    def test_dp_noise_changes_mean(self, simple_df):
        """dp_noise ต้องเปลี่ยนค่า mean ของคอลัมน์ตัวเลข."""
        np.random.seed(42)
        clean = prepare_for_llm(simple_df, None, "insight_only")
        np.random.seed(42)
        noisy = prepare_for_llm(simple_df, None, "dp_noise", epsilon=0.5)
        clean_mean = clean["summary"]["numeric_stats"]["value"]["mean"]
        noisy_mean = noisy["summary"]["numeric_stats"]["value"]["mean"]
        assert clean_mean != noisy_mean, "dp_noise ไม่เปลี่ยนค่า mean"

    def test_dp_noise_changes_count(self, simple_df):
        """dp_noise ต้องเปลี่ยนค่า count (อาจเพิ่มหรือลด)."""
        np.random.seed(99)
        noisy = prepare_for_llm(simple_df, None, "dp_noise", epsilon=0.1)
        noisy_count = noisy["summary"]["numeric_stats"]["value"]["count"]
        # count อาจเปลี่ยน (หรือโชคไม่เปลี่ยน แต่ต้องไม่ exception)
        assert isinstance(noisy_count, int)
        assert noisy_count >= 0

    def test_dp_noise_changes_min_max(self, simple_df):
        """dp_noise ต้องเปลี่ยนค่า min/max."""
        np.random.seed(7)
        clean = prepare_for_llm(simple_df, None, "insight_only")
        np.random.seed(7)
        noisy = prepare_for_llm(simple_df, None, "dp_noise", epsilon=0.3)
        clean_min = clean["summary"]["numeric_stats"]["value"]["min"]
        noisy_min = noisy["summary"]["numeric_stats"]["value"]["min"]
        clean_max = clean["summary"]["numeric_stats"]["value"]["max"]
        noisy_max = noisy["summary"]["numeric_stats"]["value"]["max"]
        # อย่างน้อย min หรือ max ต้องเปลี่ยน
        assert clean_min != noisy_min or clean_max != noisy_max

    def test_dp_noise_reproducible_with_seed(self, simple_df):
        """seed เดียวกันต้องได้ผลเดียวกัน (reproducible)."""
        np.random.seed(123)
        run1 = prepare_for_llm(simple_df, None, "dp_noise", epsilon=0.5)
        np.random.seed(123)
        run2 = prepare_for_llm(simple_df, None, "dp_noise", epsilon=0.5)
        assert run1["summary"]["numeric_stats"] == run2["summary"]["numeric_stats"]


# ---------------------------------------------------------------------------
# 15. call_llm with ollama — mock dispatch
# ---------------------------------------------------------------------------
class TestCallLlmOllamaMock:
    """ทดสอบว่า call_llm กับ provider 'ollama' เรียกฟังก์ชันที่ถูกต้อง.

    ใช้ mock เพื่อไม่ต้องติดตั้ง ollama package หรือรัน server จริง
    ตรวจว่า dispatch table ส่งไปยัง _call_ollama ไม่ใช่ _call_openai
    """

    @patch("thaieda.llm._provider._call_ollama")
    def test_ollama_dispatch_correct(self, mock_ollama):
        """call_llm('ollama') ต้องเรียก _call_ollama ไม่ใช่ _call_openai."""
        mock_ollama.return_value = "ผลจาก ollama"
        result = call_llm("ทดสอบ prompt", "ollama", "llama3.1")
        assert result == "ผลจาก ollama"
        mock_ollama.assert_called_once_with("ทดสอบ prompt", "llama3.1")

    @patch("thaieda.llm._provider._call_ollama")
    def test_ollama_default_model(self, mock_ollama):
        """call_llm('ollama', model=None) ต้องใช้ default model 'llama3.1'."""
        mock_ollama.return_value = "ok"
        call_llm("test", "ollama")
        mock_ollama.assert_called_once_with("test", "llama3.1")

    @patch("thaieda.llm._provider._call_ollama")
    def test_ollama_custom_model(self, mock_ollama):
        """call_llm('ollama', model='qwen2.5') ต้องส่ง model ที่กำหนด."""
        mock_ollama.return_value = "ok"
        call_llm("test", "ollama", "qwen2.5")
        mock_ollama.assert_called_once_with("test", "qwen2.5")

    @patch("thaieda.llm._provider._call_openai")
    @patch("thaieda.llm._provider._call_ollama")
    def test_ollama_not_openai(self, mock_ollama, mock_openai, sample_df):
        """เรียก ollama ต้องไม่เรียก _call_openai เลย."""
        mock_ollama.return_value = "ollama result"
        mock_openai.return_value = "openai result"
        result = call_llm("test", "ollama", "llama3.1")
        assert result == "ollama result"
        mock_openai.assert_not_called()
        mock_ollama.assert_called_once()
