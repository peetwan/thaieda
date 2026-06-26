"""ทดสอบ generate_narrative — template narrative ไม่ใช้ LLM (v2.0)."""

from __future__ import annotations

import json

from thaieda.insight import Insight
from thaieda.insight_engine import InsightCard, Perspective
from thaieda.narrative import NarrativeResult, generate_narrative


def _sample_dicts():
    return [
        {
            "title_th": "ยอดขายภาคเหนือโดดเด่น",
            "description_th": "ภาคเหนือมียอดขายสูงกว่าค่าเฉลี่ย 2 เท่า",
            "recommendation_th": "ตรวจสอบช่องทางขายภาคเหนือ",
            "severity": "warning",
            "score": 0.9,
        },
        {
            "title_th": "ค่าว่างในคอลัมน์รายได้",
            "description_th": "พบค่าว่าง 15%",
            "recommendation_th": "เติมค่าหรือ drop",
            "severity": "info",
            "score": 0.4,
        },
    ]


class TestGenerateNarrative:
    def test_returns_narrative_result(self):
        r = generate_narrative(_sample_dicts())
        assert isinstance(r, NarrativeResult)

    def test_executive_summaries_non_empty(self):
        r = generate_narrative(_sample_dicts())
        assert r.executive_summary_th
        assert r.executive_summary_en

    def test_th_summary_is_thai(self):
        r = generate_narrative(_sample_dicts())
        assert "ข้อค้นพบ" in r.executive_summary_th

    def test_en_summary_is_english(self):
        r = generate_narrative(_sample_dicts())
        assert "finding" in r.executive_summary_en.lower()

    def test_key_findings_from_insights(self):
        r = generate_narrative(_sample_dicts())
        assert len(r.key_findings) == 2
        assert any("ภาคเหนือ" in f for f in r.key_findings)

    def test_recommendations_collected(self):
        r = generate_narrative(_sample_dicts())
        assert any("ภาคเหนือ" in rec for rec in r.recommendations)

    def test_follow_up_questions_present(self):
        r = generate_narrative(_sample_dicts())
        assert len(r.follow_up_questions) > 0

    def test_language_en(self):
        r = generate_narrative(_sample_dicts(), language="en")
        assert any("?" in q for q in r.follow_up_questions)
        # follow-ups should be in English
        assert any("explain" in q.lower() or "segment" in q.lower() for q in r.follow_up_questions)

    def test_empty_insights_still_produces_summary(self):
        r = generate_narrative([])
        assert r.executive_summary_th
        assert r.executive_summary_en

    def test_none_insights(self):
        r = generate_narrative(None)
        assert isinstance(r, NarrativeResult)

    def test_quality_grade_adds_recommendation(self):
        r = generate_narrative([], quality_score={"score": 45, "grade": "F"})
        assert any("F" in rec for rec in r.recommendations)

    def test_quality_score_in_summary(self):
        r = generate_narrative(_sample_dicts(), quality_score={"score": 80, "grade": "B"})
        assert "80" in r.executive_summary_th

    def test_accepts_insight_card_objects(self):
        card = InsightCard(
            pattern="outstanding",
            perspective=Perspective(breakdown="region", measure="sales", agg="sum"),
            severity="info",
            score=0.7,
            title_th="ภูมิภาคเด่น",
            description_th="รายละเอียด",
            recommendation_th="คำแนะนำ",
            evidence={},
        )
        r = generate_narrative([card])
        assert any("ภูมิภาคเด่น" in f for f in r.key_findings)

    def test_accepts_insight_objects(self):
        ins = Insight(
            category="quality",
            severity="critical",
            title_th="ปัญหาคุณภาพ",
            description_th="desc",
            recommendation_th="rec",
        )
        r = generate_narrative([ins])
        assert any("ปัญหาคุณภาพ" in f for f in r.key_findings)

    def test_critical_sorted_first(self):
        items = [
            {"title_th": "info เรื่อง", "severity": "info", "score": 0.99},
            {"title_th": "critical เรื่อง", "severity": "critical", "score": 0.1},
        ]
        r = generate_narrative(items)
        # critical ควรมาก่อน แม้ score ต่ำกว่า
        assert "critical" in r.key_findings[0].lower() or "critical เรื่อง" in r.key_findings[0]

    def test_cleaning_report_warnings_included(self):
        class FakeReport:
            def to_dict(self):
                return {
                    "rows_before": 100,
                    "rows_after": 90,
                    "total_changes": 50,
                    "warnings": ["คอลัมน์ X ขาดข้อมูลสูง"],
                }

        r = generate_narrative([], cleaning_report=FakeReport())
        assert "90" in r.executive_summary_th
        assert any("ขาดข้อมูล" in rec for rec in r.recommendations)

    def test_to_dict(self):
        r = generate_narrative(_sample_dicts())
        d = r.to_dict()
        assert "executive_summary_th" in d
        assert "key_findings" in d

    def test_to_json_string_and_file(self, tmp_path):
        r = generate_narrative(_sample_dicts())
        s = r.to_json()
        assert "executive_summary_th" in json.loads(s)
        p = tmp_path / "n.json"
        r.to_json(str(p))
        assert "executive_summary_en" in json.loads(p.read_text(encoding="utf-8"))

    def test_deterministic(self):
        r1 = generate_narrative(_sample_dicts(), quality_score={"score": 80, "grade": "B"})
        r2 = generate_narrative(_sample_dicts(), quality_score={"score": 80, "grade": "B"})
        assert r1.to_dict() == r2.to_dict()
