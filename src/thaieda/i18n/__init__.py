"""Bilingual (Thai/English) labels สำหรับ ThaiEDA.

ตัวเลือกภาษาเริ่มต้นคือภาษาไทย เพราะเครื่องมือนี้ออกแบบมาเพื่อผู้ใช้ไทยเป็นหลัก
"""

from __future__ import annotations

# ภาษาเริ่มต้น (global) — เปลี่ยนได้ด้วย set_language()
_DEFAULT_LANGUAGE = "th"
_current_language = _DEFAULT_LANGUAGE

# พจนานุกรมแปลศัพท์เทคนิคให้เป็นภาษาคนอ่านสำหรับรายงาน
TECHNICAL_TO_PLAIN: dict[str, str] = {
    "modified_z_score": "ค่าที่หลุดจากช่วงปกติ",
    "modified z-score": "ค่าที่หลุดจากช่วงปกติ",
    "MAD": "ช่วงกลางของข้อมูล",
    "MeanAD": "ค่าเฉลี่ยของระยะห่างจากค่ากลาง",
    "IQR": "ช่วงกลาง 50% ของข้อมูล",
    "z_score": "คะแนนห่างจากค่าเฉลี่ย",
    "z-score": "คะแนนห่างจากค่าเฉลี่ย",
    "skewness": "การเบ้",
    "skew": "การเบ้",
    "kurtosis": "ความโดดเด่น",
    "Isolation Forest": "การตรวจจับแบบ ML",
    "Local Outlier Factor": "การตรวจจับค่าที่ต่างจากเพื่อนบ้าน",
    "LOF": "การตรวจจับค่าที่ต่างจากเพื่อนบ้าน",
    "Benjamini-Hochberg": "การคุม false positive",
    "false positive": "การแจ้งเตือนเกินจริง",
    "p-value": "ค่าความน่าเชื่อถือทางสถิติ",
    "Pearson": "ความสัมพันธ์เชิงเส้น",
    "ANOVA": "การเปรียบเทียบค่าเฉลี่ยหลายกลุ่ม",
    "Chi-square": "การทดสอบความสัมพันธ์ของหมวดหมู่",
    "Autocorrelation": "ความสัมพันธ์กับค่าก่อนหน้า",
    "ACF": "ความสัมพันธ์กับค่าก่อนหน้า",
    "STL Decomposition": "การแยกแนวโน้ม/ฤดูกาล",
    "outlier": "ค่าผิดปกติ",
    "outliers": "ค่าผิดปกติ",
    "nullity correlation": "รูปแบบค่าว่างที่เกิดร่วมกัน",
    "zero variance": "ไม่มีความแปรปรวน",
}


# พจนานุกรม label ทั้งหมดที่รายงานต้องใช้
LABELS: dict[str, dict[str, str]] = {
    # --- หัวข้อหลัก ---
    "report_title": {"en": "ThaiEDA Profile Report", "th": "รายงานวิเคราะห์ข้อมูล ThaiEDA"},
    "overview": {"en": "Overview", "th": "ภาพรวม"},
    "what_is_this_data": {"en": "1. What is this data?", "th": "1. ข้อมูลนี้คืออะไร"},
    "detected_data_type": {"en": "Detected data type", "th": "ประเภทข้อมูลที่ตรวจพบ"},
    "eda_focus": {"en": "Recommended EDA focus", "th": "ควรดูอะไรเป็นพิเศษ"},
    "most_important": {"en": "2. Most Important", "th": "2. สำคัญที่สุด"},
    "what_to_do_first": {"en": "3. What to do first", "th": "3. ควรทำอะไรก่อน"},
    "details": {"en": "4. Details", "th": "4. รายละเอียด"},
    "next_steps": {"en": "5. What to do next", "th": "5. ควรทำอะไรต่อ"},
    "business_impact": {"en": "Business impact", "th": "ผลกระทบทางธุรกิจ"},
    "technical_detail": {"en": "Technical detail", "th": "รายละเอียดเทคนิค"},
    "impact": {"en": "Impact", "th": "ผลกระทบ"},
    "detected_language": {"en": "Detected language", "th": "ภาษาข้อมูลที่ตรวจพบ"},
    "language_thai": {"en": "Thai", "th": "ไทย"},
    "language_english": {"en": "English", "th": "อังกฤษ"},
    "language_mixed": {"en": "Mixed (Thai + English)", "th": "ผสม (ไทย + อังกฤษ)"},
    "language_numeric": {
        "en": "No text (numeric/date only)",
        "th": "ไม่มีข้อความ (ตัวเลข/วันที่ล้วน)",
    },
    "language_impact": {"en": "Impact on analysis", "th": "ผลกระทบต่อการวิเคราะห์"},
    "thai_specific_recommendations": {
        "en": "Thai-specific recommendations",
        "th": "คำแนะนำเฉพาะข้อมูลไทย",
    },
    # --- auto insights (สรุปข้อค้นพบสำคัญ) ---
    "auto_insights": {"en": "Key Insights", "th": "ข้อค้นพบสำคัญ"},
    "executive_summary": {"en": "Executive Summary", "th": "บทสรุปผู้บริหาร"},
    "recommendation": {"en": "Recommendation", "th": "คำแนะนำ"},
    "no_insights": {"en": "No notable insights.", "th": "ไม่พบข้อค้นพบที่สำคัญ"},
    "icat_quality": {"en": "Quality", "th": "คุณภาพ"},
    "icat_anomaly": {"en": "Anomaly", "th": "ความผิดปกติ"},
    "icat_text": {"en": "Text", "th": "ข้อความ"},
    "icat_structure": {"en": "Structure", "th": "โครงสร้าง"},
    "icat_target": {"en": "Target", "th": "เป้าหมาย"},
    "icat_distribution": {"en": "Distribution", "th": "การกระจาย"},
    "icat_timeseries": {"en": "Timeseries", "th": "อนุกรมเวลา"},
    "icat_business": {"en": "Business", "th": "เชิงธุรกิจ"},
    "quality_issues": {"en": "Data Quality Issues", "th": "ปัญหาคุณภาพข้อมูล"},
    # --- v0.6: cross-column insight engine (ข้อค้นพบจากการผสมคอลัมน์) ---
    "business_insights": {
        "en": "Cross-Column Insights",
        "th": "ข้อค้นพบจากการวิเคราะห์คอลัมน์ผสม",
    },
    "no_business_insights": {
        "en": "No notable cross-column insights found.",
        "th": "ไม่พบข้อค้นพบจากการผสมคอลัมน์ที่น่าสนใจ",
    },
    "pattern_outstanding": {"en": "Outstanding", "th": "กลุ่มโดดเด่น"},
    "pattern_attribution": {"en": "Attribution", "th": "สัดส่วนหลัก"},
    "pattern_comparison": {"en": "Comparison", "th": "เปรียบเทียบกลุ่ม"},
    "pattern_trend": {"en": "Trend", "th": "แนวโน้ม"},
    "evidence": {"en": "Evidence", "th": "หลักฐาน"},
    "top_segments": {"en": "Top Segments", "th": "กลุ่มเด่น"},
    "share": {"en": "Share", "th": "สัดส่วน"},
    "lift": {"en": "Lift", "th": "ส่วนต่าง"},
    "breakdown": {"en": "Breakdown", "th": "จัดกลุ่มตาม"},
    "measure": {"en": "Measure", "th": "ตัววัด"},
    "segment": {"en": "Segment", "th": "กลุ่ม"},
    "value": {"en": "Value", "th": "ค่า"},
    "column_details": {"en": "Column Details", "th": "รายละเอียดคอลัมน์"},
    "generated_by": {"en": "Generated by", "th": "สร้างโดย"},
    # --- ภาพรวม ---
    "rows": {"en": "Rows", "th": "จำนวนแถว"},
    "columns": {"en": "Columns", "th": "จำนวนคอลัมน์"},
    "missing_cells": {"en": "Missing Cells", "th": "เซลล์ที่ว่าง"},
    "missing_pct": {"en": "Missing (%)", "th": "ว่าง (%)"},
    "duplicate_rows": {"en": "Duplicate Rows", "th": "แถวซ้ำ"},
    "total_cells": {"en": "Total Cells", "th": "เซลล์ทั้งหมด"},
    "column_types": {"en": "Column Types", "th": "ประเภทคอลัมน์"},
    # --- ประเภทคอลัมน์ ---
    "type_numeric": {"en": "Numeric", "th": "ตัวเลข"},
    "type_categorical": {"en": "Categorical", "th": "หมวดหมู่"},
    "type_thai_text": {"en": "Thai Text", "th": "ข้อความภาษาไทย"},
    "type_english_text": {"en": "English Text", "th": "ข้อความภาษาอังกฤษ"},
    "type_mixed_text": {"en": "Mixed Text", "th": "ข้อความผสม"},
    "type_datetime": {"en": "Datetime", "th": "วันที่/เวลา"},
    "type_id": {"en": "Identifier", "th": "ตัวระบุ (ID)"},
    "type_empty": {"en": "Empty", "th": "ว่างเปล่า"},
    # --- ความรุนแรง ---
    "severity_critical": {"en": "Critical", "th": "วิกฤต"},
    "severity_warning": {"en": "Warning", "th": "เตือน"},
    "severity_info": {"en": "Info", "th": "ข้อมูล"},
    # --- รายละเอียดปัญหา ---
    "check": {"en": "Check", "th": "การตรวจสอบ"},
    "column": {"en": "Column", "th": "คอลัมน์"},
    "count": {"en": "Count", "th": "จำนวน"},
    "percentage": {"en": "Percentage", "th": "ร้อยละ"},
    "examples": {"en": "Examples", "th": "ตัวอย่าง"},
    "suggestion": {"en": "Suggestion", "th": "ข้อเสนอแนะ"},
    "no_issues": {"en": "No data quality issues detected.", "th": "ไม่พบปัญหาคุณภาพข้อมูล"},
    # --- anomalies ---
    "anomalies": {"en": "Anomalies", "th": "ความผิดปกติ"},
    "anomaly_type": {"en": "Type", "th": "ประเภท"},
    "no_anomalies": {"en": "No anomalies detected.", "th": "ไม่พบความผิดปกติ"},
    "antype_statistical": {"en": "Statistical", "th": "เชิงสถิติ"},
    "antype_text": {"en": "Text", "th": "ข้อความ"},
    "antype_encoding": {"en": "Encoding", "th": "การเข้ารหัส"},
    "antype_pattern": {"en": "Pattern", "th": "รูปแบบ"},
    "antype_categorical": {"en": "Categorical", "th": "หมวดหมู่"},
    # --- cleaning suggestions ---
    "cleaning_suggestions": {"en": "Cleaning Suggestions", "th": "คำแนะนำการทำความสะอาด"},
    "no_cleaning": {
        "en": "No cleaning operations needed.",
        "th": "ไม่มีคำแนะนำการทำความสะอาด",
    },
    "operation": {"en": "Operation", "th": "การดำเนินการ"},
    "rows_affected": {"en": "Rows Affected", "th": "แถวที่ได้รับผลกระทบ"},
    "before": {"en": "Before", "th": "ก่อน"},
    "after": {"en": "After", "th": "หลัง"},
    # --- cleaning diff (การทำความสะอาดที่ดำเนินการจริง) ---
    "cleaning_diff": {"en": "Data Cleaning Applied", "th": "การทำความสะอาด"},
    "total_cells_changed": {"en": "Total cells changed", "th": "รวมเซลล์ที่เปลี่ยน"},
    "most_impactful": {"en": "Most impactful", "th": "การดำเนินการที่มีผลมากสุด"},
    # --- text metrics ---
    "text_metrics": {"en": "Text Metrics", "th": "สถิติข้อความ"},
    "avg_char_length": {"en": "Avg. Characters", "th": "จำนวนอักขระเฉลี่ย"},
    "avg_token_length": {"en": "Avg. Tokens", "th": "จำนวนคำเฉลี่ย"},
    "median_char_length": {"en": "Median Characters", "th": "จำนวนอักขระมัธยฐาน"},
    "min_char_length": {"en": "Min Characters", "th": "อักขระน้อยสุด"},
    "max_char_length": {"en": "Max Characters", "th": "อักขระมากสุด"},
    "total_tokens": {"en": "Total Tokens", "th": "คำทั้งหมด"},
    "unique_tokens": {"en": "Unique Tokens", "th": "คำไม่ซ้ำ"},
    "top_tokens": {"en": "Top Tokens", "th": "คำที่พบบ่อย"},
    "top_bigrams": {"en": "Top Bigrams", "th": "คู่คำที่พบบ่อย"},
    "top_trigrams": {"en": "Top Trigrams", "th": "สามคำที่พบบ่อย"},
    "wordcloud": {"en": "Word Cloud", "th": "เมฆคำ"},
    "length_distribution": {"en": "Length Distribution", "th": "การกระจายความยาว"},
    # --- กราฟชุดใหม่ (การกระจาย/สหสัมพันธ์/ค่าว่าง) ---
    "distributions_correlations": {
        "en": "Distributions & Correlations",
        "th": "การกระจายและสหสัมพันธ์",
    },
    "correlation_heatmap": {"en": "Correlation Heatmap", "th": "แผนภาพความร้อนสหสัมพันธ์"},
    "scatter_matrix": {"en": "Scatter Matrix", "th": "แผนภาพการกระจายแบบเมทริกซ์"},
    "boxplot": {"en": "Box Plot", "th": "แผนภาพกล่อง"},
    "violinplot": {"en": "Violin Plot", "th": "แผนภาพไวโอลิน"},
    "distribution": {"en": "Distribution", "th": "การแจกแจง"},
    "value_counts": {"en": "Value Counts", "th": "ความถี่ของค่า"},
    "missing_data": {"en": "Missing Data", "th": "ข้อมูลที่ขาดหาย"},
    "missing_matrix": {"en": "Missing Value Matrix", "th": "เมทริกซ์ค่าที่ขาดหาย"},
    "missing_heatmap": {"en": "Nullity Correlation", "th": "สหสัมพันธ์การขาดหาย"},
    "no_missing": {"en": "No missing data.", "th": "ไม่มีข้อมูลที่ขาดหาย"},
    "explanation": {"en": "What was fixed", "th": "สิ่งที่แก้ไข"},
    "engine_used": {"en": "Tokenizer Engine", "th": "เครื่องมือตัดคำ"},
    "sampled_note": {"en": "Sampled", "th": "สุ่มตัวอย่าง"},
    "of": {"en": "of", "th": "จาก"},
    # --- สถิติพื้นฐาน ---
    "basic_stats": {"en": "Basic Statistics", "th": "สถิติพื้นฐาน"},
    "non_null": {"en": "Non-null", "th": "ไม่ว่าง"},
    "unique": {"en": "Unique", "th": "ไม่ซ้ำ"},
    "mean": {"en": "Mean", "th": "ค่าเฉลี่ย"},
    "std": {"en": "Std. Dev.", "th": "ส่วนเบี่ยงเบนมาตรฐาน"},
    "min": {"en": "Min", "th": "ต่ำสุด"},
    "max": {"en": "Max", "th": "สูงสุด"},
    "top_values": {"en": "Top Values", "th": "ค่าที่พบบ่อย"},
    "more_columns_note": {
        "en": "For deeper numeric/categorical profiling, see ydata-profiling.",
        "th": "สำหรับการวิเคราะห์ตัวเลข/หมวดหมู่เชิงลึก แนะนำ ydata-profiling",
    },
    # --- named entities (NER) ---
    "named_entities": {"en": "Named Entities (NER)", "th": "ชื่อเฉพาะ (NER)"},
    "no_entities": {"en": "No named entities detected.", "th": "ไม่พบชื่อเฉพาะ"},
    "entity_type": {"en": "Entity Type", "th": "ประเภทชื่อเฉพาะ"},
    "top_entities": {"en": "Top Entities", "th": "ชื่อเฉพาะที่พบบ่อย"},
    "ner_engine": {"en": "NER Engine", "th": "เครื่องมือ NER"},
    "total_entities": {"en": "Total Entities", "th": "ชื่อเฉพาะทั้งหมด"},
    # --- target analysis ---
    "target_analysis": {"en": "Target Analysis", "th": "การวิเคราะห์ตัวแปรเป้าหมาย"},
    "target_column": {"en": "Target Column", "th": "คอลัมน์เป้าหมาย"},
    "association": {"en": "Association", "th": "ความสัมพันธ์"},
    "assoc_correlation": {"en": "Correlation (Pearson)", "th": "สหสัมพันธ์ (Pearson)"},
    "assoc_chi_square": {"en": "Chi-square", "th": "ไคสแควร์"},
    "assoc_anova": {"en": "ANOVA (F-test)", "th": "ANOVA (F-test)"},
    "score": {"en": "Score", "th": "คะแนน"},
    "p_value": {"en": "p-value", "th": "ค่า p"},
    "no_target": {"en": "No target column specified.", "th": "ไม่ได้ระบุคอลัมน์เป้าหมาย"},
    # --- timeseries (อนุกรมเวลา) ---
    "timeseries": {"en": "Timeseries Analysis", "th": "การวิเคราะห์อนุกรมเวลา"},
    "no_timeseries": {
        "en": "No timeseries columns detected.",
        "th": "ไม่พบคอลัมน์อนุกรมเวลา",
    },
    "ts_frequency": {"en": "Frequency", "th": "ความถี่"},
    "ts_trend": {"en": "Trend", "th": "แนวโน้ม"},
    "ts_seasonality": {"en": "Seasonality", "th": "ฤดูกาล"},
    "ts_seasonal_period": {"en": "Seasonal Period", "th": "คาบฤดูกาล"},
    "ts_gaps": {"en": "Time Gaps", "th": "ช่องว่างเวลา"},
    "ts_anomalies": {"en": "Spikes / Anomalies", "th": "ค่าผิดปกติเฉพาะช่วง"},
    "ts_autocorr": {"en": "Autocorrelation (lag-1)", "th": "สหสัมพันธ์ในตัว (lag-1)"},
    "ts_engine": {"en": "Decomposition Engine", "th": "เครื่องมือแยกองค์ประกอบ"},
    "ts_timeseries_plot": {"en": "Timeseries (with trend)", "th": "อนุกรมเวลา (พร้อมแนวโน้ม)"},
    "ts_decomposition": {"en": "STL Decomposition", "th": "การแยกองค์ประกอบ (STL)"},
    "ts_acf_plot": {"en": "Autocorrelation (ACF)", "th": "ฟังก์ชันสหสัมพันธ์ในตัว (ACF)"},
    "ts_has": {"en": "Yes", "th": "มี"},
    "ts_none": {"en": "No", "th": "ไม่มี"},
    # --- progress (ข้อความความคืบหน้าระหว่างประมวลผล) ---
    "prog_read": {"en": "Reading file...", "th": "อ่านไฟล์..."},
    "prog_detect": {"en": "Detecting column types...", "th": "ตรวจจับประเภทคอลัมน์..."},
    "prog_quality": {"en": "Checking data quality...", "th": "ตรวจสอบคุณภาพ..."},
    "prog_anomaly": {"en": "Detecting anomalies...", "th": "ตรวจความผิดปกติ..."},
    "prog_clean": {"en": "Cleaning text...", "th": "ทำความสะอาด..."},
    "prog_timeseries": {"en": "Analyzing timeseries...", "th": "วิเคราะห์อนุกรมเวลา..."},
    "prog_insights_engine": {
        "en": "Discovering cross-column insights...",
        "th": "ค้นหาข้อค้นพบจากคอลัมน์ผสม...",
    },
    "prog_insights": {"en": "Summarizing insights...", "th": "สรุปข้อค้นพบ..."},
    "prog_charts": {"en": "Building charts...", "th": "สร้างกราฟ..."},
    "prog_report": {"en": "Generating report...", "th": "สร้างรายงาน..."},
    # --- คอลัมน์ประเภทเบอร์โทร (เพิ่มเติม) ---
    "type_phone_number": {"en": "Phone Number", "th": "เบอร์โทรศัพท์"},
    # --- สรุปผลแบบตาราง (CLI) ---
    "summary_table": {"en": "Summary", "th": "สรุปผล"},
    "sampled_rows": {"en": "Sampled rows", "th": "สุ่มตัวอย่างแถว"},
    # --- v0.5: รายงานชุดข้อมูลหลายตาราง (dataset / schema) ---
    "dataset_report_title": {"en": "ThaiEDA Dataset Report", "th": "รายงานชุดข้อมูล ThaiEDA"},
    "schema_overview": {"en": "Schema Overview", "th": "ภาพรวม Schema"},
    "tables": {"en": "Tables", "th": "ตาราง"},
    "relationships": {"en": "Relationships", "th": "ความสัมพันธ์"},
    "er_diagram": {"en": "ER Diagram", "th": "แผนผังความสัมพันธ์"},
    "orphan_findings": {"en": "Orphan Records", "th": "ข้อมูลกำพร้า"},
    "from_table": {"en": "From", "th": "จากตาราง"},
    "to_table": {"en": "To", "th": "ไปตาราง"},
    "overlap": {"en": "Overlap", "th": "ครอบคลุม"},
    "orphan_count": {"en": "Orphans", "th": "กำพร้า"},
    "cardinality": {"en": "Cardinality", "th": "ความสัมพันธ์แบบ"},
    "confidence": {"en": "Confidence", "th": "ความมั่นใจ"},
    "key_candidates": {"en": "Key Columns", "th": "คอลัมน์คีย์"},
    "match_method": {"en": "Match", "th": "วิธีจับคู่"},
    "table_count": {"en": "Tables", "th": "จำนวนตาราง"},
    "relationship_count": {"en": "Relationships", "th": "จำนวนความสัมพันธ์"},
    "orphan_count_total": {"en": "Orphan Findings", "th": "จุดที่พบข้อมูลกำพร้า"},
    "row_count": {"en": "Rows", "th": "จำนวนแถว"},
    "no_relationships": {
        "en": "No relationships detected between tables.",
        "th": "ไม่พบความสัมพันธ์ระหว่างตาราง",
    },
    "no_orphans": {"en": "No orphan records found.", "th": "ไม่พบข้อมูลกำพร้า"},
    "validated": {"en": "Validated", "th": "ตรวจค่าแล้ว"},
    "name_only": {"en": "Name only", "th": "ชื่อเท่านั้น"},
    "primary_key": {"en": "Primary Key", "th": "คีย์หลัก (PK)"},
    "foreign_key": {"en": "Foreign Key", "th": "คีย์อ้างอิง (FK)"},
    # --- progress (dataset) ---
    "prog_dataset_match": {
        "en": "Matching relationships...",
        "th": "จับคู่ความสัมพันธ์...",
    },
    # --- v1.1: enhanced report — insights-driven, actionable ---
    "data_health": {"en": "Data Health", "th": "สุขภาพข้อมูล"},
    "priority_actions": {"en": "Priority Actions", "th": "สิ่งที่ควรทำก่อน"},
    "recommended_actions": {
        "en": "Recommended Actions",
        "th": "ควรทำอะไรต่อ",
    },
    "so_what": {"en": "So What?", "th": "แล้วไงต่อ?"},
    "at_a_glance": {"en": "At a Glance", "th": "สรุปประเด็น"},
    "quick_wins": {"en": "Quick Wins", "th": "แก้เร็วได้"},
    "needs_attention": {
        "en": "Needs Attention",
        "th": "ต้องดูแล",
    },
    "good_shape": {"en": "Good Shape", "th": "ไม่มีปัญหา"},
    "how_to_read": {
        "en": "How to read this report",
        "th": "วิธีอ่านรายงานนี้",
    },
    "how_to_read_desc": {
        "en": ("Start with Executive Summary, then Priority Actions, then column details."),
        "th": "เริ่มจากบทสรุปด้านบน แล้วดูสิ่งที่ควรทำก่อน แล้วค่อยเจาะรายละเอียดแต่ละคอลัมน์",
    },
    "show_details": {"en": "Show details", "th": "ดูเพิ่มเติม"},
    "hide_details": {"en": "Hide details", "th": "ซ่อน"},
    "show_more_insights": {"en": "Show more insights", "th": "แสดงข้อค้นพบเพิ่มเติม"},
    "showing_top_of": {"en": "showing top", "th": "แสดง"},
    "of_total": {"en": "of", "th": "จาก"},
    "no_priority_actions": {
        "en": "No urgent actions needed.",
        "th": "ไม่มีสิ่งเร่งด่วนที่ต้องทำ",
    },
    "top_columns_to_watch": {
        "en": "Columns to Watch",
        "th": "คอลัมน์ที่ควรระวัง",
    },
    "chart_insight": {
        "en": "What this chart tells us",
        "th": "กราฟนี้บอกอะไรเรา",
    },
}


def label(key: str, lang: str | None = None) -> str:
    """คืนค่า label ตาม key และภาษา (ถ้าไม่ระบุภาษา ใช้ภาษาปัจจุบัน).

    หากไม่พบ key หรือภาษา จะคืนค่า key เดิมเพื่อให้ระบบไม่ล่ม
    """
    use_lang = lang if lang is not None else _current_language
    entry = LABELS.get(key)
    if entry is None:
        return key
    return entry.get(use_lang) or entry.get("en") or key


def set_language(lang: str) -> None:
    """ตั้งค่าภาษาเริ่มต้นแบบ global (รองรับ 'th' หรือ 'en')."""
    global _current_language
    if lang not in ("th", "en"):
        raise ValueError(f"Unsupported language {lang!r}; expected 'th' or 'en'.")
    _current_language = lang


def get_language() -> str:
    """คืนค่าภาษาปัจจุบัน."""
    return _current_language


__all__ = ["LABELS", "TECHNICAL_TO_PLAIN", "label", "set_language", "get_language"]
