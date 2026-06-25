"""ThaiEDA — AutoEDA สำหรับข้อมูลภาษาไทย.

Exploratory data analysis that speaks Thai.
"""

__version__ = "0.3.0"
__all__ = [
    "profile",
    "ProfileReport",
    "extract_entities",
    "analyze_target",
    "generate_insights",
    "Insight",
    "InsightSummary",
    "read_data",
    "detect_encoding",
    "detect_format",
    "__version__",
]


def __getattr__(name: str):
    """Lazy import เพื่อให้ core ไม่ต้องโหลด dependencies หนักทั้งหมด."""
    if name == "profile":
        from thaieda.report import profile

        return profile
    if name == "ProfileReport":
        from thaieda.report import ProfileReport

        return ProfileReport
    if name in ("extract_entities", "NERResult", "NEREntity", "ner_available"):
        import thaieda.ner as _ner

        return getattr(_ner, name)
    if name in ("analyze_target", "TargetAssociation"):
        import thaieda.analysis as _analysis

        return getattr(_analysis, name)
    if name in ("generate_insights", "Insight", "InsightSummary"):
        import thaieda.insight as _insight

        return getattr(_insight, name)
    if name in ("read_data", "detect_encoding", "detect_format"):
        import thaieda.io as _io

        return getattr(_io, name)
    raise AttributeError(f"module 'thaieda' has no attribute {name!r}")
