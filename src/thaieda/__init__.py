"""ThaiEDA — AutoEDA สำหรับข้อมูลภาษาไทย.

Exploratory data analysis that speaks Thai.
"""

__version__ = "0.2.0"
__all__ = [
    "profile",
    "ProfileReport",
    "extract_entities",
    "analyze_target",
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
    raise AttributeError(f"module 'thaieda' has no attribute {name!r}")
