"""ThaiEDA — AutoEDA สำหรับข้อมูลภาษาไทย.

Exploratory data analysis that speaks Thai.
"""

__version__ = "0.1.0"
__all__ = ["profile", "ProfileReport", "__version__"]


def __getattr__(name: str):
    """Lazy import เพื่อให้ core ไม่ต้องโหลด dependencies หนักทั้งหมด."""
    if name == "profile":
        from thaieda.report import profile

        return profile
    if name == "ProfileReport":
        from thaieda.report import ProfileReport

        return ProfileReport
    raise AttributeError(f"module 'thaieda' has no attribute {name!r}")
