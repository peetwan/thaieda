"""ThaiEDA — AutoEDA สำหรับข้อมูลภาษาไทย.

Exploratory data analysis that speaks Thai.
"""

__version__ = "0.6.0"
__all__ = [
    "profile",
    "ProfileReport",
    "extract_entities",
    "analyze_target",
    "generate_insights",
    "Insight",
    "InsightSummary",
    "discover_insights",
    "InsightCard",
    "InsightEngineResult",
    "Perspective",
    "analyze_timeseries",
    "analyze_dataframe_timeseries",
    "detect_timeseries_columns",
    "TimeseriesResult",
    "TimeseriesComponent",
    "read_data",
    "detect_encoding",
    "detect_format",
    "profile_dataset",
    "DatasetProfile",
    "Relationship",
    "KeyCandidate",
    "TableProfile",
    "DatasetReport",
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
    if name in (
        "discover_insights",
        "InsightCard",
        "InsightEngineResult",
        "Perspective",
    ):
        import thaieda.insight_engine as _insight_engine

        return getattr(_insight_engine, name)
    if name in (
        "analyze_timeseries",
        "analyze_dataframe_timeseries",
        "detect_timeseries_columns",
        "TimeseriesResult",
        "TimeseriesComponent",
    ):
        import thaieda.timeseries as _timeseries

        return getattr(_timeseries, name)
    if name in ("read_data", "detect_encoding", "detect_format"):
        import thaieda.io as _io

        return getattr(_io, name)
    if name in (
        "profile_dataset",
        "DatasetProfile",
        "Relationship",
        "KeyCandidate",
        "TableProfile",
    ):
        import thaieda.schema as _schema

        return getattr(_schema, name)
    if name == "DatasetReport":
        from thaieda.report._dataset import DatasetReport

        return DatasetReport
    raise AttributeError(f"module 'thaieda' has no attribute {name!r}")
