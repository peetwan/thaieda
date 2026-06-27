"""Validation helpers shared across ThaiEDA entry points."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def duplicate_column_names(columns: Iterable[object]) -> list[str]:
    """Return duplicate column names after ThaiEDA's string-normalization step."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in columns:
        name = str(column)
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def ensure_unique_column_names(df: pd.DataFrame, *, context: str = "DataFrame") -> None:
    """Fail loudly when duplicate column names would make ``df[col]`` ambiguous."""
    duplicates = duplicate_column_names(df.columns)
    if not duplicates:
        return
    duplicate_text = ", ".join(repr(name) for name in duplicates)
    raise ValueError(
        f"{context} requires unique column names; duplicate column name(s): "
        f"{duplicate_text}. Rename duplicate columns before running ThaiEDA."
    )
