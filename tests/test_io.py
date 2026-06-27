"""ทดสอบ thaieda.io — read_data, detect_encoding, detect_format."""

from __future__ import annotations

import contextlib
import json

import pandas as pd
import pytest

from thaieda.io import detect_encoding, detect_format, read_data


def _has_parquet_engine() -> bool:
    with contextlib.suppress(ImportError):
        import pyarrow  # noqa: F401

        return True
    with contextlib.suppress(ImportError):
        import fastparquet  # noqa: F401

        return True
    return False


# ------------------------------------------------------------- detect_format
def test_detect_format_csv(tmp_path):
    assert detect_format(tmp_path / "data.csv") == "csv"


def test_detect_format_tsv(tmp_path):
    assert detect_format(tmp_path / "data.tsv") == "tsv"


def test_detect_format_json(tmp_path):
    assert detect_format(tmp_path / "data.json") == "json"


def test_detect_format_jsonl(tmp_path):
    assert detect_format(tmp_path / "data.jsonl") == "jsonl"
    assert detect_format(tmp_path / "data.ndjson") == "jsonl"


def test_detect_format_excel_and_parquet(tmp_path):
    assert detect_format(tmp_path / "data.xlsx") == "excel"
    assert detect_format(tmp_path / "data.xls") == "excel"
    assert detect_format(tmp_path / "data.parquet") == "parquet"


def test_detect_format_unknown_defaults_csv(tmp_path):
    # นามสกุลไม่รู้จัก/ไม่มีนามสกุล → เดาเป็น csv (read_data จะถอยไป json ให้เอง)
    assert detect_format(tmp_path / "data.dat") == "csv"
    assert detect_format(tmp_path / "data") == "csv"


# ------------------------------------------------------------- detect_encoding
def test_detect_encoding_utf8(tmp_path):
    f = tmp_path / "u.csv"
    f.write_text("ชื่อ,ราคา\nสมชาย,100\n", encoding="utf-8")
    assert detect_encoding(f) == "utf-8"


def test_detect_encoding_tis620(tmp_path):
    # ไฟล์ไทยเก่า encode ด้วย tis-620 — utf-8 ต้องถอดไม่ได้ จึงถอยไป tis-620
    f = tmp_path / "legacy.csv"
    f.write_bytes("สวัสดีครับ ทดสอบภาษาไทย".encode("tis-620"))
    assert detect_encoding(f) == "tis-620"


def test_detect_encoding_empty_file(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_bytes(b"")
    assert detect_encoding(f) == "utf-8"


def test_detect_encoding_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_encoding(tmp_path / "nope.csv")


# ------------------------------------------------------------- read_data CSV
def test_read_data_csv(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nสมชาย,30\nสมหญิง,25\n", encoding="utf-8")
    df = read_data(f)
    assert list(df.columns) == ["name", "age"]
    assert len(df) == 2
    assert df["name"].iloc[0] == "สมชาย"


def test_read_data_csv_tis620_autodetect(tmp_path):
    f = tmp_path / "legacy.csv"
    f.write_bytes("ชื่อ,เมือง\nสมชาย,กรุงเทพ\n".encode("tis-620"))
    df = read_data(f)  # encoding=auto ต้องเดา tis-620 ได้
    assert df["ชื่อ"].iloc[0] == "สมชาย"


def test_read_data_csv_explicit_encoding(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    df = read_data(f, format="csv", encoding="utf-8")
    assert len(df) == 1


def test_read_data_tsv_explicit_format(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("a\tb\n1\t2\n", encoding="utf-8")
    df = read_data(f, format="tsv", encoding="utf-8")
    assert list(df.columns) == ["a", "b"]
    assert df["b"].iloc[0] == 2


def test_read_data_excel(tmp_path):
    f = tmp_path / "data.xlsx"
    pd.DataFrame({"a": [1], "b": [2]}).to_excel(f, index=False)
    df = read_data(f)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 1


@pytest.mark.skipif(
    not _has_parquet_engine(),
    reason="ต้องติดตั้ง pyarrow หรือ fastparquet สำหรับ Parquet",
)
def test_read_data_parquet(tmp_path):
    f = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(f, index=False)
    df = read_data(f)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 1


# ------------------------------------------------------------- read_data JSON
def test_read_data_json_records(tmp_path):
    f = tmp_path / "data.json"
    f.write_text(
        json.dumps([{"a": 1, "b": "ก"}, {"a": 2, "b": "ข"}], ensure_ascii=False),
        encoding="utf-8",
    )
    df = read_data(f)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
    assert df["b"].iloc[1] == "ข"


def test_read_data_jsonl(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text(
        '{"a": 1, "b": "ก"}\n{"a": 2, "b": "ข"}\n',
        encoding="utf-8",
    )
    df = read_data(f)
    assert len(df) == 2
    assert df["a"].tolist() == [1, 2]


def test_read_data_json_explicit_format(tmp_path):
    f = tmp_path / "weird.txt"
    f.write_text(json.dumps([{"x": 10}]), encoding="utf-8")
    df = read_data(f, format="json")
    assert df["x"].iloc[0] == 10


# ------------------------------------------------------------- read_data errors
def test_read_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_data(tmp_path / "nope.csv")


def test_read_data_invalid_format(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_data(f, format="xml")


def test_read_data_no_extension_fallback_to_json(tmp_path):
    # ไม่มีนามสกุล + เนื้อหาเป็น JSON → detect_format เดา csv แต่ read_data ถอยไป json สำเร็จ
    f = tmp_path / "payload"
    f.write_text(json.dumps([{"k": "v"}]), encoding="utf-8")
    df = read_data(f)
    assert df["k"].iloc[0] == "v"


def test_read_data_returns_dataframe(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a\n1\n", encoding="utf-8")
    assert isinstance(read_data(f), pd.DataFrame)


def test_read_data_semicolon_csv(tmp_path):
    """UCI wine-quality ใช้ semicolon เป็น delimiter."""
    f = tmp_path / "wine.csv"
    f.write_text(
        '"fixed acidity";"quality"\n7.4;5\n7.8;6\n',
        encoding="utf-8",
    )
    df = read_data(f)
    assert list(df.columns) == ["fixed acidity", "quality"]
    assert len(df) == 2


def test_read_data_headerless_numeric(tmp_path):
    """UCI .data ไม่มี header — คอลัมน์สุดท้ายตั้งชื่อ target."""
    f = tmp_path / "heart.data"
    f.write_text(
        "63.0,1.0,1.0,145.0,0\n"
        "67.0,1.0,4.0,160.0,2\n",
        encoding="utf-8",
    )
    df = read_data(f)
    assert list(df.columns) == ["col_0", "col_1", "col_2", "col_3", "target"]
    assert df["target"].tolist() == [0, 2]
