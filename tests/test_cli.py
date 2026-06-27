"""ทดสอบ thaieda.cli — profile, run, clean commands + JSON input."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from thaieda.cli import main


@pytest.fixture
def csv_file(tmp_path):
    f = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "review": ["อาหารอร่อยมาก", "ร้านนี้ดี​แต่แพง", "สวัสดีครับ", "ดีมากเลย"],
            "rating": [5, 3, 4, 5],
            "year": [2567, 2024, 2568, 2023],
            "price": ["๑๒๐", "150", "๒๐๐", "300"],
        }
    )
    df.to_csv(f, index=False, encoding="utf-8")
    return f


@pytest.fixture
def json_file(tmp_path):
    f = tmp_path / "data.json"
    records = [
        {"name": "สมชาย", "score": 80},
        {"name": "สมหญิง", "score": 95},
        {"name": "สมศรี", "score": 70},
    ]
    f.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return f


@pytest.fixture
def jsonl_file(tmp_path):
    f = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"name": "ก", "v": 1}, ensure_ascii=False),
        json.dumps({"name": "ข", "v": 2}, ensure_ascii=False),
    ]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


@pytest.fixture
def xlsx_file(tmp_path):
    f = tmp_path / "data.xlsx"
    pd.DataFrame({"name": ["alice", "bob"], "score": [10, 20]}).to_excel(f, index=False)
    return f


# ------------------------------------------------------------- version / help
def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "thaieda" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "thaieda" in capsys.readouterr().out


# ------------------------------------------------------------- profile
def test_profile_csv(csv_file, tmp_path, capsys):
    out = tmp_path / "report.html"
    rc = main(["profile", str(csv_file), "-o", str(out), "--no-charts"])
    assert rc == 0
    assert out.is_file()
    captured = capsys.readouterr().out
    assert "วิเคราะห์ไฟล์" in captured
    assert "ข้อค้นพบ" in captured  # insight highlights printed


def test_profile_missing_file(tmp_path, capsys):
    rc = main(["profile", str(tmp_path / "nope.csv")])
    assert rc == 2
    assert "ไม่พบไฟล์" in capsys.readouterr().err


def test_profile_invalid_target(csv_file, capsys):
    rc = main(["profile", str(csv_file), "--no-charts", "--target", "nonexistent"])
    assert rc == 2
    assert "ไม่พบคอลัมน์เป้าหมาย" in capsys.readouterr().err


def test_profile_with_clean_flag(csv_file, tmp_path):
    out = tmp_path / "report.html"
    rc = main(["profile", str(csv_file), "-o", str(out), "--no-charts", "--clean"])
    assert rc == 0
    html = out.read_text(encoding="utf-8")
    assert "การทำความสะอาด" in html


def test_profile_xlsx_explicit_format(xlsx_file, tmp_path):
    out = tmp_path / "report.html"
    rc = main(["profile", str(xlsx_file), "-o", str(out), "--format", "excel", "--no-charts"])
    assert rc == 0
    assert out.is_file()


# ------------------------------------------------------------- run
def test_run_csv(csv_file, tmp_path, capsys):
    out = tmp_path / "report.html"
    cleaned = tmp_path / "cleaned.csv"
    rc = main(
        [
            "run",
            str(csv_file),
            "-o",
            str(out),
            "--cleaned-output",
            str(cleaned),
            "--no-charts",
        ]
    )
    assert rc == 0
    assert out.is_file()
    assert cleaned.is_file()
    captured = capsys.readouterr().out
    assert "ประมวลผลไฟล์" in captured
    assert "บันทึกข้อมูลที่สะอาดแล้ว" in captured
    # ข้อมูลที่สะอาดแล้ว: เลขไทยถูกแปลงเป็นอารบิก
    cleaned_df = pd.read_csv(cleaned)
    assert "120" in cleaned_df["price"].astype(str).tolist()


def test_run_default_cleaned_output(csv_file, tmp_path):
    out = tmp_path / "report.html"
    rc = main(["run", str(csv_file), "-o", str(out), "--no-charts"])
    assert rc == 0
    # ค่าเริ่มต้น: <input>.cleaned.csv
    assert (tmp_path / "data.cleaned.csv").is_file()


def test_run_no_clean(csv_file, tmp_path):
    out = tmp_path / "report.html"
    rc = main(["run", str(csv_file), "-o", str(out), "--no-charts", "--no-clean"])
    assert rc == 0
    assert out.is_file()
    # --no-clean → ไม่สร้างไฟล์ที่สะอาดแล้ว
    assert not (tmp_path / "data.cleaned.csv").is_file()


def test_run_json_input(json_file, tmp_path, capsys):
    out = tmp_path / "report.html"
    rc = main(["run", str(json_file), "-o", str(out), "--no-charts"])
    assert rc == 0
    assert out.is_file()
    assert "ประมวลผลไฟล์" in capsys.readouterr().out


def test_run_with_target(csv_file, tmp_path):
    out = tmp_path / "report.html"
    rc = main(["run", str(csv_file), "-o", str(out), "--no-charts", "--target", "rating"])
    assert rc == 0
    assert out.is_file()


def test_run_json_export(csv_file, tmp_path):
    out = tmp_path / "report.html"
    js = tmp_path / "report.json"
    rc = main(["run", str(csv_file), "-o", str(out), "--no-charts", "--json", str(js)])
    assert rc == 0
    parsed = json.loads(js.read_text(encoding="utf-8"))
    assert "insights" in parsed


# ------------------------------------------------------------- clean
def test_clean_csv(csv_file, tmp_path, capsys):
    out = tmp_path / "out.csv"
    rc = main(["clean", str(csv_file), "-o", str(out)])
    assert rc == 0
    assert out.is_file()
    assert "ทำความสะอาดไฟล์" in capsys.readouterr().out


def test_clean_json_input(json_file, tmp_path):
    out = tmp_path / "out.csv"
    rc = main(["clean", str(json_file), "-o", str(out)])
    assert rc == 0
    assert out.is_file()


def test_clean_jsonl_input(jsonl_file, tmp_path):
    out = tmp_path / "out.csv"
    rc = main(["clean", str(jsonl_file), "-o", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert len(df) == 2
