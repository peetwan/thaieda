"""ทดสอบ one-liner CLI — ``thaieda data.csv`` ไม่ต้องระบุ subcommand."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from thaieda.cli import _guess_target_column, main


@pytest.fixture
def ml_csv(tmp_path):
    """CSV จำลอง ML tabular พร้อมคอลัมน์ target."""
    f = tmp_path / "ads.csv"
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": ["x", "y", "x", "y", "x", "y"],
            "clicked": [0, 1, 0, 1, 0, 1],
        }
    )
    df.to_csv(f, index=False, encoding="utf-8")
    return f


@pytest.fixture
def simple_csv(tmp_path):
    f = tmp_path / "data.csv"
    pd.DataFrame({"name": ["a", "b"], "value": [1, 2]}).to_csv(f, index=False)
    return f


def test_guess_target_column():
    assert _guess_target_column(["feature", "Clicked", "id"]) == "Clicked"
    assert _guess_target_column(["feature_a", "feature_b", "feature_c"]) is None
    assert _guess_target_column(["label"]) == "label"
    assert _guess_target_column(["wine_quality"]) == "wine_quality"
    assert _guess_target_column(["annual_income"]) == "annual_income"
    assert _guess_target_column(["Quality"]) == "Quality"


def test_oneliner_creates_blueprint_report(ml_csv, capsys):
    rc = main([str(ml_csv), "--no-charts"])
    assert rc == 0
    out_html = ml_csv.with_name("ads-report.html")
    assert out_html.is_file()
    html = out_html.read_text(encoding="utf-8")
    assert 'lang="th"' in html
    assert "บทสรุปผู้บริหาร" in html
    assert "แผนสร้างโมเดล" in html
    captured = capsys.readouterr().out
    assert "บันทึกรายงาน HTML" in captured
    assert str(out_html.resolve()) in captured


def test_oneliner_auto_target(ml_csv, capsys):
    rc = main([str(ml_csv), "--no-charts"])
    assert rc == 0
    assert "Target: clicked (auto)" in capsys.readouterr().out


def test_oneliner_explicit_output(simple_csv, tmp_path, capsys):
    out = tmp_path / "custom.html"
    rc = main([str(simple_csv), "-o", str(out), "--no-charts"])
    assert rc == 0
    assert out.is_file()
    assert "custom.html" in capsys.readouterr().out


def test_oneliner_explore_mode(simple_csv):
    rc = main([str(simple_csv), "--explore", "--no-charts"])
    assert rc == 0
    html = simple_csv.with_name("data-report.html").read_text(encoding="utf-8")
    assert "ภาพรวม</button>" in html


def test_oneliner_target_from_flag(tmp_path, capsys):
    f = tmp_path / "labeled.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]}).to_csv(f, index=False)
    rc = main([str(f), "--no-charts", "-t", "a"])
    assert rc == 0
    assert "Target: a (จาก --target)" in capsys.readouterr().out


def test_oneliner_no_interactive_skips_prompt(monkeypatch, tmp_path):
    f = tmp_path / "plain.csv"
    pd.DataFrame({"feature_x": [1, 2], "feature_y": [3, 4], "result": [0, 1]}).to_csv(
        f, index=False
    )
    prompts: list[str] = []

    def fake_input(_msg: str) -> str:
        prompts.append("called")
        return "3"

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    rc = main([str(f), "--no-charts", "-y"])
    assert rc == 0
    assert prompts == []
    assert f.with_name("plain-report.html").is_file()


def test_oneliner_interactive_target(monkeypatch, tmp_path, capsys):
    f = tmp_path / "plain.csv"
    pd.DataFrame({"feature_x": [1, 2], "feature_y": [3, 4], "result": [0, 1]}).to_csv(
        f, index=False
    )
    monkeypatch.setattr("builtins.input", lambda _msg: "3")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    rc = main([str(f), "--no-charts"])
    assert rc == 0
    assert "Target: result (interactive)" in capsys.readouterr().out


def test_oneliner_target_override(simple_csv, tmp_path):
    f = tmp_path / "labeled.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]}).to_csv(f, index=False)
    rc = main([str(f), "--no-charts", "-t", "a"])
    assert rc == 0
    html = f.with_name("labeled-report.html").read_text(encoding="utf-8")
    assert "แผนสร้างโมเดล" in html


def test_oneliner_invalid_target(ml_csv, capsys):
    rc = main([str(ml_csv), "--no-charts", "--target", "missing"])
    assert rc == 2
    assert "ไม่พบคอลัมน์เป้าหมาย" in capsys.readouterr().err


def test_oneliner_missing_file(tmp_path, capsys):
    rc = main([str(tmp_path / "nope.csv")])
    assert rc == 2
    assert "ไม่พบไฟล์" in capsys.readouterr().err


def test_oneliner_no_clean(simple_csv):
    rc = main([str(simple_csv), "--no-charts", "--no-clean"])
    assert rc == 0
    assert simple_csv.with_name("data-report.html").is_file()


def test_oneliner_quiet(ml_csv, capsys):
    rc = main([str(ml_csv), "--no-charts", "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("ads-report.html")
    assert "วิเคราะห์ไฟล์" not in out


def test_subcommands_still_work(simple_csv, tmp_path):
    out = tmp_path / "profile.html"
    rc = main(["profile", str(simple_csv), "-o", str(out), "--no-charts"])
    assert rc == 0
    assert out.is_file()


def test_columns_preview(ml_csv, capsys):
    rc = main([str(ml_csv), "--columns"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Columns (3):" in out
    assert "clicked" in out
    assert "★ likely target" in out
    assert not ml_csv.with_name("ads-report.html").exists()


def test_columns_preview_alias(ml_csv, capsys):
    rc = main([str(ml_csv), "--preview"])
    assert rc == 0
    assert "Columns (3):" in capsys.readouterr().out


def test_no_target_message(simple_csv, capsys):
    rc = main([str(simple_csv), "--no-charts", "-y"])
    assert rc == 0
    assert "Target: (ไม่ระบุ — ไม่มีแผนสร้างโมเดล)" in capsys.readouterr().out
