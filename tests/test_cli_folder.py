"""ทดสอบ one-liner batch โฟลเดอร์ — ``thaieda .`` / ``thaieda folder/``."""

from __future__ import annotations

import pandas as pd
import pytest

from thaieda.cli import main


@pytest.fixture
def batch_dir(tmp_path):
    """โฟลเดอร์ที่มี CSV 2 ไฟล์."""
    d = tmp_path / "data"
    d.mkdir()
    pd.DataFrame({"x": [1, 2], "clicked": [0, 1]}).to_csv(d / "a.csv", index=False)
    pd.DataFrame({"x": [3, 4], "clicked": [1, 0]}).to_csv(d / "b.csv", index=False)
    return d


def test_folder_batch_two_csvs(batch_dir, capsys):
    rc = main([str(batch_dir), "--no-charts", "-y"])
    assert rc == 0
    assert (batch_dir / "a-report.html").is_file()
    assert (batch_dir / "b-report.html").is_file()
    out = capsys.readouterr().out
    assert "สำเร็จ: 2/2" in out
    assert "a-report.html" in out
    assert "b-report.html" in out


def test_folder_batch_dot_cwd(tmp_path, monkeypatch, capsys):
    """``thaieda .`` ใน temp dir — ประมวลผลไฟล์ใน cwd."""
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"v": [1, 2]}).to_csv("one.csv", index=False)
    pd.DataFrame({"v": [3, 4]}).to_csv("two.csv", index=False)
    rc = main([".", "--no-charts", "-y"])
    assert rc == 0
    assert (tmp_path / "one-report.html").is_file()
    assert (tmp_path / "two-report.html").is_file()


def test_folder_batch_output_dir(batch_dir, tmp_path):
    out_dir = tmp_path / "reports"
    rc = main([str(batch_dir), "--no-charts", "-y", "--output-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "a-report.html").is_file()
    assert (out_dir / "b-report.html").is_file()
    assert not (batch_dir / "a-report.html").exists()


def test_folder_batch_shared_target(batch_dir, capsys):
    rc = main([str(batch_dir), "--no-charts", "-y", "-t", "clicked"])
    assert rc == 0
    assert "Target: clicked (จาก --target)" in capsys.readouterr().out


def test_folder_skips_report_html(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    pd.DataFrame({"x": [1]}).to_csv(d / "raw.csv", index=False)
    (d / "raw-report.html").write_text("<html></html>", encoding="utf-8")
    rc = main([str(d), "--no-charts", "-y", "--quiet"])
    assert rc == 0
    # มีแค่ raw.csv — ไม่ควรพยายามประมวลผล raw-report.html
    htmls = list(d.glob("*-report.html"))
    assert len(htmls) == 1
    assert htmls[0].name == "raw-report.html"


def test_folder_empty_error(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = main([str(empty), "--no-charts", "-y"])
    assert rc == 2
    assert "ไม่พบไฟล์ข้อมูล" in capsys.readouterr().err


def test_dataset_subcommand_unchanged(tmp_path):
    """``thaieda dataset folder/`` ยังเป็น schema discovery — ไม่ใช่ batch per-file."""
    from thaieda.cli import main as cli_main

    d = tmp_path / "schema"
    d.mkdir()
    pd.DataFrame({"id": [1], "name": ["a"]}).to_csv(d / "users.csv", index=False)
    pd.DataFrame({"user_id": [1], "item": ["x"]}).to_csv(d / "orders.csv", index=False)
    out = d / "schema.html"
    rc = cli_main(["dataset", str(d), "-o", str(out), "--quiet"])
    assert rc == 0
    assert out.is_file()
    assert "erDiagram" in out.read_text(encoding="utf-8")
