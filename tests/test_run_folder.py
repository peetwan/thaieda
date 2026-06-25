"""Tests สำหรับ thaieda.run_folder() — one-liner สำหรับวิเคราะห์ทุกไฟล์ในโฟลเดอร์."""

from __future__ import annotations

import pytest

import thaieda
from thaieda import FolderResult, run_folder


# ------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------
@pytest.fixture()
def tmp_csv_folder(tmp_path):
    """สร้างโฟลเดอร์ชั่วคราวมี CSV 3 ไฟล์."""
    import pandas as pd

    # CSV 1: ข้อมูลตัวเลข
    df1 = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    df1.to_csv(tmp_path / "numeric.csv", index=False)

    # CSV 2: ข้อมูลข้อความไทย
    df2 = pd.DataFrame(
        {"review": ["อร่อยมาก", "ดี", "เยี่ยม", "ปกติ", "แย่"], "rating": [5, 4, 5, 3, 1]}
    )
    df2.to_csv(tmp_path / "thai_text.csv", index=False)

    # CSV 3: ข้อมูลผสม
    df3 = pd.DataFrame(
        {
            "name": ["สมชาย", "สมหญิง", "วิชัย"],
            "age": [25, 30, 35],
            "city": ["กรุงเทพ", "เชียงใหม่", "ภูเก็ต"],
        }
    )
    df3.to_csv(tmp_path / "mixed.csv", index=False)

    # ไฟล์ที่ไม่รองรับ (ไม่ควรถูกประมวลผล)
    (tmp_path / "readme.txt").write_text("not a csv")

    return tmp_path


@pytest.fixture()
def tmp_empty_folder(tmp_path):
    """โฟลเดอร์ว่าง (ไม่มีไฟล์ที่รองรับ)."""
    (tmp_path / "readme.txt").write_text("no csv here")
    return tmp_path


# ------------------------------------------------------------------------------
# tests — ฟังก์ชันพื้นฐาน
# ------------------------------------------------------------------------------
class TestRunFolderBasic:
    """ทดสอบฟังก์ชันพื้นฐานของ run_folder()."""

    def test_returns_folder_result(self, tmp_csv_folder):
        """run_folder() คืน FolderResult."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        assert isinstance(result, FolderResult)

    def test_finds_all_csvs(self, tmp_csv_folder):
        """พบ CSV ครบ 3 ไฟล์ (ไม่นับ .txt)."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        assert result.total_files == 3
        assert result.success == 3
        assert result.failed == 0

    def test_ignores_unsupported_files(self, tmp_csv_folder):
        """ไม่ประมวลผลไฟล์ที่ไม่รองรับ (.txt)."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        filenames = [fr.filename for fr in result.results]
        assert "readme.txt" not in filenames

    def test_folder_attribute(self, tmp_csv_folder):
        """folder attribute เก็บพาธโฟลเดอร์ถูกต้อง."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        assert str(tmp_csv_folder) in result.folder or result.folder == str(tmp_csv_folder)


# ------------------------------------------------------------------------------
# tests — error handling
# ------------------------------------------------------------------------------
class TestRunFolderErrors:
    """ทดสอบการจัดการข้อผิดพลาด."""

    def test_folder_not_found(self):
        """ถ้าโฟลเดอร์ไม่มี → FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ไม่พบโฟลเดอร์"):
            run_folder("/nonexistent/path/xyz")

    def test_no_supported_files(self, tmp_empty_folder):
        """ถ้าไม่มีไฟล์ที่รองรับ → ValueError."""
        with pytest.raises(ValueError, match="ไม่พบไฟล์ที่รองรับ"):
            run_folder(tmp_empty_folder, save_html=False, make_charts=False)

    def test_single_file_error_doesnt_crash_all(self, tmp_path):
        """ไฟล์พัง 1 ไฟล์ ไม่ทำให้ทั้งโฟลเดอร์พัง."""
        import pandas as pd

        # ไฟล์ดี
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(tmp_path / "good.csv", index=False)
        # ไฟล์พัง (CSV ผิด format)
        (tmp_path / "bad.csv").write_text("not,a,valid\nbroken")

        result = run_folder(tmp_path, save_html=False, make_charts=False)
        assert result.total_files == 2
        # อย่างน้อย 1 ไฟล์ต้องสำเร็จ
        assert result.success >= 1


# ------------------------------------------------------------------------------
# tests — HTML output
# ------------------------------------------------------------------------------
class TestRunFolderHTML:
    """ทดสอบการสร้าง HTML report."""

    def test_save_html_creates_files(self, tmp_csv_folder):
        """save_html=True สร้าง HTML ให้ทุกไฟล์ที่สำเร็จ."""
        result = run_folder(
            tmp_csv_folder, save_html=True, make_charts=False
        )
        html_files = list(tmp_csv_folder.glob("*-report.html"))
        assert len(html_files) == 3

    def test_save_html_to_custom_dir(self, tmp_csv_folder, tmp_path):
        """output_dir ใช้โฟลเดอร์ปลายทางเองได้."""
        out = tmp_path / "reports"
        result = run_folder(
            tmp_csv_folder,
            save_html=True,
            output_dir=str(out),
            make_charts=False,
        )
        html_files = list(out.glob("*-report.html"))
        assert len(html_files) == 3

    def test_save_html_false_no_files(self, tmp_csv_folder):
        """save_html=False ไม่สร้าง HTML."""
        result = run_folder(
            tmp_csv_folder, save_html=False, make_charts=False
        )
        html_files = list(tmp_csv_folder.glob("*-report.html"))
        assert len(html_files) == 0


# ------------------------------------------------------------------------------
# tests — summary & display
# ------------------------------------------------------------------------------
class TestRunFolderSummary:
    """ทดสอบ summary() และ _repr_html_()."""

    def test_summary_text(self, tmp_csv_folder):
        """summary() คืน text ที่มีข้อมูลครบ."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        s = result.summary()
        assert "ThaiEDA FolderResult" in s
        assert "Files: 3" in s
        assert "✅" in s

    def test_repr_html(self, tmp_csv_folder):
        """_repr_html_() คืน HTML string."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        html = result._repr_html_()
        assert "<h3>" in html
        assert "<table>" in html

    def test_file_result_ok_property(self, tmp_csv_folder):
        """_FileResult.ok บอกสถานะถูกต้อง."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        for fr in result.results:
            assert fr.ok is True
            assert fr.error is None
            assert fr.result is not None


# ------------------------------------------------------------------------------
# tests — kwargs forwarding
# ------------------------------------------------------------------------------
class TestRunFolderKwargs:
    """ทดสอบการส่ง kwargs ไปยัง run()."""

    def test_lang_en(self, tmp_csv_folder):
        """lang='en' ส่งไปยัง run() ได้."""
        result = run_folder(
            tmp_csv_folder, save_html=False, make_charts=False, lang="en"
        )
        assert result.success == 3

    def test_clean_false(self, tmp_csv_folder):
        """clean=False ส่งไปยัง run() ได้."""
        result = run_folder(
            tmp_csv_folder, save_html=False, make_charts=False, clean=False
        )
        assert result.success == 3

    def test_invalid_kwarg_ignored(self, tmp_csv_folder):
        """kwargs ที่ run() ไม่รองรับ ถูกกรองออก ไม่พัง."""
        result = run_folder(
            tmp_csv_folder,
            save_html=False,
            make_charts=False,
            nonexistent_param=True,
        )
        assert result.success == 3


# ------------------------------------------------------------------------------
# tests — recursive
# ------------------------------------------------------------------------------
class TestRunFolderRecursive:
    """ทดสอบการค้นหาในโฟลเดอร์ย่อย."""

    def test_recursive_finds_subfolder_files(self, tmp_path):
        """recursive=True ค้นหาในโฟลเดอร์ย่อยด้วย."""
        import pandas as pd

        # โฟลเดอร์หลัก
        pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "main.csv", index=False)
        # โฟลเดอร์ย่อย
        sub = tmp_path / "sub"
        sub.mkdir()
        pd.DataFrame({"b": [3, 4]}).to_csv(sub / "sub.csv", index=False)

        result = run_folder(
            tmp_path, save_html=False, make_charts=False, recursive=True
        )
        assert result.total_files == 2

    def test_non_recursive_ignores_subfolder(self, tmp_path):
        """recursive=False (default) ไม่ค้นหาในโฟลเดอร์ย่อย."""
        import pandas as pd

        pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "main.csv", index=False)
        sub = tmp_path / "sub"
        sub.mkdir()
        pd.DataFrame({"b": [3, 4]}).to_csv(sub / "sub.csv", index=False)

        result = run_folder(
            tmp_path, save_html=False, make_charts=False, recursive=False
        )
        assert result.total_files == 1


# ------------------------------------------------------------------------------
# tests — progress callback
# ------------------------------------------------------------------------------
class TestRunFolderProgress:
    """ทดสอบ progress callback."""

    def test_progress_called(self, tmp_csv_folder):
        """progress callback ถูกเรียก."""
        messages: list[str] = []
        result = run_folder(
            tmp_csv_folder,
            save_html=False,
            make_charts=False,
            progress=lambda msg: messages.append(msg),
        )
        assert len(messages) > 0
        assert any("numeric.csv" in m for m in messages)


# ------------------------------------------------------------------------------
# tests — master HTML
# ------------------------------------------------------------------------------
class TestRunFolderMasterHTML:
    """ทดสอบ to_master_html() — รวมทุกไฟล์เป็น master HTML."""

    def test_master_html_returns_string(self, tmp_csv_folder):
        """to_master_html() คืน HTML string."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        html = result.to_master_html()
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<html" in html.lower()

    def test_master_html_saves_to_file(self, tmp_csv_folder, tmp_path):
        """to_master_html(path) บันทึกไฟล์ได้."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        out = tmp_path / "master.html"
        result.to_master_html(str(out))
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<html" in content.lower()

    def test_master_html_contains_all_files(self, tmp_csv_folder):
        """master HTML มี section สำหรับทุกไฟล์."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        html = result.to_master_html()
        for fr in result.results:
            assert fr.filename in html

    def test_master_html_has_sidebar_nav(self, tmp_csv_folder):
        """master HTML มี sidebar navigation."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        html = result.to_master_html()
        assert "sidebar" in html.lower()
        assert "overview" in html.lower()

    def test_master_html_has_summary_table(self, tmp_csv_folder):
        """master HTML มีตารางสรุป."""
        result = run_folder(tmp_csv_folder, save_html=False, make_charts=False)
        html = result.to_master_html()
        assert "<table" in html.lower()
        assert "insights" in html.lower()

    def test_master_html_includes_failed_files(self, tmp_path):
        """master HTML แสดงไฟล์ที่พังด้วย."""
        import pandas as pd

        pd.DataFrame({"a": [1, 2, 3]}).to_csv(tmp_path / "good.csv", index=False)
        (tmp_path / "bad.csv").write_text("broken,data\nx")

        result = run_folder(tmp_path, save_html=False, make_charts=False)
        html = result.to_master_html()
        assert "bad.csv" in html
        assert "❌" in html