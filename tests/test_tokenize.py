"""ทดสอบ thaieda.tokenize — adapter, factory, และ error เมื่อไม่มี engine."""

from __future__ import annotations

import pytest

from thaieda.tokenize import Tokenizer, available_engines, get_tokenizer

_HAS_ENGINE = len(available_engines()) > 0


@pytest.mark.skipif(not _HAS_ENGINE, reason="ไม่มี Thai tokenizer engine ติดตั้ง")
def test_get_tokenizer_auto():
    tok = get_tokenizer("auto")
    assert isinstance(tok, Tokenizer)
    assert hasattr(tok, "name")


@pytest.mark.skipif(not _HAS_ENGINE, reason="ไม่มี Thai tokenizer engine ติดตั้ง")
def test_tokenize_returns_list_of_str():
    tok = get_tokenizer("auto")
    result = tok.tokenize("อาหารอร่อยมาก")
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)
    assert len(result) >= 1


@pytest.mark.skipif(not _HAS_ENGINE, reason="ไม่มี Thai tokenizer engine ติดตั้ง")
def test_tokenize_empty_string():
    tok = get_tokenizer("auto")
    assert tok.tokenize("") == []


@pytest.mark.skipif(not _HAS_ENGINE, reason="ไม่มี Thai tokenizer engine ติดตั้ง")
def test_engine_name_reported():
    tok = get_tokenizer("auto")
    # ชื่อ engine ต้องถูกรายงานใน metadata
    assert isinstance(tok.name, str) and tok.name


def test_unknown_engine_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown tokenizer engine"):
        get_tokenizer("not_a_real_engine")


def test_no_engine_raises_importerror(monkeypatch):
    """จำลองว่าไม่มี engine ใดติดตั้ง — ต้อง raise ImportError พร้อมข้อความช่วยเหลือ."""
    import thaieda.tokenize as tk

    # ทำให้ทุก engine ดูเหมือนไม่ได้ติดตั้ง
    monkeypatch.setattr(tk, "_engine_available", lambda engine: False)

    with pytest.raises(ImportError) as exc_info:
        get_tokenizer("auto")

    msg = str(exc_info.value)
    assert "thaieda[thai]" in msg
    assert "No tokenizer engine found" in msg


def test_no_engine_never_silent_whitespace_fallback(monkeypatch):
    """ยืนยันว่าไม่มีการ fallback ไปตัดด้วยช่องว่างแบบเงียบ ๆ."""
    import thaieda.tokenize as tk

    monkeypatch.setattr(tk, "_engine_available", lambda engine: False)
    # ต้อง raise ไม่ใช่คืน tokenizer ที่ตัดด้วย split()
    with pytest.raises(ImportError):
        get_tokenizer("auto")


# ----------------------------------------------------- AC-4: auto modes
class _DummyTokenizer:
    """tokenizer ปลอมสำหรับทดสอบลำดับการเลือก โดยไม่ต้องโหลด engine จริง."""

    def __init__(self, name: str) -> None:
        self.name = name

    def tokenize(self, text: str) -> list[str]:
        return [text] if text else []


def _patch_factories(monkeypatch, available: dict[str, bool]):
    """ทำให้ทุก engine ใช้ tokenizer ปลอม และกำหนดว่า engine ใด 'ติดตั้ง' บ้าง."""
    import thaieda.tokenize as tk

    monkeypatch.setattr(tk, "_engine_available", lambda e: available[e])
    monkeypatch.setattr(
        tk,
        "_FACTORIES",
        {
            "pythainlp": lambda: _DummyTokenizer("pythainlp:newmm"),
            "nlpo3": lambda: _DummyTokenizer("nlpo3"),
            "attacut": lambda: _DummyTokenizer("attacut"),
        },
    )


def test_auto_fast_prefers_nlpo3(monkeypatch):
    import thaieda.tokenize as tk

    _patch_factories(monkeypatch, {"pythainlp": True, "nlpo3": True, "attacut": True})
    assert tk.get_tokenizer("auto-fast").name == "nlpo3"


def test_auto_quality_prefers_attacut(monkeypatch):
    import thaieda.tokenize as tk

    _patch_factories(monkeypatch, {"pythainlp": True, "nlpo3": True, "attacut": True})
    assert tk.get_tokenizer("auto-quality").name == "attacut"


def test_auto_default_unchanged_prefers_pythainlp(monkeypatch):
    # โหมด "auto" เดิมต้องไม่เปลี่ยน — ยังเลือก pythainlp ก่อน
    import thaieda.tokenize as tk

    _patch_factories(monkeypatch, {"pythainlp": True, "nlpo3": True, "attacut": True})
    assert tk.get_tokenizer("auto").name == "pythainlp:newmm"


def test_auto_fast_falls_back_when_nlpo3_missing(monkeypatch):
    # nlpo3 ไม่มี -> auto-fast ถอยไป pythainlp (ตัวถัดไปในลำดับ)
    import thaieda.tokenize as tk

    _patch_factories(monkeypatch, {"pythainlp": True, "nlpo3": False, "attacut": False})
    assert tk.get_tokenizer("auto-fast").name == "pythainlp:newmm"


def test_auto_quality_falls_back_when_attacut_missing(monkeypatch):
    # attacut ไม่มี -> auto-quality ถอยไป pythainlp
    import thaieda.tokenize as tk

    _patch_factories(monkeypatch, {"pythainlp": True, "nlpo3": False, "attacut": False})
    assert tk.get_tokenizer("auto-quality").name == "pythainlp:newmm"


def test_auto_fast_no_engine_raises(monkeypatch):
    import thaieda.tokenize as tk

    monkeypatch.setattr(tk, "_engine_available", lambda e: False)
    with pytest.raises(ImportError):
        tk.get_tokenizer("auto-fast")


def test_auto_quality_no_engine_raises(monkeypatch):
    import thaieda.tokenize as tk

    monkeypatch.setattr(tk, "_engine_available", lambda e: False)
    with pytest.raises(ImportError):
        tk.get_tokenizer("auto-quality")


def _patch_factories_with_failures(monkeypatch, available, failing):
    """เหมือน _patch_factories แต่ทำให้บาง engine 'ติดตั้งแต่สร้างไม่ได้' (โยน OSError)."""
    import thaieda.tokenize as tk

    def make(name):
        def factory():
            if name in failing:
                raise OSError(f"{name} backend failed to load")
            return _DummyTokenizer(name)

        return factory

    monkeypatch.setattr(tk, "_engine_available", lambda e: available[e])
    monkeypatch.setattr(
        tk,
        "_FACTORIES",
        {
            "pythainlp": make("pythainlp:newmm"),
            "nlpo3": make("nlpo3"),
            "attacut": make("attacut"),
        },
    )


def test_auto_quality_degrades_when_attacut_import_fails(monkeypatch):
    # attacut ติดตั้งอยู่แต่ import พัง (เช่น torch DLL) -> ต้องถอยไป pythainlp ไม่ใช่ crash
    import thaieda.tokenize as tk

    _patch_factories_with_failures(
        monkeypatch,
        available={"pythainlp": True, "nlpo3": True, "attacut": True},
        failing={"attacut"},
    )
    assert tk.get_tokenizer("auto-quality").name == "pythainlp:newmm"


def test_auto_raises_when_all_installed_engines_fail_to_load(monkeypatch):
    # ทุก engine ติดตั้งแต่สร้างไม่ได้ -> fail loudly ด้วย ImportError (ไม่ใช่ OSError ดิบ)
    import thaieda.tokenize as tk

    _patch_factories_with_failures(
        monkeypatch,
        available={"pythainlp": True, "nlpo3": True, "attacut": True},
        failing={"pythainlp:newmm", "pythainlp", "nlpo3", "attacut"},
    )
    with pytest.raises(ImportError):
        tk.get_tokenizer("auto")


@pytest.mark.skipif(not _HAS_ENGINE, reason="ไม่มี Thai tokenizer engine ติดตั้ง")
def test_auto_modes_return_tokenizer_instance():
    # integration: โหมด auto ทุกแบบต้องคืน Tokenizer ที่ระบุชื่อ engine ได้
    # (ไม่เรียก .tokenize เพราะบาง engine เช่น nlpo3 ต้องลงทะเบียน dict ก่อนใช้งานจริง)
    for mode in ("auto", "auto-fast", "auto-quality"):
        tok = get_tokenizer(mode)
        assert isinstance(tok, Tokenizer)
        assert isinstance(tok.name, str) and tok.name
