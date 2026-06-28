"""Tokenizer adapter — ชั้นเชื่อมต่อกับเครื่องมือตัดคำภาษาไทย.

ออกแบบให้เป็น seam เดียวที่โค้ดส่วนอื่นเรียกใช้ ไม่ว่าจะใช้ engine ไหน
หลักการสำคัญ:
  * import ทุก engine แบบ lazy (import ภายในฟังก์ชัน ไม่ใช่บนหัวโมดูล)
  * ไม่มี fallback แบบเงียบ ๆ ไปเป็นการตัดด้วยช่องว่าง — การตัดคำที่ผิดแบบเงียบ
    แย่กว่าการ error ที่ชัดเจน
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

# ลำดับการลองเลือก engine เมื่อ engine="auto"
_AUTO_ORDER = ("pythainlp", "nlpo3", "attacut")

# v1.6: โหมด auto เฉพาะทาง — เลือก engine ตามเป้าหมาย (ความเร็ว vs คุณภาพ)
#   * auto-fast    : เน้นความเร็ว — nlpo3 (Rust) เร็วกว่า ~3-4 เท่า
#   * auto-quality : เน้นคุณภาพ — attacut (neural) ตัดคำ social media / คำนอกพจนานุกรม (OOV) ได้ดีกว่า
# ทั้งสองโหมดถอยไปใช้ engine ตัวถัดไปในลำดับถ้าตัวที่ต้องการไม่ได้ติดตั้ง (degrade อย่างสุภาพ)
_AUTO_FAST_ORDER = ("nlpo3", "pythainlp", "attacut")
_AUTO_QUALITY_ORDER = ("attacut", "pythainlp", "nlpo3")

# แผนที่ชื่อโหมด auto -> ลำดับการลอง engine
_AUTO_MODES: dict[str, tuple[str, ...]] = {
    "auto": _AUTO_ORDER,
    "auto-fast": _AUTO_FAST_ORDER,
    "auto-quality": _AUTO_QUALITY_ORDER,
}

# engine ย่อยเริ่มต้นของ pythainlp — สมดุลความเร็ว/คุณภาพดีที่สุด
_DEFAULT_PYTHAINLP_ENGINE = "newmm"

_NO_ENGINE_MESSAGE = (
    "Thai tokenization requires pip install thaieda[thai]. No tokenizer engine found."
)


@runtime_checkable
class Tokenizer(Protocol):
    """สัญญา (Protocol) ของตัวตัดคำ — ต้องมีเมธอด tokenize และ attribute name."""

    name: str

    def tokenize(self, text: str) -> list[str]:
        """ตัดข้อความเป็นรายการคำ (tokens)."""
        ...


class PyThaiNLPTokenizer:
    """Adapter สำหรับ pythainlp (engine เริ่มต้น: newmm)."""

    def __init__(self, sub_engine: str = _DEFAULT_PYTHAINLP_ENGINE) -> None:
        # lazy import — โหลด pythainlp เฉพาะตอนสร้าง object เท่านั้น
        from pythainlp.tokenize import word_tokenize

        self._word_tokenize = word_tokenize
        self.sub_engine = sub_engine
        self.name = f"pythainlp:{sub_engine}"

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        # keep_whitespace=False เพื่อไม่ให้ช่องว่างปนมาเป็น token
        return self._word_tokenize(text, engine=self.sub_engine, keep_whitespace=False)


class Nlpo3Tokenizer:
    """Adapter สำหรับ nlpo3 (Rust-based, เร็วมาก)."""

    def __init__(self) -> None:
        # lazy import
        from nlpo3 import segment

        self._segment = segment
        self.name = "nlpo3"

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = self._segment(text)
        # ตัดช่องว่างล้วน ๆ ออกเพื่อให้สอดคล้องกับ adapter อื่น
        return [t for t in tokens if t.strip()]


class AttacutTokenizer:
    """Adapter สำหรับ attacut (deep learning tokenizer)."""

    def __init__(self) -> None:
        # lazy import
        from attacut import tokenize as attacut_tokenize

        self._tokenize = attacut_tokenize
        self.name = "attacut"

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = self._tokenize(text)
        return [t for t in tokens if t.strip()]


# แผนที่ชื่อ engine -> ฟังก์ชันสร้าง adapter
_FACTORIES: dict[str, Callable[[], Tokenizer]] = {
    "pythainlp": PyThaiNLPTokenizer,
    "nlpo3": Nlpo3Tokenizer,
    "attacut": AttacutTokenizer,
}


def _engine_available(engine: str) -> bool:
    """ตรวจว่า engine ติดตั้งอยู่ไหม โดยไม่ import เนื้อหนัก ๆ."""
    import importlib.util

    module_name = {"pythainlp": "pythainlp", "nlpo3": "nlpo3", "attacut": "attacut"}[engine]
    return importlib.util.find_spec(module_name) is not None


def available_engines() -> list[str]:
    """คืนรายการ engine ที่ติดตั้งอยู่ในระบบ (ตามลำดับความนิยม)."""
    return [e for e in _AUTO_ORDER if _engine_available(e)]


def get_tokenizer(engine: str = "auto") -> Tokenizer:
    """สร้างและคืน Tokenizer ตาม engine ที่ระบุ.

    Args:
        engine: ชื่อ engine หรือโหมด auto —
            * "auto"          : ลองตามลำดับ pythainlp -> nlpo3 -> attacut (สมดุล)
            * "auto-fast"     : เน้นความเร็ว — nlpo3 -> pythainlp -> attacut
            * "auto-quality"  : เน้นคุณภาพ — attacut -> pythainlp -> nlpo3
            * "pythainlp" / "nlpo3" / "attacut" : ระบุ engine ตรง ๆ
            โหมด auto ทุกแบบจะใช้ engine ตัวแรกในลำดับที่ติดตั้งอยู่.

    Returns:
        Tokenizer ที่พร้อมใช้งาน.

    Raises:
        ImportError: เมื่อไม่มี engine ใดติดตั้งเลย (ข้อความแนะนำชัดเจน).
        ValueError: เมื่อระบุชื่อ engine ที่ไม่รู้จัก.
    """
    if engine in _AUTO_MODES:
        last_error: Exception | None = None
        for candidate in _AUTO_MODES[engine]:
            if not _engine_available(candidate):
                continue
            try:
                return _FACTORIES[candidate]()
            except (ImportError, OSError) as exc:
                # ติดตั้งไว้แต่ import/โหลดไม่ได้ (เช่น attacut ที่ torch DLL พัง) —
                # degrade อย่างสุภาพไป engine ตัวถัดไป แทนที่จะล้มทั้ง pipeline
                last_error = exc
                continue
        # ไม่มี engine ที่ใช้งานได้เลย — fail loudly ตามหลักการ
        raise ImportError(_NO_ENGINE_MESSAGE) from last_error

    if engine not in _FACTORIES:
        raise ValueError(
            f"Unknown tokenizer engine {engine!r}. "
            f"Expected one of: {', '.join(_AUTO_MODES)}, {', '.join(_FACTORIES)}."
        )

    if not _engine_available(engine):
        raise ImportError(f"Tokenizer engine {engine!r} is not installed. {_NO_ENGINE_MESSAGE}")

    return _FACTORIES[engine]()


__all__ = [
    "Tokenizer",
    "PyThaiNLPTokenizer",
    "Nlpo3Tokenizer",
    "AttacutTokenizer",
    "get_tokenizer",
    "available_engines",
]
