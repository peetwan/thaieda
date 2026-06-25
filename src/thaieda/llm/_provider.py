"""ผู้ให้บริการ LLM — เรียก API ของผู้ให้บริการต่าง ๆ (v0.9).

รองรับ 3 provider:
  * ``openai``    — OpenAI GPT (ต้องติดตั้ง ``openai`` package)
  * ``anthropic`` — Anthropic Claude (ต้องติดตั้ง ``anthropic`` package)
  * ``ollama``     — Ollama local server (ต้องติดตั้ง ``ollama`` package หรือใช้ HTTP API)

หลักการ: lazy import — ไม่ import ตอน import โมดูล จะ import เฉพาะตอนเรียกใช้
ถ้า package ไม่ได้ติดตั้ง จะ raise ImportError พร้อมคำแนะนำติดตั้ง (ไม่มี silent fallback)

ดึง API key จาก environment variable:
  * openai    → OPENAI_API_KEY
  * anthropic → ANTHROPIC_API_KEY
  * ollama    → OLLAMA_HOST (default: http://localhost:11434)
"""

from __future__ import annotations

import os

# ----------------------------------------------------------------------------
# ค่าเริ่มต่ำตั้งของ model ต่อ provider (ถ้าผู้ใช้ไม่ระบุ)
# ----------------------------------------------------------------------------
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "ollama": "llama3.1",
}

_PROVIDER_INSTALL_HINT: dict[str, str] = {
    "openai": "pip install openai",
    "anthropic": "pip install anthropic",
    "ollama": "pip install ollama  # หรือใช้ HTTP API ผ่าน Ollama server",
}


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def call_llm(prompt: str, provider: str, model: str | None = None) -> str:
    """เรียก LLM API ของผู้ให้บริการที่ระบุ — lazy import ของเสริม.

    Args:
        prompt: ข้อความ prompt ที่จะส่งให้ LLM.
        provider: ชื่อผู้ให้บริการ — "openai" | "anthropic" | "ollama".
        model: ชื่อโมเดล — None = ใช้ default ของ provider.

    Returns:
        ข้อความตอบกลับจาก LLM (เป็น string).

    Raises:
        ValueError: ถ้า provider ไม่รองรับ.
        ImportError: ถ้า package ของ provider ไม่ได้ติดตั้ง.
        RuntimeError: ถ้าเรียก API ไม่สำเร็จ (ไม่มี silent fallback).
    """
    provider_lower = provider.lower().strip()
    if provider_lower not in _DEFAULT_MODELS:
        supported = ", ".join(sorted(_DEFAULT_MODELS))
        raise ValueError(f"ไม่รองรับ provider {provider!r} — รองรับ: {supported}. ติดตั้งหรือระบุชื่อให้ถูกต้อง")

    model = model or _DEFAULT_MODELS[provider_lower]
    dispatch = {
        "openai": _call_openai,
        "anthropic": _call_anthropic,
        "ollama": _call_ollama,
    }
    return dispatch[provider_lower](prompt, model)


# ----------------------------------------------------------------------------
# แต่ละ provider (lazy import — import เฉพาะตอนเรียก)
# ----------------------------------------------------------------------------
def _call_openai(prompt: str, model: str) -> str:
    """เรียก OpenAI Chat Completions API (lazy import openai)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ไม่พบตัวแปรสภาพแวดล้อม OPENAI_API_KEY — ตั้งค่าก่อนใช้ provider 'openai' "
            "(export OPENAI_API_KEY=sk-...)"
        )
    try:
        from openai import OpenAI  # type: ignore[import-not-found]  # lazy import — optional
    except ImportError as exc:
        raise ImportError(
            f"ไม่พบแพ็กเกจ 'openai' — ติดตั้งก่อนใช้ provider 'openai': {_PROVIDER_INSTALL_HINT['openai']}"
        ) from exc

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analysis assistant for ThaiEDA. "
                        "Analyze the provided data summaries and insights. "
                        "Respond in Thai unless instructed otherwise."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001 — ครอบ error ของ API ทุกแบบ
        raise RuntimeError(f"OpenAI API เรียกไม่สำเร็จ (model={model}): {exc}") from exc

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError(f"OpenAI ส่งคืนเนื้อหาว่าง (model={model})")
    return content


def _call_anthropic(prompt: str, model: str) -> str:
    """เรียก Anthropic Messages API (lazy import anthropic)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ไม่พบตัวแปรสภาพแวดล้อม ANTHROPIC_API_KEY — ตั้งค่าก่อนใช้ provider 'anthropic' "
            "(export ANTHROPIC_API_KEY=sk-ant-...)"
        )
    try:
        from anthropic import (  # type: ignore[import-not-found]  # noqa: TID252  # lazy import
            Anthropic,
        )
    except ImportError as exc:
        raise ImportError(
            f"ไม่พบแพ็กเกจ 'anthropic' — ติดตั้งก่อนใช้ provider 'anthropic': "
            f"{_PROVIDER_INSTALL_HINT['anthropic']}"
        ) from exc

    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=(
                "You are a data analysis assistant for ThaiEDA. "
                "Analyze the provided data summaries and insights. "
                "Respond in Thai unless instructed otherwise."
            ),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001 — ครอบ error ของ API ทุกแบบ
        raise RuntimeError(f"Anthropic API เรียกไม่สำเร็จ (model={model}): {exc}") from exc

    # response.content เป็น list ของ content block — เอา text จาก text block
    parts = [block.text for block in response.content if hasattr(block, "text")]
    text = "".join(parts)
    if not text:
        raise RuntimeError(f"Anthropic ส่งคืนเนื้อหาว่าง (model={model})")
    return text


def _call_ollama(prompt: str, model: str) -> str:
    """เรียก Ollama local server (lazy import ollama หรือใช้ urllib fallback).

    Ollama เป็น local server — ไม่ต้องมี API key. ใช้ HTTP API ที่
    ``OLLAMA_HOST`` (default: http://localhost:11434).
    """
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    # ลองใช้ ollama python package ก่อน ถ้าไม่มีถอยไปใช้ urllib (built-in)
    try:
        response_text = _call_ollama_via_pkg(prompt, model, host)
    except ImportError:
        # ไม่มี ollama package — ใช้ urllib (built-in) เรียก HTTP API ตรง ๆ
        response_text = _call_ollama_via_http(prompt, model, host)

    if not response_text:
        raise RuntimeError(f"Ollama ส่งคืนเนื้อหาว่าง (model={model}, host={host})")
    return response_text


def _call_ollama_via_pkg(prompt: str, model: str, host: str) -> str:
    """เรียก Ollama ผ่าน python package (lazy import)."""
    try:
        from ollama import (  # type: ignore[import-not-found]  # noqa: TID252  # lazy import
            Client,
        )
    except ImportError as exc:
        raise ImportError(
            f"ไม่พบแพ็กเกจ 'ollama' — ติดตั้งก่อนใช้ provider 'ollama': {_PROVIDER_INSTALL_HINT['ollama']}"
        ) from exc

    client = Client(host=host)
    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analysis assistant for ThaiEDA. "
                        "Analyze the provided data summaries and insights. "
                        "Respond in Thai unless instructed otherwise."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )
    except Exception as exc:  # noqa: BLE001 — ครอบ error ของ API ทุกแบบ
        raise RuntimeError(f"Ollama API เรียกไม่สำเร็จ (model={model}, host={host}): {exc}") from exc

    # ollama package ส่งคืน dict ที่มี key "message" → "content"
    msg = response.get("message", {}) if isinstance(response, dict) else {}
    return str(msg.get("content", ""))


def _call_ollama_via_http(prompt: str, model: str, host: str) -> str:
    """เรียก Ollama ผ่าน HTTP API ตรง ๆ (fallback เมื่อไม่มี ollama package).

    ใช้ urllib.request (built-in) — ไม่ต้องติดตั้งอะไรเพิ่ม.
    """
    import json
    import urllib.request

    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a data analysis assistant for ThaiEDA. "
                    "Analyze the provided data summaries and insights. "
                    "Respond in Thai unless instructed otherwise."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — ครอบ error ของ HTTP ทุกแบบ
        raise RuntimeError(
            f"Ollama HTTP API เรียกไม่สำเร็จ (model={model}, host={host}): {exc}"
        ) from exc

    msg = body.get("message", {}) if isinstance(body, dict) else {}
    return str(msg.get("content", ""))


__all__ = ["call_llm"]
