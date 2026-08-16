"""Optional LLM narration layer.

The whole platform works with zero external API access. If (and only if) an
API key is present in the environment, this module can turn the already
computed numbers into a short Korean narrative. It never invents numbers -
it is handed the computed evidence and asked to phrase it.

Supported env vars:
    OPENAI_API_KEY   (+ optional OPENAI_BASE_URL, OPENAI_MODEL)
    ANTHROPIC_API_KEY(+ optional ANTHROPIC_MODEL)
    GEMINI_API_KEY   (+ optional GEMINI_MODEL)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

TIMEOUT_S = 25

SYSTEM_PROMPT = (
    "You are a semiconductor process integration engineer. You will be given "
    "numbers that were computed from a TCAD DOE dataset and an ML surrogate "
    "model. Summarise them in Korean in at most 4 sentences. "
    "Rules: never invent a number that is not in the input; never claim a "
    "physical mechanism the data does not show; always end by stating that "
    "TCAD/Fab verification is required."
)


def available() -> dict[str, Any]:
    provider = None
    if os.getenv("OPENAI_API_KEY"):
        provider = "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = "anthropic"
    elif os.getenv("GEMINI_API_KEY"):
        provider = "gemini"
    return {"enabled": provider is not None, "provider": provider}


def _post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def narrate(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return {'enabled', 'provider', 'text'|'error'}. Never raises."""
    info = available()
    if not info["enabled"]:
        return {
            **info,
            "text": None,
            "note": "LLM narration disabled (no API key). All core features run without it.",
        }
    user = (
        "다음은 TCAD DOE 데이터와 학습된 surrogate model에서 실제로 계산된 값입니다.\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)[:6000]
    )
    try:
        if info["provider"] == "openai":
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            data = _post(
                f"{base}/chat/completions",
                {
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                },
                {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            )
            text = data["choices"][0]["message"]["content"]
        elif info["provider"] == "anthropic":
            data = _post(
                "https://api.anthropic.com/v1/messages",
                {
                    "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
                    "max_tokens": 600,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user}],
                },
                {
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                },
            )
            text = data["content"][0]["text"]
        else:
            model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            key = os.environ["GEMINI_API_KEY"]
            data = _post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                {
                    "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"parts": [{"text": user}]}],
                },
                {},
            )
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {**info, "text": text.strip()}
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError) as exc:
        return {**info, "text": None, "error": f"{type(exc).__name__}: {exc}"}
