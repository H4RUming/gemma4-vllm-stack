"""Client request shape normalization.

Two concerns:

1. **Thinking-mode normalization.** vLLM's chat endpoint accepts the canonical
   `chat_template_kwargs.enable_thinking` (bool), but clients in the wild send
   any of `reasoning_effort`, `extra_body.thinking`, or `thinking_config`. We
   collapse all four into the canonical form before forwarding.

2. **Sampling defaults.** When a client omits `temperature`/`top_p`/`top_k`,
   we fill in the lab's defaults. We NEVER overwrite a client-provided value.

Precedence for thinking-mode (highest wins):
    chat_template_kwargs.enable_thinking
        > reasoning_effort
        > extra_body.thinking
        > thinking_config
        > THINKING_DEFAULT (ON)
"""

from typing import Any

from config import SAMPLING_DEFAULTS, THINKING_DEFAULT


_REASONING_EFFORT_ON = {"high", "medium", "low"}
_REASONING_EFFORT_OFF = {"none", "off", "minimal"}


def _coerce_bool(value: Any) -> bool | None:
    """Best-effort cast of common client encodings to bool. None on unknown."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "on", "yes", "1"}:
            return True
        if v in {"false", "off", "no", "0"}:
            return False
        if v in _REASONING_EFFORT_ON:
            return True
        if v in _REASONING_EFFORT_OFF:
            return False
    if isinstance(value, dict):
        # e.g. {"type": "enabled"} / {"enabled": True}
        if "enabled" in value:
            return bool(value["enabled"])
        if value.get("type") in {"enabled", "on"}:
            return True
        if value.get("type") in {"disabled", "off", "none"}:
            return False
    return None


def normalize_thinking(body: dict) -> str:
    """Collapse thinking-mode encodings to canonical chat_template_kwargs.enable_thinking.

    Returns a string label ("high" or "none") usable as the `reasoning_effort`
    field in the JSONL log row.
    """
    # 1. Canonical form takes precedence if already set.
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict) and "enable_thinking" in ctk:
        enabled = bool(ctk["enable_thinking"])
        return "high" if enabled else "none"

    # 2. reasoning_effort (OpenAI o-series style).
    enabled: bool | None = None
    if "reasoning_effort" in body:
        enabled = _coerce_bool(body["reasoning_effort"])

    # 3. extra_body.thinking (some Anthropic-style SDKs).
    if enabled is None:
        extra = body.get("extra_body")
        if isinstance(extra, dict) and "thinking" in extra:
            enabled = _coerce_bool(extra["thinking"])

    # 4. thinking_config (legacy clients).
    if enabled is None and "thinking_config" in body:
        enabled = _coerce_bool(body["thinking_config"])

    # 5. Default.
    if enabled is None:
        enabled = THINKING_DEFAULT

    # Write canonical form.
    ctk = body.setdefault("chat_template_kwargs", {})
    if not isinstance(ctk, dict):
        ctk = {}
        body["chat_template_kwargs"] = ctk
    ctk["enable_thinking"] = enabled

    return "high" if enabled else "none"


def inject_sampling_defaults(body: dict) -> None:
    """Fill sampling params the client omitted. Never overwrite."""
    for key, value in SAMPLING_DEFAULTS.items():
        body.setdefault(key, value)


def adapt_request(body: dict) -> str:
    """Apply both adaptations. Returns the reasoning_effort label for logging."""
    label = normalize_thinking(body)
    inject_sampling_defaults(body)
    return label
