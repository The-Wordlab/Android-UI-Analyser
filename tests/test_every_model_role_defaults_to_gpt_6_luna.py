"""Every model role defaults to GPT-6 Luna, set so it can take the company-key route where that is safe.

Measured 2026-09-23 against the DeepSeek defaults:
* controller: 4/4 tasks either way; Luna 2-4x cheaper per task, fewer output tokens, faster.
  Reasoning off lets it reach api.openai.com under an OPENAI_API_KEY, and still passed 4/4.
* judge: reasoning on, 96% pass/not-pass agreement with DeepSeek V4.1 over 27 saved rows and
  no passing row judged failed. Reasoning off turned 4 passing rows into failures, so the judge
  keeps reasoning, which keeps it on OpenRouter (OpenAI refuses tools with reasoning there).
* icon names: 3/3 on the hamburger crop, reasoning off, direct.
"""

from __future__ import annotations

import json
from pathlib import Path

from android_ui_analyser.config import Config

MANIFEST = Path(__file__).resolve().parents[1] / "experiments/aua_controller/openrouter-comparison.json"
BY_ID = {entry["id"]: entry for entry in json.loads(MANIFEST.read_text())["models"]}


def test_the_controller_is_luna_with_reasoning_off() -> None:
    controller = BY_ID[Config().controller.model]
    assert controller["repository"] == "openai/gpt-6-luna"
    assert controller["request_config"]["reasoning"] == {"enabled": False, "exclude": False}


def test_the_judge_is_luna_with_reasoning_and_vision() -> None:
    judge = BY_ID[Config().controller.judge_model]
    assert judge["repository"] == "openai/gpt-6-luna"
    assert judge["request_config"]["reasoning"].get("effort") == "low"
    assert judge["vision"] is True


def test_the_judge_falls_back_to_another_vendor() -> None:
    """An OpenAI outage must not take the judge with it."""
    for rung in Config().controller.judge_fallbacks:
        assert not BY_ID[rung]["repository"].startswith("openai/"), rung


def test_icon_naming_and_grounding_use_luna() -> None:
    models = Config().models
    assert models["hosted_vision"]["model"] == "openai/gpt-6-luna"
    assert models["openai"]["model"] == "openai/gpt-6-luna"
