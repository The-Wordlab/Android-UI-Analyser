"""Every hosted model request is routed by llm_route, so a model or key change is one place.

A module that builds its own OpenRouter or OpenAI URL decides the endpoint itself: it ignores an
OpenAI key that would have been cheaper, and it silently stops following a model switch.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = re.compile(r"/chat/completions")
# Local model servers answer /chat/completions themselves; the benchmark runners post to a
# caller-chosen local or benchmark endpoint, never a hosted default.
EXEMPT = {
    "src/android_ui_analyser/llm_route.py",
    "src/android_ui_analyser/providers/grounding/local_vllm.py",
    "experiments/aua_controller/serve_local.py",
    "experiments/aua_controller/serve_model.py",
    "experiments/aua_controller/run_live.py",
    "experiments/aua_controller/protocol_smoke.py",
}


def test_no_module_posts_to_a_hosted_model_endpoint_itself() -> None:
    offenders = []
    for path in [*ROOT.glob("src/android_ui_analyser/**/*.py"), *ROOT.glob("experiments/aua_controller/*.py")]:
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXEMPT:
            continue
        text = path.read_text(encoding="utf-8")
        if ENDPOINT.search(text):
            offenders.append(rel)
    assert offenders == [], f"route these through llm_route.prepare: {offenders}"
