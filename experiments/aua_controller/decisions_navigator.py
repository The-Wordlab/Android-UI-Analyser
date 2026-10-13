"""OpenAI Decisions screenshot transport for the bounded controller navigator.

Every refusal, missing screenshot, invalid answer or low-confidence proposal leaves the step
with the regular controller. The navigator cannot stop a run or invent text to type.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from experiments.aua_controller.bounded_navigator import BoundedNavigator, numbered, plain
from experiments.aua_controller.judgement import encode_image, frame_screenshot, screenshot_index

MODEL = "gpt-6-luna"
ENDPOINT = "https://api.openai.com/v1/decisions"
USD_PER_INPUT_TOKEN = 0.10 / 1_000_000


def translate_questions(questions: Mapping[str, Any]):
    """Send actual choices; a sole authored string is deterministic, never a fabricated option."""
    api_questions, deterministic = [], {}
    for name, original in questions.items():
        question = plain(original)
        if question.get("type") != "choice":
            raise ValueError("navigator only accepts choice questions")
        criteria = question["criteria"]
        if len(criteria) == 1:
            choice = next(iter(criteria))
            deterministic[name] = SimpleNamespace(choice=choice, confidence=1.0)
        elif len(criteria) >= 2:
            api_questions.append({"name": name, "type": "choice",
                                  "instructions": question["instructions"],
                                  "choices": [{"value": value, "description": description}
                                              for value, description in criteria.items()]})
        else:
            raise ValueError("empty choice menu")
    return api_questions, deterministic


def decode_answers(body: Mapping[str, Any], questions: Mapping[str, Any], deterministic):
    answers = dict(deterministic)
    returned = body.get("answers")
    if not isinstance(returned, list):
        raise ValueError("missing typed answers")
    for answer in returned:
        if not isinstance(answer, Mapping):
            raise ValueError("invalid typed answer")
        name = answer.get("name")
        if name not in questions or name in answers or answer.get("type") != "choice":
            raise ValueError("refused, duplicated, or unknown answer")
        choice, confidence = answer.get("choice"), answer.get("confidence")
        if not isinstance(choice, str) or choice not in plain(questions[name])["criteria"]:
            raise ValueError("choice outside the offered menu")
        if (isinstance(confidence, bool) or not isinstance(confidence, (float, int))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("confidence outside 0..1")
        answers[name] = SimpleNamespace(choice=choice, confidence=float(confidence))
    if set(answers) != set(questions):
        raise ValueError("missing answer")
    return answers


def input_usage(body: Mapping[str, Any]) -> tuple[int, int]:
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("input usage unavailable")
    tokens = usage.get("input_tokens")
    details = usage.get("input_tokens_details") or {}
    if not isinstance(details, Mapping):
        raise ValueError("invalid input usage details")
    cached, written = details.get("cached_tokens", 0), details.get("cache_write_tokens", 0)
    if any(type(v) is not int or v < 0 for v in (tokens, cached, written)) or cached + written > tokens:
        raise ValueError("invalid input usage")
    return tokens, tokens - cached - written


class DecisionsNavigator(BoundedNavigator):
    engine = "decisions"
    provider = "openai"
    def __init__(self, goal: str, *, api_key: str | None = None,
                 http_client: Any = None, manifest_path: Any = None, **kwargs: Any):
        super().__init__(goal, model=MODEL, **kwargs)
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("Decisions requires OPENAI_API_KEY")
        self.http_client = http_client
        self.manifest = (Path(manifest_path) if manifest_path else
                         self.transcript_path.parent.parent / "aua/manifest.json"
                         if self.transcript_path else None)
        self._source_frame: Any = None
        self.images_sent = self.failed_requests = self.attempted_requests = 0
        self.cost_complete = True

    async def __call__(self, result: Any):
        self._source_frame = result
        return await super().__call__(result)

    async def _ask(self, compact, questions):
        image_path = frame_screenshot(self._source_frame, screenshot_index(self.manifest))
        encoded = encode_image(image_path, max_width=720) if image_path else None
        if not encoded:
            self._decline("screenshot_unavailable")
            return None
        state = self._state(compact)
        api_questions, deterministic = translate_questions(questions)
        raw_image = ((self._source_frame.get("observation") or {}).get("meta") or {}).get("raw_image") if isinstance(self._source_frame, Mapping) else None
        image_info = {"path": image_path, "sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
                      "encoded_sha256": hashlib.sha256(base64.b64decode(encoded.split(",", 1)[1])).hexdigest(),
                      "max_width": 720, "source": "observation_raw_image" if raw_image else "fingerprint_index"}
        payload = {"model": MODEL, "input": [{"role": "user", "content": [
            {"type": "input_text", "text": json.dumps(state, ensure_ascii=False)},
            {"type": "input_image", "image_url": encoded}]}], "questions": api_questions}
        self.attempted_requests += 1
        self.images_sent += 1
        entry = {"attempt": self.attempted_requests,
                 "request": {"endpoint": "/v1/decisions", "model": MODEL, "state": state,
                             "questions": api_questions, "image": image_info},
                 "deterministic_single_options": list(deterministic), "menu": numbered(self._options)[0]}
        started = time.perf_counter()
        accounted = False
        try:
            async def post(client):
                return await client.post(ENDPOINT, json=payload,
                                         headers={"Authorization": "Bearer " + self.api_key},
                                         timeout=self.timeout_s)
            if self.http_client is None:
                async with httpx.AsyncClient(trust_env=False) as client:
                    response = await post(client)
            else:
                response = await post(self.http_client)
            entry["http_status"] = response.status_code
            response.raise_for_status()
            body = response.json()
            entry["response"] = body
            tokens, charged = input_usage(body)
            cost = charged * USD_PER_INPUT_TOKEN
            # Account a paid refusal or malformed answer before falling back.
            self.requests += 1
            self.input_tokens += tokens
            self.usd += cost
            accounted = True
            entry.update(call=self.requests, input_tokens=tokens, usd=round(cost, 9))
            answers = decode_answers(body, questions, deterministic)
        except Exception as error:
            self.failed_requests += 1
            self.cost_complete = self.cost_complete and accounted
            elapsed = (time.perf_counter() - started) * 1000
            self.request_ms.append(elapsed)
            entry.update(error_type=type(error).__name__, request_ms=round(elapsed, 1),
                         cost_unreported=not accounted)
            self._record(entry)
            self.last_decision = entry
            self._decline("request_failed:" + type(error).__name__)
            return None
        elapsed = (time.perf_counter() - started) * 1000
        self.request_ms.append(elapsed)
        entry["request_ms"] = round(elapsed, 1)
        return entry, answers

    def report(self):
        report = super().report()
        report.update(engine="decisions", provider="openai", endpoint="/v1/decisions",
                      images_sent=self.images_sent, attempted_requests=self.attempted_requests,
                      failed_requests=self.failed_requests, cost_complete=self.cost_complete,
                      price_usd_per_million_input_tokens=.10, estimated=True,
                      calibration="inherited 0.85 gate; not calibrated for Decisions")
        return report
