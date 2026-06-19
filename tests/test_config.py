"""AC5 — config precedence, profiles, secrets-by-env-name, validation (PRD §9)."""

from __future__ import annotations

import json
import os

import pytest

from android_ui_analyser.config import Config, load_config, read_env_secret
from android_ui_analyser.errors import ConfigError, ExitCode


def _isolate(monkeypatch, tmp_path) -> None:
    # Point the user-config path at an empty dir and clear AUA_* so only what each test
    # sets participates in the merge.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for key in list(os.environ):
        if key.startswith("AUA_"):
            monkeypatch.delenv(key, raising=False)


def _project(tmp_path, body: str) -> None:
    (tmp_path / ".android-ui-analyser.yaml").write_text(body, encoding="utf-8")


def test_precedence_default_lt_project_lt_env_lt_flag(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    # 1) built-in default
    assert load_config(cwd=tmp_path, env={}).output.format.value == "json"
    # 2) project file overrides default
    _project(tmp_path, "output:\n  format: pretty\n")
    assert load_config(cwd=tmp_path, env={}).output.format.value == "pretty"
    # 3) env overrides project
    assert (
        load_config(cwd=tmp_path, env={"AUA_OUTPUT__FORMAT": "compact"}).output.format.value
        == "compact"
    )
    # 4) CLI flag overrides env
    cfg = load_config(
        cwd=tmp_path,
        env={"AUA_OUTPUT__FORMAT": "compact"},
        cli_overrides={"output": {"format": "json"}},
    )
    assert cfg.output.format.value == "json"


def test_env_nested_double_underscore_and_csv_list(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    cfg = load_config(
        cwd=tmp_path,
        env={"AUA_OCR__CHAIN": "rapidocr,tesseract", "AUA_ROUTING__MAX_TIER": "hierarchy"},
    )
    assert cfg.ocr.chain == ["rapidocr", "tesseract"]
    assert cfg.routing.max_tier.value == "hierarchy"


def test_secret_referenced_by_env_name_never_printed(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value-123")
    cfg = load_config(cwd=tmp_path, env={})
    blob = json.dumps(cfg.masked_dict())
    assert "GEMINI_API_KEY" in blob  # the env-var NAME is fine to show
    assert "super-secret-value-123" not in blob  # the VALUE is never stored/printed
    # the value is read at runtime, by name
    assert read_env_secret("GEMINI_API_KEY") == "super-secret-value-123"
    assert read_env_secret("GEMINI_API_KEY", {"GEMINI_API_KEY": "x"}) == "x"
    assert read_env_secret("UNSET_VAR_zzz_qqq") is None
    assert read_env_secret(None) is None


def test_literal_secret_value_is_masked() -> None:
    # Belt-and-suspenders: even a mistakenly-pasted literal key is masked in config show.
    cfg = Config.model_validate(
        {"models": {"openai": {"api_key": "sk-LITERAL-XYZ", "model": "gpt"}}}
    )
    blob = json.dumps(cfg.masked_dict())
    assert "sk-LITERAL-XYZ" not in blob
    assert "***" in blob


def test_profile_deep_merges_over_base(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _project(
        tmp_path,
        "profiles:\n  cloud:\n    grounding:\n      enabled: true\n      chain: [gemini]\n",
    )
    assert load_config(cwd=tmp_path, env={}).grounding.enabled is False
    cloud = load_config(cwd=tmp_path, env={}, profile="cloud")
    assert cloud.grounding.enabled is True
    assert cloud.grounding.chain == ["gemini"]


def test_invalid_value_raises_config_error_exit_5(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(ConfigError) as ei:
        load_config(cwd=tmp_path, env={}, cli_overrides={"routing": {"max_tier": "bogus_tier"}})
    assert ei.value.exit_code == ExitCode.CONFIG == 5
    assert "max_tier" in ei.value.message or "routing" in ei.value.message


def test_unknown_profile_errors(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(ConfigError):
        load_config(cwd=tmp_path, env={}, profile="does_not_exist")


def test_explicit_config_path_replaces_discovery(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    explicit = tmp_path / "custom.yaml"
    explicit.write_text("output:\n  format: compact\n", encoding="utf-8")
    cfg = load_config(cwd=tmp_path, env={}, explicit_path=str(explicit))
    assert cfg.output.format.value == "compact"
