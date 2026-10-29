"""Optional developer tooling stays bounded, explicit, and device-independent on the host."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from android_ui_analyser.errors import ConfigError, UsageError
from android_ui_analyser.platforms import android_cli
from android_ui_analyser.platforms.android_cli import AndroidCliError, AndroidCliTools


@pytest.fixture
def binary(tmp_path):
    path = tmp_path / "fixture android"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, subprocess, sys, time\n"
        "args=sys.argv[1:]\n"
        "if '--version' in args:\n"
        "    print('1.1.16717390'); sys.exit(0)\n"
        "if args[-1:] == ['fail']:\n"
        "    print('fixture failure',file=sys.stderr); sys.exit(7)\n"
        "if args[-1:] == ['absent']:\n"
        "    print('Session not found',file=sys.stderr); sys.exit(1)\n"
        "if args[-1:] == ['large']:\n"
        "    print('x'*300000); print('y'*300000,file=sys.stderr); sys.exit(0)\n"
        "if args[-1:] == ['slow']:\n"
        "    subprocess.Popen([sys.executable,'-c',\"import time,pathlib; time.sleep(0.7); pathlib.Path('late-child').write_text('bad')\"])\n"
        "    time.sleep(10)\n"
        "out=next((a.split('=',1)[1] for a in args if a.startswith('--output=')),None)\n"
        "if out and 'preview' in args:\n"
        "    p=pathlib.Path(out)/'preview.png'; p.write_bytes(b'fixture'); print(json.dumps({'image':str(p)})); sys.exit(0)\n"
        "print(json.dumps({'argv': args, 'cwd':str(pathlib.Path.cwd())}))\n"
    )
    path.chmod(0o700)
    return path


@pytest.fixture
def tools(binary):
    return AndroidCliTools({"cli_path": str(binary)})


def test_missing_optional_cli_does_not_install_or_load_android_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(android_cli, "_managed_binary", lambda: tmp_path / "missing")
    monkeypatch.setattr(android_cli.shutil, "which", lambda _: None)
    monkeypatch.delenv("AUA_ANDROID_CLI", raising=False)
    tool = AndroidCliTools()
    assert not tool.status()["available"]
    assert len(tool.catalogue()) == 70
    with pytest.raises(ConfigError, match="not installed") as caught:
        tool.execute("info", [], {}, None, 1)
    assert caught.value.code == "android_cli_missing"
    assert not (tmp_path / "missing").exists()


def test_legacy_android_sdk_command_is_not_treated_as_google_cli(tmp_path):
    path = tmp_path / "legacy"
    path.write_text(f"#!{sys.executable}\nprint('Android SDK Tools 25.2.3')\n")
    path.chmod(0o700)
    tool = AndroidCliTools({"cli_path": str(path)})
    assert tool.status()["available"] is True
    assert tool.status()["compatible"] is False
    with pytest.raises(ConfigError) as caught:
        tool.require_available()
    assert caught.value.code == "android_cli_unsupported_version"


def test_host_operation_needs_no_target_and_preserves_literal_arguments(tools, tmp_path):
    query = "compose; $(touch should-not-exist) `echo nope`"
    result = tools.execute("docs.search", [query], {}, str(tmp_path), 5)
    assert result["ok"]
    assert result["data"]["argv"] == ["--no-metrics", "docs", "search", query]
    assert result["data"]["cwd"] == str(tmp_path)
    assert result["stdout"] == ""  # Parsed JSON is returned once, not twice.
    assert not (tmp_path / "should-not-exist").exists()


@pytest.mark.parametrize("option", ["device", "session", "sdk"])
def test_callers_cannot_override_a_lease_or_owned_profile(tools, option):
    with pytest.raises(UsageError, match="AUA owns"):
        tools.validate("layout", [], {option: "different"}, None, 2)


@pytest.mark.parametrize("argument", ["--device=elsewhere", "@response.txt", "bad\x00value"])
def test_positional_arguments_cannot_smuggle_native_options(tools, argument):
    with pytest.raises(UsageError, match="positional strings"):
        tools.execute("input.tap", [argument], {}, None, 2, "fixture-target")


def test_target_operation_must_select_one_explicit_target(tools):
    with pytest.raises(UsageError, match="explicit leased target"):
        tools.execute("layout", [], {}, None, 2)
    result = tools.execute("input.tap", ["text=Continue"], {"wait": 2.5}, None, 2, "fixture-target")
    assert result["data"]["argv"] == [
        "--no-metrics",
        "input",
        "tap",
        "--device=fixture-target",
        "--wait=2.5",
        "text=Continue",
    ]
    assert result["target_id"] == "fixture-target"


def test_host_operation_refuses_target_selection(tools):
    with pytest.raises(UsageError, match="host operation"):
        tools.execute("info", [], {}, None, 2, "fixture-target")


def test_cloud_device_filter_is_not_a_target_override(tools):
    result = tools.execute(
        "signals.crashes.list", [], {"device-filter": ["Fixture Phone", "Sample"]}, None, 2
    )
    argv = result["data"]["argv"]
    assert "--device=Fixture Phone" in argv
    assert "--device=Sample" in argv
    assert "--format=json" in argv
    assert result["target_id"] is None


def test_profile_never_uses_the_only_active_session_default(tools):
    with pytest.raises(UsageError, match="explicit session"):
        tools.execute("profiler.systemtrace.start", [], {}, None, 2, "fixture-target")
    result = tools.execute(
        "profiler.systemtrace.start",
        [],
        {"session": "aua-fixture-owned", "duration": 2},
        None,
        2,
        "fixture-target",
    )
    assert "--session=aua-fixture-owned" in result["data"]["argv"]
    assert result["session_id"] == "aua-fixture-owned"


def test_cleanup_names_exact_session_and_target(tools):
    result = tools.terminate_profile("aua-fixture-owned", "fixture-target")
    assert result["data"]["argv"] == [
        "--no-metrics",
        "profiler",
        "session",
        "terminate",
        "--device=fixture-target",
        "--session=aua-fixture-owned",
    ]


def test_failed_cleanup_does_not_claim_success(tools, monkeypatch):
    def fail(*args, **kwargs):
        raise AndroidCliError(
            "fixture",
            code="android_cli_failed",
            result={"stdout": "", "stderr": "transport failure", "timed_out": False},
        )

    monkeypatch.setattr(tools, "execute", fail)
    with pytest.raises(AndroidCliError):
        tools.terminate_profile("aua-fixture-owned", "fixture-target")


def test_already_finished_profile_cleanup_is_idempotent(tools, monkeypatch):
    def absent(*args, **kwargs):
        raise AndroidCliError(
            "fixture",
            code="android_cli_failed",
            result={
                "stdout": "",
                "stderr": "Session not found: aua-fixture-owned",
                "timed_out": False,
            },
        )

    monkeypatch.setattr(tools, "execute", absent)
    assert (
        tools.terminate_profile("aua-fixture-owned", "fixture-target")["already_finished"] is True
    )


def test_nonzero_exit_retains_bounded_diagnostics(tools):
    with pytest.raises(AndroidCliError) as caught:
        tools.execute("docs.search", ["fail"], {}, None, 2)
    assert caught.value.code == "android_cli_failed"
    assert caught.value.to_dict()["error"]["result"]["exit_code"] == 7
    assert caught.value.result["stderr"] == "fixture failure\n"


def test_large_native_output_is_explicitly_truncated(tools):
    result = tools.execute("docs.search", ["large"], {}, None, 2)
    assert result["stdout_truncated"] and result["stderr_truncated"]
    assert len(result["stdout"]) == android_cli.OUTPUT_LIMIT
    assert len(result["stderr"]) == android_cli.OUTPUT_LIMIT
    assert result["data"] is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_timeout_kills_build_descendants_before_return(tools, tmp_path):
    with pytest.raises(AndroidCliError) as caught:
        tools.execute("docs.search", ["slow"], {}, str(tmp_path), 0.2)
    assert caught.value.code == "android_cli_timeout"
    time.sleep(0.8)
    assert not (tmp_path / "late-child").exists()


def test_preview_result_contains_real_reported_artifacts(tools, tmp_path):
    result = tools.execute(
        "preview.render",
        [":app"],
        {
            "fqn": ["com.example.fixture.One", "com.example.fixture.Two"],
            "component-tree": True,
            "output": str(tmp_path),
        },
        str(tmp_path),
        2,
    )
    assert result["data"] == {"image": str(tmp_path / "preview.png")}
    assert result["artifacts"] == [
        {"path": str(tmp_path / "preview.png"), "size_bytes": 7, "kind": "png"}
    ]


def test_build_arguments_have_an_explicit_separator(tools):
    result = tools.execute(
        "build", [":app"], {"build-args": ["--offline", "-Pfixture=one two"]}, None, 2
    )
    assert result["data"]["argv"][-4:] == [":app", "--", "--offline", "-Pfixture=one two"]


@pytest.mark.parametrize(
    "operation, options",
    [
        ("run", {"debug": True}),
        ("install", {"install-options": "-g"}),
        ("layout", {"flat": "yes"}),
        ("preview.render", {}),
        ("info", {"not-a-flag": "x"}),
    ],
)
def test_unsupported_or_invalid_options_fail_before_execution(tools, operation, options):
    with pytest.raises(UsageError):
        tools.validate(operation, [], options, None, 2)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, 3601])
def test_timeouts_are_finite_and_bounded(tools, timeout):
    with pytest.raises(UsageError, match="timeout_s"):
        tools.validate("info", [], {}, None, timeout)


def test_catalogue_reports_all_families_and_is_not_mutable(tools):
    catalogue = tools.catalogue()
    families = {item["name"].split(".")[0] for item in catalogue}
    assert {
        "preview",
        "profiler",
        "signals",
        "studio",
        "device",
        "sdk",
        "input",
        "screen",
        "build",
        "install",
        "run",
        "docs",
        "skills",
    } <= families
    catalogue[0]["name"] = "changed"
    assert tools.catalogue()[0]["name"] != "changed"
    assert next(x for x in catalogue if x["name"] == "emulator.start")["route"] == "virtual.start"


def test_setup_downloads_only_official_launcher_and_checks_version(tmp_path, monkeypatch):
    destination = tmp_path / "tools" / "android"
    monkeypatch.setattr(android_cli, "_managed_binary", lambda: destination)
    monkeypatch.setattr(android_cli.shutil, "which", lambda _: None)
    monkeypatch.delenv("AUA_ANDROID_CLI", raising=False)
    monkeypatch.setattr(android_cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(android_cli.platform, "machine", lambda: "arm64")
    calls = []

    def download(url, path):
        calls.append(url)
        path.write_bytes(b"\xcf\xfa\xed\xfefixture")

    monkeypatch.setattr(android_cli, "_download", download)
    monkeypatch.setattr(
        android_cli,
        "_run",
        lambda *a, **kw: {
            "stdout": "1.1.16717390\n",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
        },
    )
    result = AndroidCliTools().setup()
    assert result["installed"] and result["compatible"]
    assert calls == ["https://dl.google.com/android/cli/latest/darwin_arm64/android"]
    assert destination.read_bytes().startswith(b"\xcf\xfa\xed\xfe")
    assert destination.stat().st_mode & 0o777 == 0o700


def test_update_cannot_download_an_arbitrary_executable(tools):
    with pytest.raises(UsageError, match="official HTTPS"):
        tools.validate("update", [], {"url": "https://example.test/android"}, None, 2)


def test_download_banner_does_not_hide_structured_operation_result():
    data, logs = android_cli._structured_output('Downloading component...\n{"result":42}\n')
    assert data == {"result": 42}
    assert logs == "Downloading component..."


def test_json_log_fragment_is_not_mistaken_for_structured_result():
    output = '{"status":"running"}\nBuild failed\n'
    assert android_cli._structured_output(output) == (None, output)


def test_documentation_results_are_structured_for_agents():
    output = "Waiting for index...\n1. Fixture topic\n   URL: kb://android/fixture\n   Short excerpt.\n\n2. Second topic\n   URL: kb://android/second\n   Another excerpt.\n"
    result = android_cli._documentation_results(output)
    assert result == {
        "results": [
            {"title": "Fixture topic", "url": "kb://android/fixture", "excerpt": "Short excerpt."},
            {"title": "Second topic", "url": "kb://android/second", "excerpt": "Another excerpt."},
        ]
    }


def test_actual_cli_missing_exact_session_message_is_idempotent(tools, monkeypatch):
    def absent(*args, **kwargs):
        raise AndroidCliError(
            "fixture",
            code="android_cli_failed",
            result={
                "stdout": "",
                "stderr": "error: Session 'aua-fixture-owned' not found.\n",
                "timed_out": False,
            },
        )

    monkeypatch.setattr(tools, "execute", absent)
    assert (
        tools.terminate_profile("aua-fixture-owned", "fixture-target")["already_finished"] is True
    )


@pytest.mark.parametrize(
    "code", ["android_cli_failed", "android_cli_timeout", "development_tool_failed"]
)
def test_daemon_error_roundtrip_retains_native_diagnostics_and_device_exit(code):
    from android_ui_analyser.cli import _daemon_error
    from android_ui_analyser.errors import DevelopmentToolError, ExitCode

    original = AndroidCliError(
        "Fixture native failure",
        code=code,
        result={
            "stderr": "Useful native reason",
            "exit_code": 7,
            "stdout": "",
            "operation": "layout",
        },
    )
    wire_error = json.loads(json.dumps(original.to_dict()["error"]))
    restored = _daemon_error(wire_error)
    assert isinstance(restored, DevelopmentToolError)
    assert restored.exit_code == ExitCode.DEVICE
    assert restored.to_dict() == original.to_dict()


def test_profile_trace_fields_are_value_options_not_flags(tools):
    for operation in ("profiler.systemtrace.prepare", "profiler.systemtrace.start"):
        spec = tools.validate(
            operation, [], {"enable": ["cpu.sched", "gpu"], "disable": "cpu.syscalls"}, None, 30
        )
        assert spec["options"]["enable"]["type"] == "string"
        assert spec["options"]["disable"]["type"] == "string"
        with pytest.raises(UsageError):
            tools.validate(operation, [], {"enable": True}, None, 30)


def test_catalogue_describes_positional_operand_order(tools):
    spec = next(x for x in tools.catalogue() if x["name"] == "input.text")
    assert spec["arguments"]["labels"] == ["<string>", "<selector>"]


def test_daemon_missing_optional_tool_keeps_config_exit_code():
    from android_ui_analyser.cli import _daemon_error
    from android_ui_analyser.errors import ExitCode

    original = ConfigError("Optional tool missing", code="android_cli_missing", hint="Install it")
    restored = _daemon_error(original.to_dict()["error"])
    assert isinstance(restored, ConfigError)
    assert restored.exit_code == ExitCode.CONFIG
    assert restored.to_dict() == original.to_dict()


def test_virtual_stop_requires_explicit_owned_target(tools):
    with pytest.raises(UsageError, match="expects 1"):
        tools.validate("emulator.stop", [], {}, None, 30)
    with pytest.raises(UsageError, match="Unsupported"):
        tools.validate("emulator.remove", ["Fixture"], {"force": True}, None, 30)


def test_helper_preparation_records_absent_apk_without_device_mutation(tools, monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_ui_helper_instance", lambda _: "fixture-boot")
    monkeypatch.setattr(tools, "_ui_helper_forwards", lambda _: [])
    monkeypatch.setattr(
        tools,
        "_helper_command",
        lambda serial, args: calls.append((serial, args)) or {"stdout": ""},
    )
    assert tools.prepare_ui_helper("fixture-target") == {
        "remove_installed_helper": True,
        "instance_token": "fixture-boot",
        "forward_ports": [],
    }
    assert calls == [
        ("fixture-target", ["shell", "pm", "list", "packages", android_cli._UI_HELPER_PACKAGE])
    ]


def test_helper_preparation_preserves_matching_preexisting_apk(tools, monkeypatch):
    monkeypatch.setattr(tools, "_ui_helper_instance", lambda _: "fixture-boot")
    monkeypatch.setattr(tools, "_ui_helper_forwards", lambda _: [])
    package = android_cli._UI_HELPER_PACKAGE
    outputs = iter(
        [
            f"package:{package}\n",
            "package:/data/app/fixture/base.apk\n",
            android_cli._UI_HELPER_HASHES[android_cli.VERIFIED_VERSION]
            + "  /data/app/fixture/base.apk\n",
        ]
    )
    monkeypatch.setattr(tools, "_helper_command", lambda *_: {"stdout": next(outputs)})
    assert tools.prepare_ui_helper("fixture-target") == {
        "remove_installed_helper": False,
        "instance_token": "fixture-boot",
        "forward_ports": [],
    }


def test_helper_preparation_refuses_overwriting_unknown_preexisting_apk(tools, monkeypatch):
    monkeypatch.setattr(tools, "_ui_helper_instance", lambda _: "fixture-boot")
    monkeypatch.setattr(tools, "_ui_helper_forwards", lambda _: [])
    outputs = iter(
        [
            f"package:{android_cli._UI_HELPER_PACKAGE}\n",
            "package:/data/app/fixture/base.apk\n",
            "0" * 64 + "  /data/app/fixture/base.apk\n",
        ]
    )
    monkeypatch.setattr(tools, "_helper_command", lambda *_: {"stdout": next(outputs)})
    with pytest.raises(ConfigError, match="different Android CLI UI helper"):
        tools.prepare_ui_helper("fixture-target")


def test_helper_restore_does_not_require_cli_or_remove_preexisting_apk(monkeypatch):
    tool = AndroidCliTools({"cli_path": "/missing/android"})
    calls = []
    monkeypatch.setattr(tool, "_ui_helper_instance", lambda _: "fixture-boot")
    monkeypatch.setattr(tool, "_ui_helper_forwards", lambda _: [])
    monkeypatch.setattr(tool, "_helper_command", lambda serial, args: calls.append((serial, args)))
    assert tool.release_ui_helper(
        "fixture-target",
        {"remove_installed_helper": False, "instance_token": "fixture-boot", "forward_ports": []},
    )["ok"]
    assert calls == [
        ("fixture-target", ["shell", "am", "force-stop", android_cli._UI_HELPER_PACKAGE])
    ]


def test_helper_restore_removes_only_newly_installed_exact_package(tools, monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_ui_helper_instance", lambda _: "fixture-boot")
    monkeypatch.setattr(tools, "_ui_helper_forwards", lambda _: [])
    installed = iter([True, False])
    monkeypatch.setattr(tools, "_ui_helper_installed", lambda _: next(installed))
    monkeypatch.setattr(tools, "_helper_command", lambda serial, args: calls.append((serial, args)))
    result = tools.release_ui_helper(
        "fixture-target",
        {"remove_installed_helper": True, "instance_token": "fixture-boot", "forward_ports": []},
    )
    assert result["helper_removed"] is True
    assert calls == [
        ("fixture-target", ["shell", "am", "force-stop", android_cli._UI_HELPER_PACKAGE]),
        ("fixture-target", ["uninstall", android_cli._UI_HELPER_PACKAGE]),
    ]


def test_helper_restore_propagates_transport_failure_for_durable_retry(tools, monkeypatch):
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setattr(android_cli.shutil, "which", lambda _: "/fixture/adb")
    calls = []
    monkeypatch.setattr(tools, "_ui_helper_instance", lambda _: "fixture-boot")
    monkeypatch.setattr(tools, "_ui_helper_forwards", lambda _: [])

    def offline(argv, **kwargs):
        calls.append(argv)
        assert kwargs["timeout_s"] == 15
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": "error: device offline",
            "timed_out": False,
            "stdout_truncated": False,
        }

    monkeypatch.setattr(android_cli, "_run", offline)
    with pytest.raises(AndroidCliError, match="restoration failed"):
        tools.release_ui_helper(
            "fixture-target",
            {
                "remove_installed_helper": False,
                "instance_token": "fixture-boot",
                "forward_ports": [],
            },
        )
    assert calls == [
        [
            "/fixture/adb",
            "-s",
            "fixture-target",
            "shell",
            "am",
            "force-stop",
            android_cli._UI_HELPER_PACKAGE,
        ]
    ]


def test_helper_preparation_rejects_unverified_newer_helper_lifecycle(tools, monkeypatch):
    monkeypatch.setattr(tools, "require_available", lambda: "/fixture/android")
    monkeypatch.setattr(tools, "status", lambda: {"version": "1.2.20000000"})
    with pytest.raises(ConfigError, match="has not been verified"):
        tools.prepare_ui_helper("fixture-target")


@pytest.mark.parametrize("recorded", [None, "different-boot"])
def test_helper_restore_never_mutates_a_different_or_unrecorded_boot(tools, monkeypatch, recorded):
    monkeypatch.setattr(tools, "_ui_helper_instance", lambda _: "current-boot")
    monkeypatch.setattr(tools, "_helper_command", lambda *_: pytest.fail("mutated wrong boot"))
    with pytest.raises(ConfigError, match="boot"):
        tools.release_ui_helper(
            "fixture-target",
            {"remove_installed_helper": True, "instance_token": recorded, "forward_ports": []},
        )


def test_integral_float_values_use_native_integer_spelling(tools):
    result = tools.execute(
        "profiler.heapdump.capture",
        [],
        {"session": "aua-fixture-heap", "pid": 1234.0, "timeout": 120.0},
        None,
        5,
        "fixture-target",
    )
    assert "--timeout=120" in result["data"]["argv"]
    assert "--pid=1234" in result["data"]["argv"]


@pytest.mark.parametrize(
    "options", [{}, {"pid": 1234, "process": "com.example.fixture"}, {"pid": 1234, "process": ""}]
)
def test_heap_capture_requires_exactly_one_target_selector_before_execution(
    tools, monkeypatch, options
):
    monkeypatch.setattr(
        tools, "require_available", lambda: pytest.fail("validation used native tool")
    )
    with pytest.raises(UsageError, match="exactly one|requires a string"):
        tools.validate("profiler.heapdump.capture", [], options, None, 30)


@pytest.mark.parametrize("options", [{"pid": 1234}, {"process": "com.example.fixture"}])
def test_heap_capture_accepts_either_single_target_selector(tools, options):
    assert tools.validate("profiler.heapdump.capture", [], options, None, 30)["scope"] == "target"


def test_profiler_saved_artifact_and_owned_next_operation(tools, monkeypatch, tmp_path):
    tools.require_available()
    artifact = tmp_path / "fixture.perfetto-trace"
    artifact.write_bytes(b"trace")
    monkeypatch.setattr(
        android_cli,
        "_run",
        lambda *_, **__: {
            "exit_code": 0,
            "stdout": f"Trace saved at: {artifact}\nNext: android profiler systemtrace stop --session=aua-fixture-profile\n",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "timed_out": False,
            "duration_ms": 1,
        },
    )
    result = tools.execute(
        "profiler.systemtrace.start",
        [],
        {"session": "aua-fixture-profile"},
        None,
        5,
        "fixture-target",
    )
    assert result["artifacts"][0]["path"] == str(artifact)
    assert result["next_operation"] == "profiler.systemtrace.stop"
    assert "Next: android" not in result["stdout"]


def test_helper_forward_cleanup_preserves_prior_and_unrelated_mappings(tools, monkeypatch):
    calls = []
    socket = android_cli._UI_HELPER_SOCKET
    monkeypatch.setattr(tools, "_ui_helper_instance", lambda _: "fixture-boot")

    def command(serial, args):
        calls.append((serial, args))
        return {
            "stdout": (
                f"fixture-target tcp:4001 {socket}\nfixture-target tcp:4002 {socket}\n"
                f"other-target tcp:4003 {socket}\nfixture-target tcp:4004 localabstract:other\n"
                if args == ["forward", "--list"]
                else ""
            )
        }

    monkeypatch.setattr(tools, "_helper_command", command)
    result = tools.release_ui_helper(
        "fixture-target",
        {
            "remove_installed_helper": False,
            "instance_token": "fixture-boot",
            "forward_ports": ["tcp:4001"],
        },
    )
    assert result["forwards_removed"] == ["tcp:4002"]
    assert calls == [
        ("fixture-target", ["shell", "am", "force-stop", android_cli._UI_HELPER_PACKAGE]),
        ("fixture-target", ["forward", "--list"]),
        ("fixture-target", ["forward", "--remove", "tcp:4002"]),
    ]


def test_profile_next_operation_omitted_when_waiting_for_finite_trace(tools):
    assert (
        tools.validate("profiler.systemtrace.prepare", [], {}, None, 30)["next_operation"]
        == "profiler.systemtrace.start"
    )
    assert (
        tools.validate("profiler.systemtrace.start", [], {}, None, 30)["next_operation"]
        == "profiler.systemtrace.stop"
    )
    assert (
        tools.validate("profiler.systemtrace.start", [], {"duration": 1, "wait": True}, None, 30)[
            "next_operation"
        ]
        is None
    )


def test_failed_native_build_exposes_log_path_without_loading_contents(
    tools, monkeypatch, tmp_path
):
    tools.require_available()
    log = tmp_path / "build-fixture.log"
    log.write_text("private compiler detail remains in file")
    monkeypatch.setattr(
        android_cli,
        "_run",
        lambda *_, **__: {
            "exit_code": 1,
            "stdout": f"Build log: {log}\n",
            "stderr": "Compilation failed",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "timed_out": False,
            "duration_ms": 1,
        },
    )
    with pytest.raises(AndroidCliError) as caught:
        tools.execute("build", [":app:assembleDebug"], {}, None, 5)
    assert caught.value.result["artifacts"] == [
        {"path": str(log), "size_bytes": log.stat().st_size, "kind": "log"}
    ]
    assert "private compiler" not in str(caught.value.to_dict())


@pytest.mark.parametrize("exit_code", [1, 2])
def test_failed_native_target_mutation_does_not_claim_no_delivery(tools, monkeypatch, exit_code):
    tools.require_available()
    monkeypatch.setattr(
        android_cli,
        "_run",
        lambda *_, **__: {
            "exit_code": exit_code,
            "stdout": "",
            "stderr": "Native failure",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "timed_out": False,
            "duration_ms": 1,
        },
    )
    with pytest.raises(AndroidCliError) as caught:
        tools.execute("input.text", ["fixture", "text=Field"], {}, None, 5, "fixture-target")
    assert caught.value.result["outcome_unknown"] is True
    assert "do not repeat" in caught.value.result["next_action"]
    with pytest.raises(AndroidCliError) as read_failure:
        tools.execute("layout", [], {}, None, 5, "fixture-target")
    assert read_failure.value.result["outcome_unknown"] is False
