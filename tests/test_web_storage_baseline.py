"""Storage baselines use features the selected ephemeral browser context supports."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms.web_tools import PlaywrightConnection, WebLaunchOptions


@pytest.mark.parametrize("browser", ["chromium", "firefox", "webkit"])
def test_session_baseline_preserves_supported_storage_without_requesting_webkit_opfs(browser):
    captured = []
    restored = []
    baseline = {
        "cookies": [{"name": "fixture", "value": "original"}],
        "origins": [{"origin": "https://fixture.test", "localStorage": [], "indexedDB": []}],
    }

    def storage_state(**options):
        if browser == "webkit" and options.get("opfs"):
            raise RuntimeError("Unable to serialize OPFS")
        captured.append(options)
        return baseline

    with ThreadPoolExecutor(max_workers=1) as executor:
        connection = PlaywrightConnection(
            executor,
            object(),
            object(),
            object(),
            WebLaunchOptions(browser=browser),
            "https://fixture.test",
        )
        connection._context = SimpleNamespace(storage_state=storage_state)
        connection._page = SimpleNamespace(
            url="https://fixture.test",
            is_closed=lambda: False,
            evaluate=lambda _: {"fixture": "session-value"},
        )
        connection._replace_context = lambda **state: restored.append(state)
        assert connection.session_begin("s1")["ok"]
        assert connection.session_finish("s1")["restored"]

    assert captured == [{"indexed_db": True, "opfs": browser != "webkit", "credentials": False}]
    assert restored == [
        {
            "storage_state": baseline,
            "session_storage": {"fixture": "session-value"},
            "target_url": "https://fixture.test",
        }
    ]


def test_unexpected_storage_errors_still_fail_instead_of_returning_an_empty_baseline():
    def broken_storage(**_options):
        raise RuntimeError("fixture storage transport disconnected")

    with ThreadPoolExecutor(max_workers=1) as executor:
        connection = PlaywrightConnection(
            executor,
            object(),
            object(),
            object(),
            WebLaunchOptions(browser="webkit"),
            "https://fixture.test",
        )
        connection._context = SimpleNamespace(storage_state=broken_storage)
        connection._page = SimpleNamespace(is_closed=lambda: False)
        with pytest.raises(DeviceError, match="fixture storage transport disconnected"):
            connection.session_begin("s1")
        assert not connection._session_baselines
