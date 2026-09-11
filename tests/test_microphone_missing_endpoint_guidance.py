"""Missing endpoints explain the existing bootstrap opt-in without guessing a cause."""

import pytest

from android_ui_analyser import mic
from android_ui_analyser.errors import DeviceError


def test_missing_endpoint_recommends_audio_bootstrap(tmp_path):
    with pytest.raises(DeviceError) as error:
        mic.discover_emulator_endpoint("emulator-9998", running_dirs=[tmp_path])
    assert error.value.code == "mic_endpoint_missing"
    assert "session start --audio" in error.value.hint
    assert "does not prove" in error.value.hint
