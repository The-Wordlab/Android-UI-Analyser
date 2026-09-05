from __future__ import annotations

import pytest

from android_ui_analyser.platforms.geometry import DisplayGeometry


def test_retina_points_map_to_screenshot_pixels_and_back() -> None:
    geometry = DisplayGeometry.scaled(
        native_size=(390.0, 844.0),
        canonical_size=(1170, 2532),
    )

    assert geometry.to_canonical((10.0, 20.0)) == (30.0, 60.0)
    assert geometry.to_native((30.0, 60.0)) == (10.0, 20.0)
    assert geometry.bounds_to_canonical((10.0, 20.0, 30.0, 50.0)) == (
        30,
        60,
        90,
        150,
    )


def test_rotated_affine_mapping_encloses_all_rectangle_corners() -> None:
    # Native portrait (100x200) rotated clockwise into a 400x200 screenshot at 2x scale.
    geometry = DisplayGeometry(
        native_size=(100.0, 200.0),
        canonical_size=(400, 200),
        native_to_canonical=(0.0, 2.0, -2.0, 0.0, 400.0, 0.0),
    )

    assert geometry.to_canonical((10.0, 20.0)) == (360.0, 20.0)
    assert geometry.to_native((360.0, 20.0)) == (10.0, 20.0)
    assert geometry.bounds_to_canonical((10.0, 20.0, 30.0, 50.0)) == (
        300,
        20,
        360,
        60,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"canonical_size": (0, 10), "native_size": (10.0, 10.0)},
        {"canonical_size": (10, 10), "native_size": (0.0, 10.0)},
        {
            "canonical_size": (10, 10),
            "native_size": (10.0, 10.0),
            "native_to_canonical": (1.0, 2.0, 2.0, 4.0, 0.0, 0.0),
        },
    ],
)
def test_invalid_geometry_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DisplayGeometry(**kwargs)  # type: ignore[arg-type]
