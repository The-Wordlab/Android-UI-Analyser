"""The controller must be told when the screen it is looking at is not the app under test.

A downstream suite lost a scenario twice this way: the run tapped an attachment affordance into
`com.android.documentsui`, spent its last six steps browsing Downloads, and verified 1 of its 10
criteria. Every observation already carried `screen.package`; nothing ever mentioned it.
"""

from experiments.aua_controller.run_realapp import note_if_off_app

APP = "com.example.appundertest"


def frame(package):
    return {"observation": {"screen": {"package": package, "width": 720, "height": 1280},
                            "elements": []}}


class TestOffAppNote:
    def test_a_foreign_package_is_named_in_the_result(self) -> None:
        payload = frame("com.android.documentsui")
        note_if_off_app(payload, APP)
        note = payload.get("foreground_note", "")
        assert "com.android.documentsui" in note
        assert APP in note

    def test_it_says_what_to_do_about_it(self) -> None:
        payload = frame("com.android.documentsui")
        note_if_off_app(payload, APP)
        assert "return before continuing" in payload["foreground_note"]

    def test_the_app_under_test_is_not_annotated(self) -> None:
        payload = frame(APP)
        note_if_off_app(payload, APP)
        assert "foreground_note" not in payload

    def test_an_observation_without_a_screen_is_left_alone(self) -> None:
        for payload in ({}, {"observation": {}}, {"observation": {"screen": None}}):
            note_if_off_app(payload, APP)
            assert "foreground_note" not in payload

    def test_a_missing_package_is_not_treated_as_foreign(self) -> None:
        """Absence of the field is not evidence of leaving the app."""
        payload = {"observation": {"screen": {"width": 720}, "elements": []}}
        note_if_off_app(payload, APP)
        assert "foreground_note" not in payload

    def test_it_never_raises_on_odd_payloads(self) -> None:
        for payload in (None, [], "text", 7):
            note_if_off_app(payload, APP)   # must not raise
