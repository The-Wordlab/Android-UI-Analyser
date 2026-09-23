"""Jev is reachable with the OpenRouter key a controller run already holds."""

from experiments.aua_controller.typesafe_cost import OPENROUTER_BASE_URL, client_options


def test_an_openrouter_key_alone_routes_jev_through_openrouter():
    assert client_options({"OPEN_ROUTER_API_KEY": "or-key"}) == {
        "api_key": "or-key", "base_url": OPENROUTER_BASE_URL}


def test_a_typesafe_key_is_left_to_the_sdk():
    assert client_options({"TYPESAFE_API_KEY": "ts", "OPEN_ROUTER_API_KEY": "or"}) == {}


def test_an_explicit_base_url_is_kept():
    options = client_options({"OPEN_ROUTER_API_KEY": "or", "TYPESAFE_BASE_URL": "https://proxy"})
    assert options["base_url"] == "https://proxy"


def test_no_key_leaves_the_sdk_to_report_it():
    assert client_options({}) == {}


def test_the_navigator_builds_its_client_from_those_options(monkeypatch):
    import typesafe_sdk
    from experiments.aua_controller import typesafe_navigator

    seen = {}
    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", lambda **kw: seen.update(kw) or object())
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "or-key")
    typesafe_navigator.TypeSafeNavigator("1. Tap New chat")
    assert seen == {"api_key": "or-key", "base_url": OPENROUTER_BASE_URL}
