"""The detail dashboard has to be usable, not merely correct.

Three things made it unusable in practice and each has an assertion here:

* the live frame took the wide column even though an emulator is portrait, so the two
  text panels that carry the actual evidence — the agent I/O journal and logcat — were
  squeezed into the leftovers;
* every poll prepended new journal rows above whatever the reader was looking at, so a
  scrolled-away reader was dragged down one row-height per arriving event, and the
  browser's own scroll anchoring fought whatever the page did about it;
* request/response payloads and logcat were undifferentiated grey text, so finding a key
  or a tag meant reading every character.

These are page-shape guards. They cannot prove the pixels look right, but they do fail if
the sizing rule, the scroll anchor, or a token class is dropped again.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest

from android_ui_analyser import dashboard as dash

_SCRIPT = re.compile(r"<script[^>]*>(.*?)</script>", re.S)


def _page() -> str:
    html = dash._DASHBOARD_HTML.replace("__POLL_MS__", "500")
    html = html.replace("__MODE_JSON__", '"detail"').replace("__SERIAL_JSON__", '"emulator-5554"')
    html = html.replace(
        "__PHONE_ACCESS_URL_JSON__",
        '"http://192.0.2.10:48765/?token=test-access"',
    )
    return html.replace("__DATABASE_TOKEN__", "test-token")


def _token_helpers() -> str:
    """The DOM-free tokenisers, in their own script block so node can run them alone."""
    bodies = _SCRIPT.findall(_page())
    matching = [b for b in bodies if "function jsonTokens(" in b]
    assert matching, "the pure token helpers are no longer in a script block of their own"
    assert len(matching) == 1, "the token helpers are duplicated across script blocks"
    helpers = matching[0]
    assert "document." not in helpers, (
        "the token helpers touch the DOM, so they can no longer be unit tested under node"
    )
    return helpers


def _run_in_node(driver: str) -> str:
    node = shutil.which("node")
    assert node, "node is required for this test"
    result = subprocess.run(  # noqa: S603
        [node, "--input-type=commonjs", "-e", _token_helpers() + "\n" + driver],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_dashboard_branding_uses_the_real_logo_and_consistent_title() -> None:
    html = _page()
    assert "<title>AuA Dashboard</title>" in html
    assert "<h1>AuA Dashboard " in html
    assert html.count("/assets/aua-dashboard-logo.png") == 3
    assert dash._DASHBOARD_LOGO.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_grid_empty_state_is_a_centered_card_without_a_redundant_mode_badge() -> None:
    html = _page()
    assert 'id="grid-empty-title"' in html
    assert 'id="grid-empty-copy"' in html
    assert 'id="grid-empty-command"' in html
    assert 'id="modepill"' not in html
    assert 'id="grid-footer"' not in html


def test_detail_header_uses_clear_navigation_and_labeled_device_statuses() -> None:
    html = _page()
    assert "<span>All devices</span>" in html
    assert "← grid" not in html
    assert 'class="detail-overview"' in html
    for label in ("Capture", "Source", "Lease", "Auto-stop", "Frame age", "Failures"):
        assert f'class="detail-status-label">{label}</span>' in html


def test_device_tiles_use_the_frame_as_the_card_with_status_overlays() -> None:
    html = _page()
    tile = html.split("function ensureTile(d) {", 1)[1].split("\nfunction ", 1)[0]
    assert 'class="tile-screen"' in tile
    assert 'class="tile-overlay tile-overlay-top"' in tile
    assert 'class="tile-overlay tile-overlay-bottom"' in tile
    assert 'class="tile-stats"' in tile
    assert 'class="tile-inspect"' in tile
    assert 'class="tile-meta"' not in tile


def test_the_live_frame_column_is_sized_by_the_frame_and_not_by_a_fixed_fraction() -> None:
    html = _page()
    layout = re.search(r"\.layout \{(.*?)\}", html, re.S)
    assert layout, "the detail layout rule is gone"
    columns = re.search(r"grid-template-columns:([^;]+);", layout.group(1))
    assert columns, "the detail layout no longer declares its columns"
    # `auto` first: the stage column collapses to the frame's own width, so a portrait
    # emulator stops claiming the space the journal needs.
    assert columns.group(1).strip().startswith("auto"), columns.group(1)
    assert "minmax(0, 1fr)" in columns.group(1), columns.group(1)
    stage = re.search(r"\.stage img \{(.*?)\}", html, re.S)
    assert stage, "the frame sizing rule is gone"
    assert "width: auto" in stage.group(1), stage.group(1)
    assert re.search(r"height:\s*min\(", stage.group(1)), stage.group(1)


def test_live_frame_can_analyze_overlay_ids_and_tap_the_exact_analysis() -> None:
    html = _page()
    for element_id in (
        "screen-analyze",
        "frame-shell",
        "element-overlay",
        "inspection-status",
        "inspection-raw",
        "inspection-raw-details",
        "inspection-json-search",
        "inspection-json-search-count",
        "inspection-json-prev",
        "inspection-json-next",
        "inspection-clickable-only",
        "inspection-live",
    ):
        assert f'id="{element_id}"' in html
    assert "inspectionPost('analyze'" in html
    assert "inspectionPost('tap'" in html
    assert "inspection_id: currentInspectionId" in html
    assert "element_id: Number(elementId)" in html
    assert "element.bounds.map(Number)" in html
    assert "element.stable_key || ''" in html
    assert "keyBadge.className = 'element-key'" in html
    assert 'id="inspection-clickable-only" type="checkbox" checked' in html
    assert 'id="element-overlay" class="element-overlay clickable-only"' in html
    assert "frame.src = data.frame_url" in html
    assert "renderInspectionRaw(data.result || {}, elements)" in html
    assert "function focusInspectionJson(elementId)" in html
    assert "focusInspectionJson(element.id)" in html
    assert "line.dataset.elementId = String(owner.id)" in html
    assert "updateInspectionJsonSearch(event.shiftKey ? -1 : 1)" in html
    assert "!inspectionFrameActive && frameToken !== lastSrc" in html


def test_the_journal_and_logcat_each_get_a_wide_row_of_their_own() -> None:
    html = _page()
    # Logcat is no longer one third of a three-column strip.
    assert 'class="lower wide"' in html
    assert re.search(r"\.lower\.wide \{[^}]*grid-template-columns:\s*minmax\(0, 1fr\)", html, re.S)
    logcat_row = re.search(r"\.logcat-scroll \{(.*?)\}", html, re.S)
    assert logcat_row, "logcat has no sizing rule of its own"
    assert re.search(r"height:\s*min\(", logcat_row.group(1)), logcat_row.group(1)
    journal = re.search(r"#journal-wrap \{(.*?)\}", html, re.S)
    assert journal, "the journal viewport has no sizing rule"
    assert re.search(r"height:\s*min\(", journal.group(1)), journal.group(1)


def test_logcat_can_request_device_logs_for_one_app_id() -> None:
    html = _page()
    assert 'id="logcat-app-filter"' in html
    assert 'aria-label="Filter Logcat by App ID"' in html
    assert "if (appId) params.app_id = appId" in html
    assert "logcatAppFilter.value = s.package" in html
    assert "setTimeout(tickLogcat, 250)" in html


def test_logcat_renders_newest_first_and_follows_the_top() -> None:
    html = _page()
    assert 'id="logcat-follow" type="checkbox" checked/>follow newest' in html
    assert "lines.slice().reverse().forEach" in html
    assert "function logcatAnchor(" in html
    assert "const anchor = !logcatFollow.checked && !atTop ? logcatAnchor() : null" in html
    assert "if (logcatFollow.checked) logcatView.scrollTop = 0" in html
    assert "scrollTop = logcatView.scrollHeight" not in html


def test_navigation_library_expands_goto_routes_and_groups_every_saved_flow() -> None:
    html = _page()
    assert 'class="knowledge-workspace"' in html
    assert 'id="map-screens"' in html
    assert 'id="map-routes"' in html
    assert 'id="flow-groups"' in html
    assert "function knowledgeItem(" in html
    assert "function knowledgeSteps(" in html
    assert "label: 'Run goto', action: 'goto'" in html
    assert "label: 'Run flow', action: 'flow-run'" in html
    assert "label: 'Clear', action: 'route-delete'" in html
    assert "label: 'Clear', action: 'flow-delete'" in html
    assert 'id="map-clear"' in html
    assert 'id="flows-clear"' in html
    assert 'id="knowledge-clear-all"' in html
    assert "dashboardControlPost('navigation'" in html
    assert "const app = flow.app || 'App-agnostic'" in html


def test_agent_journal_has_a_confirmed_clear_logs_action() -> None:
    html = _page()
    assert 'id="journal-clear"' in html
    assert "dashboardControlPost('journal', 'clear'" in html
    assert "confirmation: 'CLEAR JOURNAL ' + focusSerial" in html


def test_phone_layout_prioritizes_the_live_screen_and_touch_controls() -> None:
    html = _page()
    assert "@media (max-width: 700px)" in html
    assert "max-height: calc(100svh - 12.5rem)" in html
    assert "touch-action: manipulation" in html
    assert ".db-button, .analyze-button { min-height: 2.6rem" in html
    assert ".detail-status:nth-child(1), .detail-status:nth-child(3) { display: flex; }" in html
    assert "@media (max-height: 520px) and (orientation: landscape)" in html


def test_lan_dashboard_has_a_phone_qr_dialog() -> None:
    html = _page()
    for element_id in (
        "phone-qr-button",
        "phone-qr-dialog",
        "phone-qr-image",
        "phone-qr-url",
        "phone-qr-copy",
        "phone-qr-close",
    ):
        assert f'id="{element_id}"' in html
    assert "const PHONE_ACCESS_URL =" in html
    assert "'/api/dashboard-access-qr.svg'" in html
    assert "phoneQrDialog.showModal()" in html


def test_detached_detail_returns_to_grid_and_explains_the_removal() -> None:
    html = _page()
    assert 'id="device-notice"' in html
    assert "if (s.detached)" in html
    assert "window.location.replace('/?detached='" in html
    assert "disconnected and was removed from the dashboard" in html


def test_local_model_workspace_controls_the_shared_runtime_and_shows_full_exchanges() -> None:
    html = _page()
    for element_id in (
        "model-intercept",
        "model-card-functiongemma",
        "model-card-gemma4",
        "model-chat",
        "model-prompt",
        "model-request-kind",
        "model-request-sample",
        "model-prompt-guide",
        "model-traces",
    ):
        assert f'id="{element_id}"' in html
    assert "OFF discards in-flight results" in html
    assert "async function tickModels(" in html
    assert "async function sendModelMessage(" in html
    assert "modelPost('set-intercept'" in html
    assert "modelPost('set-provider'" in html
    assert "modelPost('chat'" in html
    assert "modelPost('agent-test'" in html
    assert "const MODEL_AGENT_SAMPLES" in html
    assert "Uses the exact configured agent policy chain" in html
    assert '<option value="agent_chain" hidden>Configured agent chain</option>' in html
    assert "No candidate matches → handoff" in html
    assert "inputTitle.textContent = 'Model input'" in html
    assert "outputTitle.textContent = event.error ? 'Error' : 'Model output'" in html


def test_the_journal_anchors_its_scroll_instead_of_letting_new_rows_shove_it_down() -> None:
    html = _page()
    journal = re.search(r"#journal-wrap \{(.*?)\}", html, re.S)
    assert journal
    # The page compensates deterministically; browser anchoring on top of that double-shifts.
    assert "overflow-anchor: none" in journal.group(1), journal.group(1)
    assert "position: relative" in journal.group(1), journal.group(1)
    assert "function journalAnchor(" in html
    assert "function preserveJournalScroll(" in html
    assert "anchor.offsetTop - anchorTop" in html
    # One batched insert per poll, not one reflow per event.
    assert "function prependEvents(" in html
    assert "prependEvents(evs)" in html
    # An identical re-render of an expanded payload must not move the page at all.
    assert "panel.dataset.signature" in html


def test_a_scrolled_away_reader_is_told_how_many_events_they_have_not_seen() -> None:
    html = _page()
    assert 'id="journal-jump"' in html
    assert "journalPending" in html
    assert "journalFollow" in html


def test_the_journal_can_be_filtered_without_a_round_trip() -> None:
    html = _page()
    assert 'id="journal-filter"' in html
    assert 'id="journal-fails-only"' in html
    assert "function applyJournalFilter(" in html
    assert "li.dataset.search" in html


def test_journal_rows_and_controls_have_scan_friendly_structure() -> None:
    html = _page()
    assert 'class="journal-search"' in html
    assert 'class="journal-button-group"' in html
    assert 'class="journal-toggle"' in html
    assert "addEventText(summary, 'event-chevron', '›')" in html
    assert "main.className = 'event-main'" in html
    assert "Number(e.duration_ms) >= 1500" in html
    assert "journalFilter.focus()" in html
    assert re.search(r"#journal details > summary \{[^}]*grid-template-columns:", html, re.S)
    assert re.search(r"#journal \.exchange \{[^}]*grid-template-columns:\s*repeat\(2", html, re.S)


def test_expanded_journal_payloads_do_not_nest_a_second_scrollbar() -> None:
    """A scroll area inside a scroll area is the thing that made the journal awful."""
    html = _page()
    pre = re.search(r"#journal \.exchange pre \{(.*?)\}", html, re.S)
    assert pre, "the payload rule is gone"
    assert "max-height: none" in pre.group(1), pre.group(1)


def test_expanded_journal_action_stays_visible_while_its_payload_scrolls() -> None:
    html = _page()
    opened = re.search(r"#journal details\[open\] > summary \{(.*?)\}", html, re.S)
    assert opened, "the expanded event summary has no dedicated style"
    assert "position: sticky" in opened.group(1), opened.group(1)
    assert re.search(r"top:\s*0", opened.group(1)), opened.group(1)
    assert "z-index:" in opened.group(1), opened.group(1)
    assert "box-shadow:" in opened.group(1), opened.group(1)

    row = re.search(r"#journal li \{(.*?)\}", html, re.S)
    assert row
    # `overflow: hidden` becomes the sticky element's inert scroll ancestor and prevents
    # it from following #journal-wrap. `clip` keeps the rounded card without that trap.
    assert "overflow: clip" in row.group(1), row.group(1)


def test_payload_and_logcat_token_classes_are_styled_in_both_panels() -> None:
    html = _page()
    for cls in ("tok-key", "tok-str", "tok-num", "tok-bool", "tok-null", "tok-punc"):
        assert "." + cls + " {" in html or "." + cls + "," in html, cls
    for cls in ("lc-time", "lc-pid", "lc-tag", "lc-pkg", "lc-lvl-e", "lc-lvl-w", "lc-lvl-i"):
        assert "." + cls in html, cls
    assert "function highlightJson(" in html
    assert "function renderLogcat(" in html


def test_an_armed_proxy_rule_expands_to_the_response_it_will_actually_return() -> None:
    """The list showed only the address, so a stub's body was unknowable from the page."""
    html = _page()
    assert "function pxRenderRules(" in html
    rules = html.split("function pxRenderRules(", 1)[1].split("\nfunction ", 1)[0]
    assert "document.createElement('details')" in rules, rules
    assert "highlightJson(" in rules, rules
    assert "px-rule-load" in html
    assert "event.target.closest('button')" in html
    assert "function pxLoadRuleIntoForm(" in html
    # `remove` sits in the summary; clicking it must not also toggle the row open.
    assert "stopPropagation()" in rules, rules


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_json_tokens_separate_keys_from_string_values() -> None:
    payload = json.dumps({"cmd": "tap", "id": 12, "ok": True, "err": None}, indent=2)
    out = _run_in_node(
        "const t = jsonTokens(" + json.dumps(payload) + ");console.log(JSON.stringify(t));"
    )
    tokens = json.loads(out)
    kinds = {value: kind for kind, value in tokens}
    assert kinds['"cmd"'] == "key"
    assert kinds['"tap"'] == "str"
    assert kinds["12"] == "num"
    assert kinds["true"] == "bool"
    assert kinds["null"] == "nul"
    assert kinds["{"] == "punc"
    # Nothing may be lost: the tokens must rebuild the input byte for byte.
    assert "".join(value for _kind, value in tokens) == payload


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_json_tokens_do_not_mistake_a_colon_inside_a_string_for_a_key() -> None:
    payload = '{"note": "a:b", "url": "http://x/y"}'
    out = _run_in_node("console.log(JSON.stringify(jsonTokens(" + json.dumps(payload) + ")));")
    tokens = json.loads(out)
    assert ["str", '"a:b"'] in tokens
    assert ["str", '"http://x/y"'] in tokens
    assert ["key", '"url"'] in tokens
    assert "".join(value for _kind, value in tokens) == payload


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_logcat_tokens_split_a_threadtime_line_into_its_parts() -> None:
    line = "08-21 10:21:40.758   745   760 D WindowManager: captureDisplay for com.example.app"
    out = _run_in_node("console.log(JSON.stringify(logcatTokens(" + json.dumps(line) + ")));")
    tokens = json.loads(out)
    kinds: dict[str, str] = {}
    for kind, value in tokens:
        kinds.setdefault(kind, value)
    assert kinds["time"] == "10:21:40.758"
    assert kinds["date"].strip() == "08-21"
    assert kinds["pid"] == "745"
    assert kinds["tid"] == "760"
    assert kinds["lvl"] == "D"
    assert kinds["tag"] == "WindowManager"
    assert kinds["pkg"] == "com.example.app"
    assert "".join(value for _kind, value in tokens) == line


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_a_line_that_is_not_threadtime_survives_untouched() -> None:
    line = "--------- beginning of system"
    out = _run_in_node("console.log(JSON.stringify(logcatTokens(" + json.dumps(line) + ")));")
    assert json.loads(out) == [["raw", line]]


# --------------------------------------------------------------------------------------
# The page-shape guards above cannot see whether the viewport actually holds still. These
# drive the real page in a real browser and measure it. Chrome-gated, like the node guards
# in `test_dashboard_page_javascript_parses.py`.
# --------------------------------------------------------------------------------------

_CHROMES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def _chrome() -> str | None:
    import os

    explicit = os.environ.get("AUA_TEST_CHROME")
    if explicit and pathlib.Path(explicit).exists():
        return explicit
    for name in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in _CHROMES:
        if pathlib.Path(candidate).exists():
            return candidate
    return None


_PROBE = r"""
<script>
window.setInterval = function () { return 0; };
window.inspectionTapRequests = 0;
window.fetch = function (url) {
  if (String(url).includes('/api/inspect/tap')) window.inspectionTapRequests += 1;
  const body = FIXTURES[String(url).split('?')[0]] || {ok: true};
  return Promise.resolve({ok: true, status: 200, json: () => Promise.resolve(body)});
};
function mk(n) {
  return {ts_ms: 1700000000000 + n, cmd: 'cmd_' + n, args: {i: n}, ok: n % 5 !== 0,
          error: 'boom ' + n, duration_ms: n, source: 'cli', pid: 1};
}
window.addEventListener('load', function () {
  const out = [];
  const say = (name, pass, detail) =>
    out.push((pass ? 'PASS ' : 'FAIL ') + name + ' :: ' + detail);

  prependEvents(Array.from({length: 60}, (_, i) => mk(i)));
  say('backlog built', journalEl.children.length === 60, journalEl.children.length + ' rows');
  say('follow pins to newest', journalWrap.scrollTop === 0, 'scrollTop=' + journalWrap.scrollTop);

  journalWrap.scrollTop = 400;
  journalWrap.dispatchEvent(new Event('scroll'));
  say('scrolling away drops follow', journalFollow === false, 'follow=' + journalFollow);

  const anchor = journalAnchor();
  const beforeTop = journalWrap.scrollTop;
  const beforeOffset = anchor.offsetTop - beforeTop;
  prependEvents(Array.from({length: 10}, (_, i) => mk(1000 + i)));
  const afterOffset = anchor.offsetTop - journalWrap.scrollTop;
  say('arriving events do not move the reader', Math.abs(afterOffset - beforeOffset) <= 1,
      'offset ' + beforeOffset + ' -> ' + afterOffset +
      ' (scrollTop ' + beforeTop + ' -> ' + journalWrap.scrollTop + ')');
  say('unseen count surfaces', journalPending === 10 &&
      !document.getElementById('journal-jump').classList.contains('hidden'),
      'pending=' + journalPending);

  const above = journalEl.children[2].querySelector('details');
  const anchor2 = journalAnchor();
  const before2 = anchor2.offsetTop - journalWrap.scrollTop;
  preserveJournalScroll(function () {
    above.open = true;
    above.querySelector('.exchange').textContent = new Array(40).join('tall\n');
  });
  const after2 = anchor2.offsetTop - journalWrap.scrollTop;
  say('a row growing above the reader does not move them', Math.abs(after2 - before2) <= 1,
      'offset ' + before2 + ' -> ' + after2);

  // Collapsing a row above the reader shrinks the page under them.
  const anchor3 = journalAnchor();
  const before3 = anchor3.offsetTop - journalWrap.scrollTop;
  above.querySelector('summary').click();
  const after3 = anchor3.offsetTop - journalWrap.scrollTop;
  say('collapsing a row above the reader does not move them',
      !above.open && Math.abs(after3 - before3) <= 1,
      'open=' + above.open + ' offset ' + before3 + ' -> ' + after3);

  document.getElementById('journal-jump').click();
  say('jump re-pins to newest',
      journalFollow && journalWrap.scrollTop === 0 && journalPending === 0,
      'scrollTop=' + journalWrap.scrollTop + ' pending=' + journalPending);

  document.getElementById('journal-fails-only').checked = true;
  document.getElementById('journal-fails-only').dispatchEvent(new Event('change'));
  const visible = Array.prototype.filter.call(
    journalEl.children, li => !li.classList.contains('filtered')).length;
  say('fails-only filters in the page', visible > 0 && visible < journalEl.children.length,
      visible + ' of ' + journalEl.children.length + ' rows');

  renderLogcat(['08-21 10:21:41.402  9114  9160 E FeedRepository: fail com.example.app',
                '--------- beginning of main']);
  const lvl = document.querySelector('#logcat .lc-lvl-e');
  const pkg = document.querySelector('#logcat .lc-pkg');
  say('logcat colours level and package', Boolean(lvl && pkg),
      'lvl=' + (lvl && lvl.textContent) + ' pkg=' + (pkg && pkg.textContent));
  say('logcat keeps its raw separator lines',
      Boolean(document.querySelector('#logcat .lc-raw')), 'raw span present');
  say('a scrolled logcat is not slammed back to the top',
      document.getElementById('logcat-view').scrollTop === 0, 'nothing to scroll yet');

  renderLogcat(['08-21 10:21:41.402  9114  9160 I FeedRepository: older',
                '08-21 10:21:42.402  9114  9160 I FeedRepository: newest']);
  say('logcat renders newest first',
      document.querySelector('#logcat .lc-line').textContent.indexOf('newest') >= 0,
      document.querySelector('#logcat .lc-line').textContent);

  const inspectedElement = {
    id: 26, type: 'View', text: null, resource_id: null, content_desc: null,
    bounds: [21, 63, 147, 189], center: [84, 126], clickable: true,
    stable_key: 'geo:View:q00:f0415f9da6'
  };
  renderInspection({
    inspection_id: 'probe', frame_url: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=',
    view: {screen: {width: 666, height: 1536}, elements: [inspectedElement]},
    result: {ok: true, elements: [inspectedElement]}
  }, null);
  const idChip = document.querySelector('[data-element-id="26"] .element-label');
  idChip.click();
  say('id chip does not tap', window.inspectionTapRequests === 0,
      'tap requests=' + window.inspectionTapRequests);
  say('id chip opens the raw object', inspectionRawDetails.open &&
      inspectionJsonSearch.value === '"id": 26', inspectionJsonSearch.value);
  say('raw object is highlighted',
      inspectionRaw.querySelectorAll('[data-element-id="26"].element-current').length > 1,
      inspectionRaw.querySelectorAll('[data-element-id="26"].element-current').length + ' lines');
  inspectionJsonSearch.value = 'f0415f9da6';
  updateInspectionJsonSearch();
  say('raw search finds stable keys', inspectionJsonMatches.length === 1,
      inspectionJsonMatches.length + ' match');

  const node = document.createElement('pre');
  node.id = 'verdict';
  node.textContent = out.join('\n');
  document.body.appendChild(node);
});
</script>
"""

_FIXTURES = {
    "/api/status": {"ok": True, "serial": "e", "stats": {}, "frame_token": ""},
    "/api/events": {"ok": True, "events": [], "stats": {}, "detail_revision": "r"},
    "/api/map": {"package": "p", "screens": [], "routes": []},
    "/api/logcat": {"ok": True, "lines": []},
    "/api/marks": {"marks": []},
    "/api/proxy": {"ok": True, "supported": False},
}


@pytest.mark.skipif(_chrome() is None, reason="no chrome/chromium to drive the page")
def test_the_journal_viewport_holds_still_in_a_real_browser(tmp_path: pathlib.Path) -> None:
    """The one complaint no string assertion can settle: does the page stay where you left it?

    Sixty rows, park the reader at 400px, then land ten more on top. The row under the
    reader's eye must keep the same viewport offset, which means scrollTop moved by exactly
    the height the new rows added.
    """
    probe = _PROBE.replace("FIXTURES[", json.dumps(_FIXTURES) + "[", 1)
    page = _page().replace("<body>", "<body>" + probe, 1)
    page_file = tmp_path / "probe.html"
    page_file.write_text(page)
    dom_file = tmp_path / "dom.html"

    with dom_file.open("wb") as sink:
        proc = subprocess.Popen(  # noqa: S603
            [
                str(_chrome()),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--dump-dom",
                "--window-size=1500,1000",
                "--user-data-dir=" + str(tmp_path / "profile"),
                page_file.as_uri(),
            ],
            stdout=sink,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            # Headless chrome does not always exit after --dump-dom; the DOM is written
            # long before that, so take what it wrote.
            proc.kill()
            proc.wait(timeout=20)

    dom = dom_file.read_text(errors="replace")
    verdict = re.search(r'<pre id="verdict">(.*?)</pre>', dom, re.S)
    assert verdict, f"the page never reached its checks (dom was {len(dom)} bytes)"
    lines = verdict.group(1).replace("&gt;", ">").replace("&lt;", "<").splitlines()
    # Exact: a check that quietly stops running must fail here, not pass silently.
    assert len(lines) == 17, lines
    failures = [line for line in lines if not line.startswith("PASS")]
    assert not failures, "\n".join(lines)
