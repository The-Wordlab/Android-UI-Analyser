# Chrome Web Store listing

The attached-Chrome extension is published unlisted on the Chrome Web Store. This page keeps the
listing answers so an update is a copy-paste.

## Package

```bash
scripts/package-chrome-extension.py            # writes dist/aua-chrome-extension-<version>.zip
```

The store sets the extension's identity, so the zip has no manifest `key`. Upload it under
**Package → Upload new package** and submit; Chrome updates installed copies by itself.

## Store listing

- **Category:** Developer Tools · **Language:** English
- **Summary** (from the manifest): Share one explicitly selected Chrome tab with Android UI Analyser.
- **Description:**

  > Let AUA (Android UI Analyser), a command-line UI testing tool, inspect and drive one tab of
  > your own Chrome — the tab where you are already signed in.
  >
  > Choose Attach this tab in the popup, or set a start URL in AUA and it opens the page in a new
  > tab and attaches to it. AUA can then read the page, take screenshots, click, type, scroll and
  > pick from dropdowns, and read that tab's console and network activity. Coding agents use it
  > to test web apps behind a login.
  >
  > Only that one tab is shared, and only while it is attached. The extension cannot see other
  > tabs, cookies, history or downloads. Everything stays on your computer: the extension talks
  > only to the AUA program you installed, through Chrome native messaging.
  >
  > Needs AUA: https://github.com/The-Wordlab/Android-UI-Analyser — run
  > `aua browser extension install` once to connect it.

- **Homepage:** https://github.com/The-Wordlab/Android-UI-Analyser/blob/main/docs/web.md
- **Support:** https://github.com/The-Wordlab/Android-UI-Analyser/issues
- **Images:** the 128 px icon from `icons/attached-128.png`, a 1280×800 screenshot and a
  440×280 small promo tile.

## Privacy practices

- **Single purpose:** Connect one Chrome tab the user approves, or one the local AUA tool opens,
  to the AUA command-line testing tool on the same computer, so it can inspect and operate that
  tab for UI testing.
- **activeTab:** Attaches only to the tab the user picks in the popup, without access to any
  other tab.
- **debugger:** Reads the attached tab's page structure, screenshots, console and network
  activity, and performs the clicks, typing and scrolling of a UI test through the Chrome
  DevTools Protocol. Used on the one attached tab only.
- **nativeMessaging:** Exchanges commands and results with the AUA program the user installed
  on the same computer. There is no server.
- **Remote code:** No. The extension runs only its packaged code. The page scripts it evaluates
  come from the AUA program installed on the user's computer, never from a server.
- **Data usage:** Website content, handled on the user's computer only, never sold, never used
  for anything but the tab automation the user starts, never used for credit or lending.
- **Privacy policy:**
  https://github.com/The-Wordlab/Android-UI-Analyser/blob/main/docs/chrome-extension-privacy.md

## Distribution

Visibility **Unlisted**: anyone with the link can install it; it does not appear in search.
