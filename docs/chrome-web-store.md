# Chrome Web Store listing

The attached-Chrome extension is published unlisted on the Chrome Web Store as item
`fblgfpfhgomcofbecmnpiceiahaebmia` (submitted 2026-11-27, in review). This page keeps the listing
answers so an update is a copy-paste.

The store copy has its own ID; a copy loaded unpacked keeps `bjbfhnhjmddoehdkhjjnlpjfepbfkncb`.
The native host accepts both, and AUA opens tabs through the store copy when Chrome has it.

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

```text
AUA tab access connects one tab of your own Chrome to AUA (Android UI Analyser), so your coding agent can test web apps the way you see them: signed in, with your real data and settings, without you handing it a password.

It lets your agent close the loop on its own work: change the code, open the page, try the feature, read the console and network errors, fix and check again, without you clicking through every step or pasting screenshots back.

AUA works as a command-line tool and as an MCP server, so any agent that runs commands or speaks MCP can use it. Give AUA a URL and it opens the page in a new tab and attaches by itself, or choose Attach this tab in the popup to share a tab you already have open.

Once attached, your agent can:
• read the page structure and text, and take screenshots
• click, type, scroll, drag and choose from dropdowns
• navigate, go back and reload
• read the tab's console messages and network activity to debug what failed

You stay in control. Only that one tab is shared, and only while it is attached. Chrome shows its debugging banner the whole time, and Detach in the popup ends access at once. The extension cannot see other tabs, cookies, history or downloads, and it sends nothing to any server: it talks only to the AUA program on your computer, through Chrome native messaging.

Requires AUA: https://github.com/The-Wordlab/Android-UI-Analyser
Run `aua browser extension install` once to connect it.
```

- **Homepage:** https://github.com/The-Wordlab/Android-UI-Analyser/blob/main/docs/web.md
- **Support:** https://github.com/The-Wordlab/Android-UI-Analyser/issues
- **Images:** the 128 px icon from `icons/attached-128.png`, a 1280×800 screenshot and a
  440×280 small promo tile.

## Test instructions (for the reviewers)

Credentials stay empty. Additional instructions:

```text
No account needed. The extension connects one tab to AUA, a local CLI (macOS/Linux):
1. pipx install "git+https://github.com/The-Wordlab/Android-UI-Analyser"
2. aua browser extension install --no-open
3. printf 'platforms:\n  web:\n    connection: existing-chrome\n    url: https://example.com/\n' > /tmp/aua.yaml
4. aua --config /tmp/aua.yaml --platform web analyze
Chrome opens example.com in a new tab, the AUA icon turns blue and the command prints the page's elements.
```

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
- **Data usage:** Website content and User activity, handled on the user's computer only, never sold, never used
  for anything but the tab automation the user starts, never used for credit or lending.
- **Privacy policy:**
  https://github.com/The-Wordlab/Android-UI-Analyser/blob/main/docs/chrome-extension-privacy.md

## Distribution

Visibility **Unlisted**: anyone with the link can install it; it does not appear in search.
