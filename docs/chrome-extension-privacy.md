# AUA Chrome extension — privacy

The AUA Chrome extension connects one Chrome tab to AUA (Android UI Analyser), a command-line
testing tool you install and run on your own computer. This page describes what the extension
does with data.

## What it reads

Only the tab AUA opens for you, and only while it is attached. From that tab it reads what a UI test needs: the page structure and text,
screenshots, and the page's console messages and network activity.

It cannot see other tabs, cookies, browsing history, bookmarks, downloads or the clipboard.

## Where it goes

The extension sends what it reads to the AUA program on the same computer, through Chrome's
native messaging. It sends nothing to the developer or to any server, and it has no analytics.

AUA hands the results to whatever you run it from, for example a terminal or a coding agent you
use, and may save screenshots and logs on your computer as you configure it. That part is under
your control and outside the extension.

## What it stores

Nothing. The extension keeps no data of its own; detaching or closing the tab ends its access.

## Permissions

- `debugger`: read the attached tab and perform the clicks, typing and scrolling AUA asks for.
- `nativeMessaging`: talk to the AUA program on your computer.

## Contact

Questions and reports: [GitHub issues](https://github.com/The-Wordlab/Android-UI-Analyser/issues).
