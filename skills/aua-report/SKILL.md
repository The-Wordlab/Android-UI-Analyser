---
name: aua-report
description: Turn an AUA run's recording and screenshots into a page people can open (video player with chapters, screenshots, steps) and publish it on Drophere, public or limited to an email domain, with an expiry. Use when asked for test evidence, a report page, a pull-request evidence link, or to share what a run showed.
---

# Evidence reports: a page people can open

A verdict is a claim; a page with the recording and screenshots is what lets a person check it.
How the page looks is up to you. AUA gives you the parts every report needs: this guide, a video
player that seeks reliably, and a publish step that sets who can see the page before anything is
uploaded, then checks the result.

## 1. Collect the evidence

- Recording: `aua record start` before the flow and `aua record stop <work>/flow.mp4` after it
  (MCP `screen_record_start` / `screen_record_stop`). Save it outside the bundle: Android keeps
  `flow.mp4.segments/` and `flow.mp4.recording.json` next to it, and those should not be
  published. Copy only the MP4 into `<bundle>/media/`. If you have a tool that cuts the waiting
  (loading, typing, network), use it, and keep each result on screen for about a second.
- Screenshots: `aua screenshot <bundle>/media/01-<state>.png` at each state the page claims
  (MCP `screenshot`).
- Write down the recording time of each step you describe. Those times become chapters.
- Leave out raw logs, HAR files, tokens, account, device and advertising ids, and anything
  personal. Quote only the few log lines that prove a point, with identifiers removed.

## 2. Write the page

Make one folder with `index.html` at the top and the media beside it (`media/...`). Put the CSS
and scripts inline so the page works on its own. Lead with what changed, what was checked and the
verdict, then the evidence.

The video needs only this markup:

```html
<video src="media/flow.mp4" poster="media/poster.jpg" controls playsinline muted preload="metadata"
       aria-label="Recording: ..." style="width:320px"></video>
<ol class="steps">
  <li data-seek="4"><b>Opens the settings</b> <span>...</span></li>
  <li data-seek="0:07.5"><b>Saves the change</b> <span>...</span></li>
</ol>
```

- Every `<video controls>` becomes the player; `data-native` keeps the browser's own controls.
- Size the video by width. The player's height follows the clip, plus about 80 px of controls.
  A portrait clip sits in a phone frame; a landscape clip, or `data-frame="none"`, gets a plain
  rounded frame. Colours follow the page's light/dark theme (`prefers-color-scheme`, or
  `data-theme="light|dark"` on `<html>`).
- `data-seek="7.5"` or `"0:07.5"` on any element makes it a chapter of the nearest video (same
  section): a mark on the timeline, and tapping it plays from there. The step that is playing
  gets the class `vp-on`. The timeline label is the element's `<b>`/`<strong>` text, or
  `data-label`. Say "Tap a step to jump to it." near the steps.
- Add `data-fps` if the clip isn't 30 fps (frame steps use it). `preload="none"` waits for the
  first play; otherwise the file loads when the player scrolls into view.
- Viewers get: a timeline that shows the time and chapter under the pointer, -1 s / -1 frame /
  play / +1 frame / +1 s, speed 0.25x-2x, full screen, and keys (Space play, arrows step,
  `<` `>` speed, `[` `]` chapter, F full screen).

## 3. Add the player

`aua report player <bundle>/index.html` (MCP `report_player`) puts the player into the page just
before `</body>`. Running it again replaces the earlier copy. Pages without a `<video>` are left
alone.

## 4. Check the page

Open it in a browser before publishing, for example with AUA on the web platform. MP4 (H.264)
needs installed Chrome (`platforms.web.channel: chrome`); Playwright's own Chromium cannot play
it. Check that the duration shows, the chapter marks sit on the timeline, tapping a step jumps
there, and nothing spills over at phone width.

## 5. Publish

`aua report publish <bundle> --ttl 7d --allow-domain example.com` (MCP `report_publish`) uploads
the folder to Drophere and prints the link.

- With a Drophere API key (`DROPHERE_API_KEY`, or `API_KEY=...` in `~/.drophere/credentials`)
  the page expires after `--ttl` (default 7d). `--allow-domain` and `--allow-email` limit who can
  open it: a visitor confirms their email with a one-time code. Access is set before the upload,
  so the page is never public on the way.
- Without a key, anyone with the link can open the page, and it expires after 24 hours.
- `--slug <slug>` publishes a new version at the same link. It keeps its original expiry.
- The result gives the link, expiry, who can view it, the status an anonymous visitor got, and
  how many files loaded. `ok: false` means one of those is wrong; fix it before sharing the link.
- Publishing puts the files on an external host. Ask the user before publishing anything internal
  without an allow-list.
