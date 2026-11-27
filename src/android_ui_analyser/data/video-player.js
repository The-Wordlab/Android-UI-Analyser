/* Video player for report pages. Inline it at the end of <body> (or <script src="video-player.js" defer>).
   Every <video controls> becomes a player; add data-native to keep the browser's own controls.
   An element with data-seek="7.5" (or "0:07.5") next to a video becomes a chapter: a mark on the timeline,
   click to play from there, highlighted while that part plays. data-label overrides its timeline label.
   The file is loaded into memory before playing: Drophere ignores Range requests, so the browser's own
   streaming can't seek reliably. Keys when the player has focus: Space/K play, ←/→ one frame (Shift: 1 s),
   < / > speed, [ / ] chapter, 0–9 jump to 0–90 %, F full screen. Hold a step button to keep stepping.
   Portrait clips sit in a phone frame; landscape clips and data-frame="none" get a plain rounded frame.
   The player is sized by its width (the video's class/style move to it); colours follow the page's
   light/dark theme (prefers-color-scheme, or data-theme on <html>).
   Every <img> outside a link and outside [data-no-zoom] opens in a viewer when clicked: fitted to the
   screen, click or wheel or pinch to zoom, drag to pan, ←/→ through the page's images, Esc to close.
   The caption is the image's figcaption, or its alt text. */
(function () {
  "use strict";

  const SPEEDS = [0.25, 0.5, 1, 1.5, 2];
  const ICON = {
    play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z"/></svg>',
    pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>',
    full: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h5v2H7v3H5zM14 5h5v5h-2V7h-3zM5 14h2v3h3v2H5zM17 14h2v5h-5v-2h3z"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.4 5 12 10.6 17.6 5 19 6.4 13.4 12 19 17.6 17.6 19 12 13.4 6.4 19 5 17.6 10.6 12 5 6.4z"/></svg>',
    prev: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15.4 5.4 14 4l-8 8 8 8 1.4-1.4L8.8 12z"/></svg>',
    next: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.6 5.4 10 4l8 8-8 8-1.4-1.4 6.6-6.6z"/></svg>',
    minus: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 11h14v2H5z"/></svg>',
    plus: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z"/></svg>',
  };
  const DARK = "--vp-accent:#8f9bff;--vp-on-accent:#10121b;--vp-panel:#1c1f2a;--vp-ink:#eef0f6;--vp-muted:#9a9fb2;--vp-line:rgba(255,255,255,.1);--vp-rail:rgba(255,255,255,.16);--vp-mark:rgba(255,255,255,.7);--vp-hover:rgba(255,255,255,.08);--vp-shadow:0 12px 32px -14px rgba(0,0,0,.8);--vp-drop:0 26px 50px -20px rgba(0,0,0,.9)";
  const CSS = `
:where(.vp){--vp-accent:#4b5be0;--vp-on-accent:#fff;--vp-panel:#fff;--vp-ink:#1d2030;--vp-muted:#6b7084;--vp-line:rgba(29,32,48,.12);--vp-rail:rgba(29,32,48,.13);--vp-mark:rgba(29,32,48,.5);--vp-hover:rgba(29,32,48,.07);--vp-shadow:0 1px 2px rgba(20,22,40,.06),0 10px 26px -14px rgba(20,22,40,.3);--vp-drop:0 26px 50px -22px rgba(20,22,40,.55)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) :where(.vp){${DARK}}}
:root[data-theme="dark"] :where(.vp){${DARK}}
.vp.vp{position:relative;display:flex;flex-direction:column;gap:12px;overflow:visible;background:none;border-radius:22px;box-shadow:none;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-tap-highlight-color:transparent}
.vp:focus{outline:none}
.vp:focus-visible{outline:2px solid var(--vp-accent);outline-offset:6px}
.vp-device{position:relative;flex:1 1 auto;min-height:0;display:flex;margin:0 3px;padding:8px;border:2px solid transparent;border-radius:10.5%/6%;background:linear-gradient(#07080b,#07080b) padding-box,linear-gradient(140deg,#c3c7d1 0%,#454955 20%,#80858f 46%,#2b2e36 70%,#a7abb6 100%) border-box;box-shadow:0 0 0 1px rgba(0,0,0,.4),var(--vp-drop)}
.vp-device::before,.vp-device::after{content:"";position:absolute;width:3px;border-radius:2px;background:linear-gradient(90deg,#4d515b,#b9bdc7 55%,#5d616b)}
.vp-device::before{right:-6px;top:25%;height:9%}
.vp-device::after{left:-6px;top:15%;height:14%}
.vp-cam{position:absolute;top:1px;left:50%;width:6px;height:6px;margin-left:-3px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#4a5a85,#0b0d14 65%);pointer-events:none}
.vp-stage{position:relative;flex:1 1 auto;min-width:0;min-height:0;display:flex;overflow:hidden;border-radius:8%/4.5%;background:#000;cursor:pointer;isolation:isolate}
.vp.vp video{flex:1 1 auto;min-width:0;min-height:0;width:100%;height:auto;aspect-ratio:var(--vp-ar,9/16);max-width:none;max-height:none;margin:0;padding:0;border:0;border-radius:0;box-shadow:none;object-fit:contain;background:#000;display:block}
.vp-plain .vp-device{margin:0;padding:0;border:1px solid var(--vp-line);border-radius:14px;background:#000;box-shadow:var(--vp-shadow)}
.vp-plain .vp-device::before,.vp-plain .vp-device::after,.vp-plain .vp-cam{display:none}
.vp-plain .vp-stage{border-radius:13px}
.vp-big{position:absolute;left:50%;top:50%;width:64px;height:64px;margin:-32px 0 0 -32px;border-radius:50%;background:var(--vp-accent);box-shadow:0 0 0 7px color-mix(in srgb,var(--vp-accent) 30%,transparent),0 10px 28px rgba(0,0,0,.45);display:grid;place-items:center;pointer-events:none;transition:opacity .2s,transform .2s}
.vp-big svg{width:28px;height:28px;fill:var(--vp-on-accent);margin-left:3px}
.vp-playing .vp-big,.vp-started .vp-big{opacity:0;transform:scale(.85)}
.vp-bar{flex:none;container-type:inline-size;margin:0 3px;padding:4px 10px 6px;border:1px solid var(--vp-line);border-radius:16px;background:var(--vp-panel);color:var(--vp-ink);box-shadow:var(--vp-shadow);font:600 12px/1 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.vp-plain .vp-bar{margin:0}
.vp-track{position:relative;height:24px;cursor:pointer;touch-action:none}
.vp-rail{position:absolute;left:0;right:0;top:10px;height:4px;border-radius:2px;background:var(--vp-rail);overflow:hidden;transition:height .12s,top .12s}
.vp-track:hover .vp-rail{top:9px;height:6px}
.vp-load,.vp-fill{position:absolute;left:0;top:0;bottom:0;width:0}
.vp-load{background:var(--vp-rail)}
.vp-fill{background:var(--vp-accent)}
.vp-knob{position:absolute;top:5px;left:0;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:var(--vp-accent);box-shadow:0 0 0 3px var(--vp-panel),0 1px 4px 2px rgba(0,0,0,.22);pointer-events:none;transition:transform .12s}
.vp-track:hover .vp-knob{transform:scale(1.15)}
.vp-mark{position:absolute;top:7px;width:2px;height:10px;margin-left:-1px;border-radius:1px;background:var(--vp-mark);pointer-events:none}
.vp-tip{position:absolute;bottom:28px;left:0;transform:translateX(-50%);max-width:100%;padding:6px 9px;border-radius:8px;background:#151824;color:#fff;box-shadow:0 6px 18px rgba(0,0,0,.3);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;display:none}
.vp-row{display:flex;align-items:center;gap:1px}
.vp-row button{flex:none;display:grid;place-items:center;height:30px;min-width:28px;padding:0 4px;margin:0;border:0;border-radius:8px;background:none;color:inherit;font:inherit;cursor:pointer}
.vp-row button:hover{background:var(--vp-hover)}
.vp-row button:focus-visible{outline:2px solid var(--vp-accent);outline-offset:-2px}
.vp-row svg{width:18px;height:18px;fill:currentColor}
.vp-row .vp-play{width:32px;min-width:32px;height:32px;margin:0 3px;border-radius:50%;background:var(--vp-accent);color:var(--vp-on-accent);box-shadow:0 3px 10px -3px var(--vp-accent)}
.vp-row .vp-play:hover{background:var(--vp-accent);filter:brightness(1.08)}
.vp-row .vp-play:focus-visible{outline-offset:2px}
.vp-row .vp-speed{min-width:34px;height:24px;margin:0 2px;border-radius:999px;box-shadow:inset 0 0 0 1px var(--vp-line);font-size:11px}
.vp-time{flex:1 1 auto;min-width:0;text-align:center;font-variant-numeric:tabular-nums;font-weight:600;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vp-time span{font-weight:500;color:var(--vp-muted)}
@container (max-width:309px){.vp-row .vp-sec{display:none}}
@container (max-width:279px){.vp-time span{display:none}}
@container (max-width:209px){.vp-row .vp-frame{display:none}}
@container (max-width:159px){.vp-row .vp-speed{display:none}}
.vp.vp:fullscreen{gap:0;border-radius:0;background:#000;${DARK}}
.vp:fullscreen .vp-device{margin:0;padding:0;border:0;border-radius:0;background:#000;box-shadow:none}
.vp:fullscreen .vp-device::before,.vp:fullscreen .vp-device::after,.vp:fullscreen .vp-cam{display:none}
.vp:fullscreen .vp-stage{border-radius:0}
.vp.vp:fullscreen video{aspect-ratio:auto}
.vp:fullscreen .vp-bar{margin:0;border-width:1px 0 0;border-radius:0;box-shadow:none}
[data-seek]{cursor:pointer;border-radius:8px;transition:background-color .15s,box-shadow .15s}
[data-seek]:hover{background:color-mix(in srgb,currentColor 6%,transparent);box-shadow:0 0 0 6px color-mix(in srgb,currentColor 6%,transparent)}
[data-seek].vp-on{background:color-mix(in srgb,currentColor 11%,transparent);box-shadow:0 0 0 6px color-mix(in srgb,currentColor 11%,transparent)}
[data-seek]:focus-visible{outline:2px solid #6c7bf0;outline-offset:6px}
img.iv-zoomable{cursor:zoom-in}
img.iv-zoomable:focus-visible{outline:2px solid #6c7bf0;outline-offset:4px}
.iv.iv{position:fixed;inset:0;z-index:2147483000;background:rgba(9,10,15,.96);-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);color:#eef0f6;font:600 13px/1.2 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;opacity:0;transition:opacity .16s;-webkit-user-select:none;user-select:none;-webkit-tap-highlight-color:transparent}
.iv.iv-open{opacity:1}
.iv-stage{position:absolute;inset:0;overflow:hidden;touch-action:none;cursor:zoom-out}
.iv-stage img{position:absolute;left:0;top:0;max-width:none;max-height:none;margin:0;padding:0;border:0;border-radius:0;box-shadow:0 20px 60px rgba(0,0,0,.5);transform-origin:0 0;cursor:zoom-in;-webkit-user-drag:none}
.iv-zoomed .iv-stage img{cursor:grab}
.iv-dragging .iv-stage img{cursor:grabbing}
.iv-animate .iv-stage img{transition:transform .22s ease}
.iv-top{position:absolute;left:0;right:0;top:0;display:flex;align-items:center;gap:6px;padding:12px;pointer-events:none}
.iv-top>*{pointer-events:auto}
.iv-count{flex:1 1 auto;padding-left:6px;color:#9a9fb2;font-variant-numeric:tabular-nums}
.iv button{display:grid;place-items:center;min-width:40px;height:40px;padding:0 10px;margin:0;border:0;border-radius:999px;background:rgba(255,255,255,.1);color:inherit;font:inherit;cursor:pointer}
.iv button:hover{background:rgba(255,255,255,.18)}
.iv button:focus-visible{outline:2px solid #8f9bff;outline-offset:2px}
.iv [hidden]{display:none}
.iv svg{width:20px;height:20px;fill:currentColor}
.iv .iv-level{min-width:64px;font-variant-numeric:tabular-nums}
.iv .iv-nav{position:absolute;top:50%;width:48px;height:48px;margin-top:-24px;padding:0}
.iv .iv-prev{left:12px}
.iv .iv-next{right:12px}
@media (pointer:coarse){.iv .iv-nav{display:none}}
.iv-caption{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);width:max-content;max-width:min(720px,calc(100% - 32px));padding:8px 14px;border-radius:12px;background:rgba(0,0,0,.6);text-align:center;font-weight:500;line-height:1.35;pointer-events:none}
.iv-caption:empty{display:none}
`;

  function format(seconds, decimals) {
    if (!isFinite(seconds) || seconds < 0) return "–:––";
    const scale = Math.pow(10, decimals);
    const units = Math.floor(seconds * scale + 1e-6);
    const minutes = Math.floor(units / (60 * scale));
    const rest = (units % (60 * scale)) / scale;
    return minutes + ":" + (rest < 10 ? "0" : "") + rest.toFixed(decimals);
  }

  function parseTime(text) {
    let seconds = 0;
    for (const part of String(text).split(":")) seconds = seconds * 60 + parseFloat(part);
    return seconds;
  }

  function node(tag, className, html) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (html) element.innerHTML = html;
    return element;
  }

  function button(className, label, html) {
    const element = node("button", className, html);
    element.type = "button";
    element.title = label;
    element.setAttribute("aria-label", label);
    return element;
  }

  function makePlayer(video) {
    const fps = parseFloat(video.getAttribute("data-fps")) || 30;
    const sourceElement = video.querySelector("source");
    const url = video.currentSrc || video.src || (sourceElement ? sourceElement.src : "");
    const lazy = video.getAttribute("preload") === "none";
    const chapters = [];
    let activeChapter = null;
    let loading = false;
    let loaded = false;
    let wantPlay = false;
    let pendingSeek = -1;
    let scrubbing = false;
    let resumeAfterScrub = false;
    let shownPlaying = null;
    let frameRequest = 0;

    const wrap = node("div", ("vp " + video.className).trim());
    const device = node("div", "vp-device");
    const camera = node("span", "vp-cam");
    const stage = node("div", "vp-stage");
    const big = node("div", "vp-big", ICON.play);
    const bar = node("div", "vp-bar");
    const track = node("div", "vp-track");
    const rail = node("div", "vp-rail");
    const loadFill = node("div", "vp-load");
    const playFill = node("div", "vp-fill");
    const knob = node("div", "vp-knob");
    const tip = node("div", "vp-tip");
    const row = node("div", "vp-row");
    const back = button("vp-sec", "Back 1 s, hold to repeat (Shift+←)", "−1s");
    const prevFrame = button("vp-frame", "Previous frame, hold to keep stepping (←)", "−1f");
    const play = button("vp-play", "Play / pause (Space)", ICON.play);
    const nextFrame = button("vp-frame", "Next frame, hold to keep stepping (→)", "+1f");
    const forward = button("vp-sec", "Forward 1 s, hold to repeat (Shift+→)", "+1s");
    const time = node("span", "vp-time");
    const timeNow = node("b");
    const timeTotal = node("span");
    const speed = button("vp-speed", "Speed (< >)", "1×");
    speed.setAttribute("aria-label", "Speed 1×");
    const full = button("vp-full", "Full screen (F)", ICON.full);

    if (video.getAttribute("style")) wrap.setAttribute("style", video.getAttribute("style"));
    if (video.getAttribute("data-frame") === "none") wrap.classList.add("vp-plain");
    video.removeAttribute("style");
    video.className = "";
    video.removeAttribute("controls");
    video.removeAttribute("src");
    video.querySelectorAll("source").forEach((source) => source.remove());
    video.load();

    video.parentNode.insertBefore(wrap, video);
    stage.append(video, big);
    device.append(stage, camera);
    rail.append(loadFill, playFill);
    track.append(rail, knob, tip);
    track.setAttribute("role", "slider");
    track.setAttribute("aria-label", "Seek");
    time.append(timeNow, timeTotal);
    row.append(back, prevFrame, play, nextFrame, forward, time, speed, full);
    bar.append(track, row);
    wrap.append(device, bar);
    wrap.tabIndex = 0;

    function useShape(width, height) {
      if (!width || !height) return;
      wrap.style.setProperty("--vp-ar", width + " / " + height);
      if (width > height) wrap.classList.add("vp-plain");
    }

    if (video.poster) {
      const posterImage = new Image();
      posterImage.onload = () => useShape(posterImage.naturalWidth, posterImage.naturalHeight);
      posterImage.src = video.poster;
    }

    function duration() {
      return loaded && isFinite(video.duration) ? video.duration : 0;
    }

    function safePlay() {
      const promise = video.play();
      if (promise && promise.catch) promise.catch(() => {});
    }

    function useSource(src) {
      loaded = true;
      loadFill.style.width = "100%";
      video.preload = "auto";
      video.src = src;
      if (wantPlay) safePlay();
      render();
    }

    function startLoading() {
      if (loading || !url) return;
      loading = true;
      const request = new XMLHttpRequest();
      request.open("GET", url);
      request.responseType = "blob";
      request.onprogress = (event) => {
        if (event.lengthComputable) loadFill.style.width = (100 * event.loaded) / event.total + "%";
      };
      request.onload = () => useSource(request.status === 200 ? URL.createObjectURL(request.response) : url);
      request.onerror = () => useSource(url);
      request.send();
      render();
    }

    function togglePlay() {
      if (!loaded) {
        wantPlay = !wantPlay;
        startLoading();
        render();
        return;
      }
      if (video.paused || video.ended) safePlay();
      else video.pause();
    }

    function seek(seconds) {
      const total = duration();
      if (!total) return;
      video.currentTime = Math.max(0, Math.min(total, seconds));
      render();
    }

    function stepFrame(direction) {
      video.pause();
      const frame = Math.floor(video.currentTime * fps + 1e-3) + direction;
      seek((frame + 0.5) / fps);
    }

    function changeSpeed(direction) {
      let index = SPEEDS.indexOf(video.playbackRate);
      if (index < 0) index = SPEEDS.indexOf(1);
      if (direction === 0) index = (index + 1) % SPEEDS.length;
      else index = Math.max(0, Math.min(SPEEDS.length - 1, index + direction));
      video.defaultPlaybackRate = SPEEDS[index];
      video.playbackRate = SPEEDS[index];
      speed.textContent = SPEEDS[index] + "×";
      speed.setAttribute("aria-label", "Speed " + SPEEDS[index] + "×");
    }

    function jumpChapter(direction) {
      const now = video.currentTime;
      let target = direction < 0 ? 0 : -1;
      for (const chapter of chapters) {
        if (direction > 0 && chapter.seconds > now + 0.1) {
          target = chapter.seconds;
          break;
        }
        if (direction < 0 && chapter.seconds < now - 1) target = chapter.seconds;
      }
      if (target >= 0) seek(target);
    }

    function playFrom(seconds) {
      wrap.focus({ preventScroll: true });
      if (!loaded) {
        pendingSeek = seconds;
        wantPlay = true;
        startLoading();
        render();
        return;
      }
      seek(seconds);
      safePlay();
    }

    function toggleFullscreen() {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      } else if (wrap.requestFullscreen) {
        wrap.requestFullscreen();
      } else if (wrap.webkitRequestFullscreen) {
        wrap.webkitRequestFullscreen();
      } else if (video.webkitEnterFullscreen) {
        video.webkitEnterFullscreen();
      }
    }

    function chapterAt(seconds) {
      let found = null;
      for (const chapter of chapters) {
        if (chapter.seconds <= seconds + 0.05) found = chapter;
      }
      return found;
    }

    function render() {
      const total = duration();
      const now = loaded ? video.currentTime : 0;
      const percent = total ? Math.min(100, (100 * now) / total) : 0;
      playFill.style.width = percent + "%";
      knob.style.left = percent + "%";
      track.setAttribute("aria-valuetext", format(now, 2));
      if (loading && !loaded && wantPlay) {
        timeNow.textContent = "Loading…";
        timeTotal.textContent = "";
      } else {
        timeNow.textContent = format(now, 2);
        timeTotal.textContent = " / " + format(total || NaN, 0);
      }

      const playing = loaded ? !video.paused : wantPlay;
      if (playing !== shownPlaying) {
        shownPlaying = playing;
        wrap.classList.toggle("vp-playing", playing);
        play.innerHTML = playing ? ICON.pause : ICON.play;
      }

      const chapter = loaded ? chapterAt(now) : null;
      if (chapter !== activeChapter) {
        if (activeChapter) activeChapter.element.classList.remove("vp-on");
        if (chapter) chapter.element.classList.add("vp-on");
        activeChapter = chapter;
      }
    }

    function drawMarks() {
      const total = duration();
      track.querySelectorAll(".vp-mark").forEach((mark) => mark.remove());
      if (!total) return;
      for (const chapter of chapters) {
        if (chapter.seconds > total) continue;
        const mark = node("div", "vp-mark");
        mark.style.left = (100 * chapter.seconds) / total + "%";
        track.insertBefore(mark, knob);
      }
    }

    function fractionAt(clientX) {
      const box = track.getBoundingClientRect();
      return Math.max(0, Math.min(1, (clientX - box.left) / box.width));
    }

    function showTip(clientX) {
      const total = duration();
      if (!total) return;
      const fraction = fractionAt(clientX);
      const chapter = chapterAt(fraction * total);
      tip.textContent = format(fraction * total, 1) + (chapter ? " · " + chapter.label : "");
      tip.style.display = "block";
      const width = track.offsetWidth;
      const half = tip.offsetWidth / 2;
      tip.style.left = Math.max(half, Math.min(width - half, fraction * width)) + "px";
    }

    function endScrub(event) {
      if (!scrubbing) return;
      scrubbing = false;
      if (event.pointerType !== "mouse") tip.style.display = "none";
      if (resumeAfterScrub) safePlay();
    }

    function loop() {
      render();
      frameRequest = video.paused ? 0 : requestAnimationFrame(loop);
    }

    video.addEventListener("play", () => {
      if (!frameRequest) frameRequest = requestAnimationFrame(loop);
    });
    ["pause", "seeked", "timeupdate", "ended"].forEach((type) => video.addEventListener(type, render));
    ["play", "seeking"].forEach((type) => video.addEventListener(type, () => wrap.classList.add("vp-started")));
    video.addEventListener("loadedmetadata", () => {
      useShape(video.videoWidth, video.videoHeight);
      if (pendingSeek >= 0) video.currentTime = pendingSeek;
      pendingSeek = -1;
      drawMarks();
      render();
    });

    function holdToRepeat(element, action, everyMs) {
      let delay = 0;
      let timer = 0;
      function stop() {
        clearTimeout(delay);
        clearInterval(timer);
      }
      element.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        action();
        delay = setTimeout(() => {
          timer = setInterval(action, everyMs);
        }, 400);
      });
      ["pointerup", "pointerleave", "pointercancel"].forEach((type) => element.addEventListener(type, stop));
      element.addEventListener("click", (event) => {
        if (event.detail === 0) action();
      });
      element.addEventListener("contextmenu", (event) => event.preventDefault());
    }

    stage.addEventListener("click", togglePlay);
    play.addEventListener("click", togglePlay);
    holdToRepeat(back, () => seek(video.currentTime - 1), 250);
    holdToRepeat(forward, () => seek(video.currentTime + 1), 250);
    holdToRepeat(prevFrame, () => stepFrame(-1), 80);
    holdToRepeat(nextFrame, () => stepFrame(1), 80);
    speed.addEventListener("click", () => changeSpeed(0));
    full.addEventListener("click", toggleFullscreen);
    wrap.addEventListener("pointerdown", () => wrap.focus({ preventScroll: true }));

    track.addEventListener("pointerdown", (event) => {
      if (!loaded) {
        startLoading();
        return;
      }
      scrubbing = true;
      resumeAfterScrub = !video.paused;
      video.pause();
      track.setPointerCapture(event.pointerId);
      seek(fractionAt(event.clientX) * duration());
      showTip(event.clientX);
      event.preventDefault();
    });
    track.addEventListener("pointermove", (event) => {
      if (event.pointerType === "mouse" || scrubbing) showTip(event.clientX);
      if (scrubbing) seek(fractionAt(event.clientX) * duration());
    });
    track.addEventListener("pointerup", endScrub);
    track.addEventListener("pointercancel", endScrub);
    track.addEventListener("pointerleave", () => {
      if (!scrubbing) tip.style.display = "none";
    });

    wrap.addEventListener("keydown", (event) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const key = event.key;
      if (key === " " || key === "k") togglePlay();
      else if (key === "ArrowLeft" && event.shiftKey) seek(video.currentTime - 1);
      else if (key === "ArrowRight" && event.shiftKey) seek(video.currentTime + 1);
      else if (key === "ArrowLeft") stepFrame(-1);
      else if (key === "ArrowRight") stepFrame(1);
      else if (key === ",") stepFrame(-1);
      else if (key === ".") stepFrame(1);
      else if (key === "<") changeSpeed(-1);
      else if (key === ">") changeSpeed(1);
      else if (key === "[") jumpChapter(-1);
      else if (key === "]") jumpChapter(1);
      else if (key === "Home") seek(0);
      else if (key === "End") seek(duration());
      else if (key === "f") toggleFullscreen();
      else if (key.length === 1 && key >= "0" && key <= "9") seek((duration() * Number(key)) / 10);
      else return;
      event.preventDefault();
      event.stopPropagation();
    });

    if (!lazy && "IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
        (entries) => {
          if (!entries.some((entry) => entry.isIntersecting)) return;
          observer.disconnect();
          startLoading();
        },
        { rootMargin: "300px" },
      );
      observer.observe(wrap);
    } else if (!lazy) {
      startLoading();
    }
    render();

    return {
      wrap,
      playFrom,
      addChapter(seconds, element, label) {
        chapters.push({ seconds, element, label });
        chapters.sort((a, b) => a.seconds - b.seconds);
        drawMarks();
        render();
      },
    };
  }

  function attachChapters(players) {
    document.querySelectorAll("[data-seek]").forEach((element) => {
      let owner = element.parentElement;
      while (owner && !owner.querySelector(".vp")) owner = owner.parentElement;
      if (!owner) return;
      const player = players.get(owner.querySelector(".vp"));
      const seconds = parseTime(element.getAttribute("data-seek"));
      if (!player || !isFinite(seconds)) return;
      const heading = element.querySelector("b, strong") || element;
      const label = element.getAttribute("data-label") || heading.textContent.trim();
      player.addChapter(seconds, element, label);
      if (element.tabIndex < 0) element.tabIndex = 0;
      element.title = "Play from " + format(seconds, 1);
      element.addEventListener("click", (event) => {
        if (event.target.closest("a")) return;
        player.playFrom(seconds);
      });
      element.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        event.stopPropagation();
        player.playFrom(seconds);
      });
    });
  }

  function captionOf(image) {
    const figure = image.closest("figure");
    const figcaption = figure ? figure.querySelector("figcaption") : null;
    return (figcaption && figcaption.textContent.trim()) || image.alt || "";
  }

  function makeViewer(images) {
    const SIDE = 16;
    const TOP = 64;
    const BOTTOM = 64;
    const overlay = node("div", "iv");
    const stage = node("div", "iv-stage");
    const picture = node("img");
    const bar = node("div", "iv-top");
    const count = node("span", "iv-count");
    const zoomOut = button("iv-out", "Zoom out (−)", ICON.minus);
    const level = button("iv-level", "Fit to the screen (0)", "100%");
    const zoomIn = button("iv-in", "Zoom in (+)", ICON.plus);
    const close = button("iv-close", "Close (Esc)", ICON.close);
    const prev = button("iv-nav iv-prev", "Previous image (←)", ICON.prev);
    const next = button("iv-nav iv-next", "Next image (→)", ICON.next);
    const caption = node("div", "iv-caption");
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Image viewer");
    picture.draggable = false;
    stage.append(picture);
    bar.append(count, zoomOut, level, zoomIn, close);
    overlay.append(stage, bar, prev, next, caption);

    const pointers = new Map();
    let gesture = null;
    let index = 0;
    let x = 0;
    let y = 0;
    let scale = 1;
    let fit = 1;
    let opener = null;
    let pageOverflow = "";

    function zoomed() {
      return scale > fit * 1.01;
    }

    function maxScale() {
      return Math.max(4, fit * 4);
    }

    // Centred while it fits the room between the bars, then the whole screen; panned past that.
    function place(length, start, room, full, position) {
      if (length <= room) return start + (room - length) / 2;
      if (length <= full) return (full - length) / 2;
      return Math.min(0, Math.max(full - length, position));
    }

    function apply(animate) {
      const width = stage.clientWidth;
      const height = stage.clientHeight;
      x = place(picture.naturalWidth * scale, SIDE, width - 2 * SIDE, width, x);
      y = place(picture.naturalHeight * scale, TOP, height - TOP - BOTTOM, height, y);
      overlay.classList.toggle("iv-animate", Boolean(animate));
      overlay.classList.toggle("iv-zoomed", zoomed());
      picture.style.transform = "translate(" + x + "px," + y + "px) scale(" + scale + ")";
      level.textContent = Math.round(scale * 100) + "%";
    }

    function zoomTo(target, px, py, animate) {
      const wanted = Math.min(maxScale(), Math.max(fit, target));
      x = px - ((px - x) * wanted) / scale;
      y = py - ((py - y) * wanted) / scale;
      scale = wanted;
      apply(animate);
    }

    function zoomBy(factor) {
      zoomTo(scale * factor, stage.clientWidth / 2, stage.clientHeight / 2, true);
    }

    function fitToScreen(animate) {
      const width = picture.naturalWidth;
      const height = picture.naturalHeight;
      if (!width || !height) return;
      picture.style.width = width + "px";
      picture.style.height = height + "px";
      const room = Math.min((stage.clientWidth - 2 * SIDE) / width, (stage.clientHeight - TOP - BOTTOM) / height);
      fit = Math.max(0.05, Math.min(1, room));
      scale = fit;
      picture.style.visibility = "";
      apply(animate);
    }

    function show(target) {
      index = (target + images.length) % images.length;
      const source = images[index];
      count.textContent = images.length > 1 ? index + 1 + " / " + images.length : "";
      caption.textContent = captionOf(source);
      picture.alt = source.alt || "";
      picture.style.visibility = "hidden";
      picture.onload = () => fitToScreen(false);
      picture.src = source.currentSrc || source.src;
      if (picture.complete) fitToScreen(false);
    }

    function open(target) {
      opener = document.activeElement;
      pageOverflow = document.documentElement.style.overflow;
      document.documentElement.style.overflow = "hidden";
      document.body.append(overlay);
      prev.hidden = images.length < 2;
      next.hidden = images.length < 2;
      show(target);
      requestAnimationFrame(() => overlay.classList.add("iv-open"));
      close.focus({ preventScroll: true });
    }

    function shut() {
      overlay.classList.remove("iv-open");
      overlay.remove();
      pointers.clear();
      gesture = null;
      document.documentElement.style.overflow = pageOverflow;
      if (opener && opener.focus) opener.focus({ preventScroll: true });
    }

    function local(event) {
      const box = stage.getBoundingClientRect();
      return [event.clientX - box.left, event.clientY - box.top];
    }

    function onPicture(px, py) {
      return px >= x && py >= y && px <= x + picture.naturalWidth * scale && py <= y + picture.naturalHeight * scale;
    }

    function pinchState() {
      const [a, b] = Array.from(pointers.values());
      return { distance: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
    }

    stage.addEventListener("pointerdown", (event) => {
      if (event.button > 0) return;
      stage.setPointerCapture(event.pointerId);
      const point = local(event);
      pointers.set(event.pointerId, point);
      if (pointers.size === 1) {
        gesture = { kind: "pan", start: point, last: point, moved: false };
      } else if (pointers.size === 2) {
        const pinch = pinchState();
        gesture = { kind: "pinch", distance: pinch.distance, mid: pinch.mid, scale };
      }
      event.preventDefault();
    });

    stage.addEventListener("pointermove", (event) => {
      if (!gesture || !pointers.has(event.pointerId)) return;
      const point = local(event);
      pointers.set(event.pointerId, point);
      if (gesture.kind === "pinch") {
        if (pointers.size < 2) return;
        const pinch = pinchState();
        x += pinch.mid[0] - gesture.mid[0];
        y += pinch.mid[1] - gesture.mid[1];
        gesture.mid = pinch.mid;
        zoomTo((gesture.scale * pinch.distance) / gesture.distance, pinch.mid[0], pinch.mid[1], false);
        return;
      }
      if (Math.abs(point[0] - gesture.start[0]) + Math.abs(point[1] - gesture.start[1]) > 6) gesture.moved = true;
      if (zoomed()) {
        x += point[0] - gesture.last[0];
        y += point[1] - gesture.last[1];
        overlay.classList.add("iv-dragging");
        apply(false);
      }
      gesture.last = point;
    });

    function endPointer(event) {
      if (!pointers.delete(event.pointerId)) return;
      overlay.classList.remove("iv-dragging");
      if (!gesture || gesture.kind === "pinch") {
        if (!pointers.size) gesture = null;
        return;
      }
      const point = local(event);
      const dx = point[0] - gesture.start[0];
      const dy = point[1] - gesture.start[1];
      const tapped = !gesture.moved && event.type === "pointerup";
      gesture = null;
      if (tapped && !onPicture(point[0], point[1])) {
        shut();
      } else if (tapped) {
        zoomTo(zoomed() ? fit : Math.max(fit * 2.5, 1), point[0], point[1], true);
      } else if (!zoomed() && event.pointerType !== "mouse" && images.length > 1 && Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy)) {
        show(index + (dx < 0 ? 1 : -1));
      }
    }

    stage.addEventListener("pointerup", endPointer);
    stage.addEventListener("pointercancel", endPointer);
    stage.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const point = local(event);
        const delta = Math.max(-300, Math.min(300, event.deltaMode === 1 ? event.deltaY * 33 : event.deltaY));
        zoomTo(scale * Math.exp(-delta * (event.ctrlKey ? 0.01 : 0.002)), point[0], point[1], false);
      },
      { passive: false },
    );

    zoomIn.addEventListener("click", () => zoomBy(1.5));
    zoomOut.addEventListener("click", () => zoomBy(1 / 1.5));
    level.addEventListener("click", () => fitToScreen(true));
    close.addEventListener("click", shut);
    prev.addEventListener("click", () => show(index - 1));
    next.addEventListener("click", () => show(index + 1));
    window.addEventListener("resize", () => {
      if (overlay.isConnected) fitToScreen(false);
    });

    overlay.addEventListener("keydown", (event) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const key = event.key;
      if (key === "Escape") shut();
      else if (key === "ArrowLeft" && images.length > 1) show(index - 1);
      else if (key === "ArrowRight" && images.length > 1) show(index + 1);
      else if (key === "+" || key === "=") zoomBy(1.5);
      else if (key === "-" || key === "_") zoomBy(1 / 1.5);
      else if (key === "0") fitToScreen(true);
      else if (key === "Tab") {
        const items = Array.from(overlay.querySelectorAll("button:not([hidden])"));
        const at = items.indexOf(document.activeElement);
        items[(at + (event.shiftKey ? -1 : 1) + items.length) % items.length].focus();
      } else return;
      event.preventDefault();
      event.stopPropagation();
    });

    return { open };
  }

  function attachImages() {
    const images = Array.from(document.querySelectorAll("img")).filter(
      (image) => !image.closest("a, .vp, .iv, [data-no-zoom]"),
    );
    if (!images.length) return;
    const viewer = makeViewer(images);
    images.forEach((image, position) => {
      image.classList.add("iv-zoomable");
      if (image.tabIndex < 0) image.tabIndex = 0;
      if (!image.title) image.title = "Open larger";
      image.addEventListener("click", () => viewer.open(position));
      image.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        viewer.open(position);
      });
    });
  }

  function init() {
    if (!document.getElementById("vp-css")) {
      const style = node("style");
      style.id = "vp-css";
      style.textContent = CSS;
      document.head.appendChild(style);
    }
    const players = new Map();
    document.querySelectorAll("video[controls]:not([data-native])").forEach((video) => {
      if (video.closest(".vp")) return;
      const player = makePlayer(video);
      players.set(player.wrap, player);
    });
    attachChapters(players);
    attachImages();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
