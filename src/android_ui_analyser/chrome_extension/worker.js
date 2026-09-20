const HOST_NAME = "com.aua.chrome_bridge";
const PROTOCOL = 1;

let attachedTabId = null;
let nativePort = null;
let nativeReady = null;
let resolveNativeReady = null;
let rejectNativeReady = null;

function debuggee() {
  if (attachedTabId === null) throw new Error("No tab is attached");
  return {tabId: attachedTabId};
}

async function cdp(method, params = {}) {
  return await chrome.debugger.sendCommand(debuggee(), method, params);
}

async function pageState() {
  const tab = await chrome.tabs.get(attachedTabId);
  const viewport = await evaluate(
    "({width: window.innerWidth, height: window.innerHeight, title: document.title, url: location.href})"
  );
  return {
    page_id: `tab-${attachedTabId}`,
    tab_id: attachedTabId,
    url: viewport.url || tab.url || "",
    title: viewport.title || tab.title || "",
    viewport: {width: viewport.width, height: viewport.height}
  };
}

async function evaluate(expression, awaitPromise = false) {
  const response = await cdp("Runtime.evaluate", {
    expression,
    awaitPromise,
    returnByValue: true,
    userGesture: true
  });
  if (response.exceptionDetails) {
    const detail = response.exceptionDetails.exception?.description ||
      response.exceptionDetails.text || "JavaScript evaluation failed";
    throw new Error(detail);
  }
  return response.result?.value;
}

function sendToAua(message) {
  if (nativePort) nativePort.postMessage(message);
}

function sendHello() {
  if (attachedTabId === null) return;
  sendToAua({
    type: "hello",
    protocol: PROTOCOL,
    attached: true,
    tab_id: attachedTabId
  });
}

async function detachCurrent() {
  const tabId = attachedTabId;
  attachedTabId = null;
  if (tabId !== null) {
    try {
      await chrome.debugger.detach({tabId});
    } catch (_error) {
      // The tab may already be gone.  The invariant we need is simply no attachment.
    }
  }
}

function connectNative() {
  if (nativePort && nativeReady) return nativeReady;
  nativeReady = new Promise((resolve, reject) => {
    resolveNativeReady = resolve;
    rejectNativeReady = reject;
  });
  nativePort = chrome.runtime.connectNative(HOST_NAME);
  nativePort.onMessage.addListener(message => {
    if (message?.type === "host_ready" && message.protocol === PROTOCOL) {
      resolveNativeReady?.();
      resolveNativeReady = null;
      rejectNativeReady = null;
      return;
    }
    handleAuaMessage(message).catch(error => {
      if (message && message.id !== undefined) {
        sendToAua({
          reply_to: String(message.id),
          ok: false,
          error: {code: "chrome_extension_operation_failed", message: String(error.message || error)}
        });
      }
    });
  });
  nativePort.onDisconnect.addListener(() => {
    // Reading lastError suppresses Chrome's unchecked-runtime-error warning.
    const detail = chrome.runtime.lastError?.message || "native host disconnected";
    rejectNativeReady?.(new Error(`AUA native host unavailable: ${detail}`));
    resolveNativeReady = null;
    rejectNativeReady = null;
    nativeReady = null;
    nativePort = null;
    detachCurrent();
  });
  return nativeReady;
}

async function attachTab(tabId) {
  const tab = await chrome.tabs.get(tabId);
  if (tab.url && !/^https?:\/\//i.test(tab.url)) {
    throw new Error("AUA can attach only to an http(s) page");
  }
  if (attachedTabId !== null && attachedTabId !== tabId) await detachCurrent();
  await connectNative();
  if (attachedTabId === null) {
    await chrome.debugger.attach({tabId}, "1.3");
    attachedTabId = tabId;
    try {
      await cdp("Page.enable");
      await cdp("Runtime.enable");
      await cdp("Network.enable");
      const protocol = await evaluate("location.protocol");
      if (protocol !== "http:" && protocol !== "https:") {
        throw new Error("AUA can attach only to an http(s) page");
      }
    } catch (error) {
      await detachCurrent();
      throw error;
    }
  }
  sendHello();
  return await pageState();
}

function keyDescription(key) {
  const table = {
    Enter: {key: "Enter", code: "Enter", windowsVirtualKeyCode: 13},
    Tab: {key: "Tab", code: "Tab", windowsVirtualKeyCode: 9},
    Escape: {key: "Escape", code: "Escape", windowsVirtualKeyCode: 27},
    Backspace: {key: "Backspace", code: "Backspace", windowsVirtualKeyCode: 8},
    Space: {key: " ", code: "Space", text: " ", windowsVirtualKeyCode: 32},
    ArrowLeft: {key: "ArrowLeft", code: "ArrowLeft", windowsVirtualKeyCode: 37},
    ArrowUp: {key: "ArrowUp", code: "ArrowUp", windowsVirtualKeyCode: 38},
    ArrowRight: {key: "ArrowRight", code: "ArrowRight", windowsVirtualKeyCode: 39},
    ArrowDown: {key: "ArrowDown", code: "ArrowDown", windowsVirtualKeyCode: 40},
    PageUp: {key: "PageUp", code: "PageUp", windowsVirtualKeyCode: 33},
    PageDown: {key: "PageDown", code: "PageDown", windowsVirtualKeyCode: 34}
  };
  return table[key] || {key, code: key};
}

async function pressKey(key) {
  const detail = keyDescription(key);
  await cdp("Input.dispatchKeyEvent", {type: "keyDown", ...detail});
  await cdp("Input.dispatchKeyEvent", {type: "keyUp", ...detail, text: undefined});
}

async function handleAuaMessage(message) {
  if (!message || message.id === undefined || typeof message.method !== "string") return;
  const id = String(message.id);
  const params = message.params || {};
  let result;
  switch (message.method) {
    case "page_state":
      result = await pageState();
      break;
    case "evaluate":
      result = await evaluate(String(params.expression || ""), Boolean(params.await_promise));
      break;
    case "capture_screenshot":
      result = await cdp("Page.captureScreenshot", {format: "png", fromSurface: true});
      result.viewport = await evaluate("({width: window.innerWidth, height: window.innerHeight})");
      break;
    case "click":
      await cdp("Input.dispatchMouseEvent", {type: "mousePressed", x: params.x, y: params.y, button: "left", clickCount: 1});
      await cdp("Input.dispatchMouseEvent", {type: "mouseReleased", x: params.x, y: params.y, button: "left", clickCount: 1});
      result = {};
      break;
    case "long_click":
      await cdp("Input.dispatchMouseEvent", {type: "mousePressed", x: params.x, y: params.y, button: "left", clickCount: 1});
      await new Promise(resolve => setTimeout(resolve, Math.max(1, Number(params.duration_ms) || 600)));
      await cdp("Input.dispatchMouseEvent", {type: "mouseReleased", x: params.x, y: params.y, button: "left", clickCount: 1});
      result = {};
      break;
    case "type_text":
      await cdp("Input.insertText", {text: String(params.text || "")});
      result = {};
      break;
    case "clear_text":
      result = await evaluate(`(() => {
        const element = document.activeElement;
        if (!element || !("value" in element)) return false;
        const prototype = element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
        if (setter) setter.call(element, ""); else element.value = "";
        element.dispatchEvent(new InputEvent("input", {bubbles: true, inputType: "deleteContentBackward"}));
        element.dispatchEvent(new Event("change", {bubbles: true}));
        return true;
      })()`);
      break;
    case "press":
      await pressKey(String(params.key || ""));
      result = {};
      break;
    case "go_back": {
      const history = await cdp("Page.getNavigationHistory");
      const entry = history.entries?.[history.currentIndex - 1];
      if (entry) await cdp("Page.navigateToHistoryEntry", {entryId: entry.id});
      result = {};
      break;
    }
    case "reload":
      await cdp("Page.reload", {ignoreCache: false});
      result = {};
      break;
    case "scroll":
      await cdp("Input.dispatchMouseEvent", {type: "mouseMoved", x: params.x, y: params.y});
      await cdp("Input.dispatchMouseEvent", {
        type: "mouseWheel", x: params.x, y: params.y,
        deltaX: params.delta_x, deltaY: params.delta_y
      });
      result = {};
      break;
    case "goto":
      result = await cdp("Page.navigate", {url: String(params.url || "")});
      break;
    case "wait_idle": {
      const deadline = Date.now() + Math.max(1, Number(params.timeout_ms) || 5000);
      while (Date.now() < deadline) {
        const state = await evaluate("document.readyState");
        if (state === "interactive" || state === "complete") break;
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      result = {};
      break;
    }
    case "focus":
      await chrome.tabs.update(attachedTabId, {active: true});
      result = {};
      break;
    case "detach":
      await detachCurrent();
      result = {detached: true};
      break;
    default:
      throw new Error(`Unsupported AUA extension method: ${message.method}`);
  }
  sendToAua({reply_to: id, ok: true, result});
}

function diagnosticEvent(kind, level, message, url = "") {
  sendToAua({
    type: "event",
    event: {kind, level, message: String(message || ""), url: String(url || ""), timestamp_ms: Date.now()}
  });
}

chrome.debugger.onEvent.addListener((source, method, params) => {
  if (source.tabId !== attachedTabId) return;
  if (method === "Runtime.consoleAPICalled") {
    const message = (params.args || []).map(arg => arg.value ?? arg.description ?? arg.type).join(" ");
    diagnosticEvent("console", params.type === "error" ? "error" : params.type, message);
  } else if (method === "Runtime.exceptionThrown") {
    diagnosticEvent("page_error", "error", params.exceptionDetails?.exception?.description || params.exceptionDetails?.text);
  } else if (method === "Network.loadingFailed") {
    diagnosticEvent("request_failed", "error", params.errorText || "request failed");
  } else if (method === "Network.requestWillBeSent") {
    diagnosticEvent("request", "info", `${params.request?.method || "GET"} ${params.request?.url || ""}`, params.request?.url);
  } else if (method === "Network.responseReceived") {
    diagnosticEvent("response", Number(params.response?.status) >= 400 ? "warning" : "info", `${params.response?.status || ""} ${params.response?.url || ""}`, params.response?.url);
  } else if (method === "Network.webSocketCreated") {
    diagnosticEvent("websocket", "info", `opened ${params.url || ""}`, params.url);
  }
});

chrome.debugger.onDetach.addListener(source => {
  if (source.tabId === attachedTabId) attachedTabId = null;
});

chrome.tabs.onRemoved.addListener(tabId => {
  if (tabId === attachedTabId) attachedTabId = null;
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message?.action === "status") {
      const [active] = await chrome.tabs.query({active: true, currentWindow: true});
      return {
        attached: attachedTabId !== null,
        current_tab: attachedTabId !== null && active?.id === attachedTabId,
        state: attachedTabId === null ? null : await pageState()
      };
    }
    if (message?.action === "attach") {
      let tabId = Number.isInteger(message.tabId) ? message.tabId : null;
      if (tabId === null) {
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        tabId = tab?.id ?? null;
      }
      if (tabId === null) throw new Error("No active tab");
      return {attached: true, current_tab: true, state: await attachTab(tabId)};
    }
    if (message?.action === "detach") {
      await detachCurrent();
      if (nativePort) nativePort.disconnect();
      nativePort = null;
      return {attached: false, current_tab: false};
    }
    throw new Error("Unknown popup action");
  })().then(sendResponse).catch(error => sendResponse({error: String(error.message || error)}));
  return true;
});
