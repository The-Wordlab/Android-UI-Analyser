const status = document.querySelector("#status");
const url = decodeURIComponent(location.hash.slice(1));
status.textContent = url;

function fail(message) {
  status.textContent = message;
  document.title = `AUA could not open the page: ${message}`;
}

async function open() {
  const result = await chrome.runtime.sendMessage({action: "open", url});
  if (result?.error === "Unknown popup action") {
    // A copy loaded before AUA could open tabs. This page is read from disk, so it can still
    // reload that copy; the reload closes this tab, and the next run opens the page.
    chrome.runtime.reload();
    return;
  }
  if (result?.error) fail(result.error);
}

open().catch(error => fail(String(error.message || error)));
