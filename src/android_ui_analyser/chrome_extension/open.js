const status = document.querySelector("#status");
const url = decodeURIComponent(location.hash.slice(1));
status.textContent = url;

async function open() {
  const result = await chrome.runtime.sendMessage({action: "open", url});
  if (result?.error === "Unknown popup action") {
    // A copy loaded before AUA could open tabs. This page is read from disk, so it can still
    // reload that copy; the reload closes this tab, and the next run opens the page.
    chrome.runtime.reload();
    return;
  }
  if (result?.error) status.textContent = result.error;
}

open().catch(() => {
  // The tab has already moved on to the page; nothing is left to show here.
});
