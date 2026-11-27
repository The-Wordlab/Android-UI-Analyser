const status = document.querySelector("#status");
const detach = document.querySelector("#detach");

function render(result) {
  if (result?.error) {
    status.textContent = result.error;
    return;
  }
  const attached = Boolean(result?.attached);
  status.textContent = attached
    ? `AUA is using: ${result.state?.title || result.state?.url || "a tab"}`
    : "Not in use. Your agent opens the page it needs by itself.";
  document.querySelector("#logo").src = attached ? "icons/attached-32.png" : "icons/idle-32.png";
  detach.hidden = !attached;
}

async function send(action) {
  detach.disabled = true;
  try {
    render(await chrome.runtime.sendMessage({action}));
  } catch (error) {
    render({error: String(error.message || error)});
  } finally {
    detach.disabled = false;
  }
}

detach.addEventListener("click", () => send("detach"));
send("status");
