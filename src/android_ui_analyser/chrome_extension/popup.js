const status = document.querySelector("#status");
const attach = document.querySelector("#attach");
const detach = document.querySelector("#detach");

function render(result) {
  if (result?.error) {
    status.textContent = result.error;
    return;
  }
  const attached = Boolean(result?.attached);
  const currentTab = Boolean(result?.current_tab);
  status.textContent = currentTab
    ? `Attached: ${result.state?.title || result.state?.url || "this tab"}`
    : attached
      ? `Another tab is attached: ${result.state?.title || result.state?.url || "Chrome tab"}`
      : "This tab is not attached.";
  attach.hidden = currentTab;
  attach.textContent = attached ? "Switch AUA to this tab" : "Attach this tab";
  detach.hidden = !attached;
}

async function send(action) {
  attach.disabled = true;
  detach.disabled = true;
  status.textContent = action === "attach" ? "Attaching…" : "Detaching…";
  try {
    render(await chrome.runtime.sendMessage({action}));
  } catch (error) {
    render({error: String(error.message || error)});
  } finally {
    attach.disabled = false;
    detach.disabled = false;
  }
}

attach.addEventListener("click", () => send("attach"));
detach.addEventListener("click", () => send("detach"));
send("status");
