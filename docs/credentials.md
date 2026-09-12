# Enter API keys privately

Use AUA's credential dialog when an agent needs an API key or another secret stored in a
project's `.env` file. The agent supplies the variable name and destination; the user types
the value directly into a masked field and clicks **Save**. The response never contains the
value. No Android device, AUA session, model, or remote API call is involved.

## Save a missing key

Run this on the computer where the user can see the desktop:

```bash
aua config secret OPEN_ROUTER_API_KEY --env-file /absolute/project/.env
```

The dialog shows the variable name and destination. Enter the key and click **Save**, or
click **Cancel** to leave the file unchanged. A successful response looks like:

```json
{"ok":true,"status":"saved","name":"OPEN_ROUTER_API_KEY","env_file":"/absolute/project/.env"}
```

Use the variable name that the consuming program expects. `OPEN_ROUTER_API_KEY` is an
example for a consumer that uses that spelling; this operation does not select a provider,
enable an experimental harness, or configure a model. The same command works for names such
as `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `EXAMPLE_SERVICE_TOKEN`.

If that variable already has a nonempty value in the destination, AUA returns
`status: "already_set"` without opening a dialog. This checks presence, not provider validity.
To replace it, explicitly request another dialog:

```bash
aua config secret OPEN_ROUTER_API_KEY --env-file /absolute/project/.env --replace
```

`--env-file` defaults to `.env` in the command's working directory. Prefer an absolute path
when an agent launches the command. `--timeout 300` sets how long the dialog may remain open
in seconds. Exit code `0` means saved or already set, `130` means cancelled, and `1` means an
error; the JSON result distinguishes those outcomes. Timeout and failure do not authorize a
fallback that asks the user to put the key into chat.

## Use from an MCP agent

An agent connected to AUA can call:

```json
{
  "name": "OPEN_ROUTER_API_KEY",
  "env_file": "/absolute/project/.env",
  "replace": false,
  "timeout_s": 300
}
```

The tool name is `credential_request`. There is deliberately no `value`, `api_key`, stdin,
or clipboard argument. The tool returns the same status JSON as the CLI and bypasses device
sessions, device journals, and screenshots. It does not attach an AUA observation or image.
When the user cancels or the operation fails, the MCP response has `isError: true`.

The dialog opens on the **MCP server's computer**. A remote server cannot open a dialog on
the client's desktop. macOS uses a native hidden-input dialog; other supported desktop
environments use Tk when available. A headless host or missing graphical dialog support
returns an error. Run the CLI on the intended desktop host and destination instead; do not
fall back to requesting the value in the conversation or terminal arguments.

## Load the saved file in the consumer

The `.env` file is configuration **data**. Saving it does not change an agent's environment,
a parent shell, the MCP server, or an already running AUA daemon. Your program must load
the exact file before using the key. Do not print, inspect through an agent tool, or shell
`source` the file: literal characters such as `$`, quotes, and backticks belong to the value.

For Python consumers using `python-dotenv`, disable interpolation:

```python
from dotenv import load_dotenv

load_dotenv("/absolute/project/.env", interpolate=False, override=False)
# The consumer's existing SDK can now read its expected environment variable.
```

`override=False` preserves any variable already present in that process. If you deliberately
rotated the key, restart the consumer with the new configuration or explicitly replace that
one variable. Restart an existing AUA daemon only when that daemon actually uses the changed
provider configuration and its current work can be stopped.

For a child process that should receive a specific saved variable, keep the loading and
launching in application code:

```python
import os
import subprocess

from dotenv import dotenv_values

name = "GEMINI_API_KEY"
values = dotenv_values("/absolute/project/.env", interpolate=False)
value = values.get(name)
if not value:
    raise SystemExit("Required credential is not configured")
child_env = os.environ.copy()
child_env[name] = value
subprocess.run(["aua", "doctor"], env=child_env, check=True)
```

Neither example emits the key. Other languages should use a dotenv parser that preserves
literal values and disables variable expansion.

## File behavior

AUA preserves unrelated entries and comments, and saves the selected variable using an
atomic replacement. It protects the destination against unsafe filesystem targets and
conflicting changes rather than overwriting a file that changed while the dialog was open.
On POSIX systems the saved file has owner-only permissions. Keep it ignored by version control;
the dialog does not change your `.gitignore` or upload the file anywhere.

The dialog keeps values out of AUA's agent-facing results and command arguments. The `.env`
file remains readable by the local user and their programs; it is not an encrypted secret
store. Never ask an agent to read it back to confirm success—use the returned status.

An interrupted save can leave a sibling lock file, such as `.env.aua-lock`. If a later
request reports `credential_save_busy`, wait for any active save to finish. Only after
confirming no AUA credential save is running for that destination, remove that lock file
and retry. Do not remove the `.env` file or read its contents to recover.
