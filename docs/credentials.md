# Enter API keys privately and continue automatically

Use AUA's credential dialog when an agent needs an API key or another secret stored in a
project's `.env` file. The agent supplies the variable name and destination; the user types
the value directly into a masked field and clicks **Save**. The response never contains the
value. Credential entry itself needs no Android device, AUA session, model, or remote API call.

## Save and run in one command

Wrap the program that needs the key:

```bash
aua config exec --env-file /absolute/project/.env --require OPEN_ROUTER_API_KEY \
  -- python /absolute/project/qa_runner.py --scenario smoke
```

Replace the example runner and arguments with your existing program. AUA checks the required
variable, opens the private dialog only if it is missing, then launches the program **once**
after **Save**. The program receives the value through its environment; neither the agent
nor the user needs to copy it into another command. On later runs, a configured value skips
the dialog. No request to a model or test execution occurs until the child program starts.

The lookup order is:

1. A nonempty value already in the wrapper's process environment.
2. The named variable in the exact file passed to `--env-file`, parsed without interpolation.
3. The private Save/Cancel dialog, if the value is still missing.

Only variables named by `--require` are imported from the file. Other existing process
environment variables are inherited normally. Repeat `--require` when the program needs
several credentials; all must be available before launch:

```bash
aua config exec --env-file /absolute/project/.env \
  --require EXAMPLE_API_KEY --require EXAMPLE_SERVICE_TOKEN \
  -- python /absolute/project/runner.py
```

Always put `--` before the child command. Its arguments, including further `--` separators,
are preserved. This also wraps an existing AUA agent run:

```bash
aua config exec --env-file /absolute/project/.env --require GEMINI_API_KEY \
  -- aua run exec /absolute/project/aua-run.json -- ask "Which control opens settings?"
```

The run file must already exist, and its provider configuration must use that variable.
The wrapper supplies credentials; it does not configure the provider or install an external
harness. No generic command-execution tool is added to MCP.

**Cancel** returns exit `130` and starts no child. Setup failure also starts no child. If
several dialogs are needed, values saved before a later cancellation remain saved, but the
program still does not launch. A child failure is returned without retrying the command.
`--timeout 300` bounds each dialog, not the child program's execution time.

For CI or another unattended environment, use `--no-prompt`; missing credentials then fail
without a dialog or launch. There is no headless fallback that requests the key in chat:

```bash
aua config exec --env-file /absolute/project/.env --require EXAMPLE_API_KEY --no-prompt \
  -- python /absolute/project/runner.py
```

Normal child text stdout/stderr and exit status pass through, without an extra success JSON
object. Wrapper errors are safe JSON on stderr. Literal required credential values are
redacted from child stdout/stderr, including values split across output chunks. This is a
last defense, not a secret-printing interface: the program must not dump secrets, encode or
transform them into output, or write them to logs/files. Those transformed values and files
are outside the wrapper's redaction boundary.

A wrapper error can override the child's exit status. For example, if the direct child has
exited but a background descendant keeps an output pipe open beyond the one-second drain
window, the wrapper returns `credential_output_incomplete` with `started: true`. It closes
its read handles without killing background descendants or replaying the command. Output
may be incomplete even if the child finished successfully. Treat any `started: true` error
as an already-started operation: inspect the result through the program's normal status or
evidence interface, and do not rerun side effects merely to recover output.

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

`credential_request` only saves the value. An agent that needs to launch an external program
and continue after Save should use the CLI wrapper above through its terminal tool, or call
the Python helper below from its own application code. The MCP server does not acquire the
saved key in its process environment.

The dialog opens on the **MCP server's computer**. A remote server cannot open a dialog on
the client's desktop. macOS uses a native hidden-input dialog; other supported desktop
environments use Tk when available. A headless host or missing graphical dialog support
returns an error. Run the CLI on the intended desktop host and destination instead; do not
fall back to requesting the value in the conversation or terminal arguments.

## Use the continuation helper from Python

The same implementation is available to Python applications:

```python
from android_ui_analyser.credential_exec import run_with_credentials

result, exit_code = run_with_credentials(
    ["python", "/absolute/project/runner.py", "--scenario", "smoke"],
    required=["OPEN_ROUTER_API_KEY"],
    env_file="/absolute/project/.env",
    timeout_s=300,
    prompt=True,
)
raise SystemExit(exit_code)
```

The helper streams redacted child output and returns only status/error metadata and the
exit code. `result["started"]` distinguishes a launched child from credential/setup failure.
It never enriches the caller's `os.environ`; required values are passed only to the child.
Do not pass secret values in the command list.

## Load a file manually when embedding an SDK

The `.env` file is configuration **data**. Saving it does not change an agent's environment,
a parent shell, the MCP server, or an already running AUA daemon. The continuation wrapper
loads it for its child. A program embedding an SDK directly must instead load the exact file
before using the key. Do not print, inspect through an agent tool, or shell
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

A sibling lock file, such as `.env.aua-lock`, stays in place after saving. This is normal:
the operating system holds the active lock and releases it automatically when the save
finishes or its process dies. Existing leftover lock files from earlier versions are reused;
do not delete the lock file to recover. If a request reports `credential_save_busy`, another
save is active—wait for it to finish. Finish any saves running an older AUA version before
upgrading, because old writers do not participate in the new OS-held lock.
