# Shared agent runs and responses

The optional agent wrapper keeps a CLI goal's configuration and caller context together and
gives CLI and MCP integrations one response shape. It helps callers distinguish a command
failure from an empty screen and use observations already returned by AUA.

Existing CLI commands and MCP responses keep their current behavior unless explicitly opted in.

## CLI: one run file per caller and goal

Initialize a new path, then execute goal commands through it:

```bash
aua --observe-fields id,text,clickable,enabled run init /tmp/example-aua-run.json
aua run exec /tmp/example-aua-run.json -- session start --goal 'Verify Settings opens'
aua run exec /tmp/example-aua-run.json -- tap-and-analyze 'el:<returned-id>'
aua run exec /tmp/example-aua-run.json -- session finish
```

Replace the placeholder with the actual handle from the returned observation. Follow the
normal session protocol, including evidence for completion and the returned cleanup call.
An incomplete finish remains incomplete under the wrapper.

`run init` only writes private host files. It does not connect, lease, observe or provision a
target. It freezes the effective configuration and observation defaults, creates an isolated
AUA cache, and retains the shared lease registry. `session start` is the first executable goal
command; it selects or provisions the target and supplies the goal session, target and owner.

Put run-wide `--config`, `--profile`, `--owner`, `--platform`, `--serial`, `--needs` and
observation defaults before `run init`. Omit `--serial` for ordinary automatic selection.
Put per-call options after `run exec PATH --`; they cannot replace the saved config, profile,
owner, platform, target or capability requirements (`--needs`), disable leasing, or request
a non-JSON output format.
Child commands execute from the directory where the run was initialized, including resolution
of relative paths in their arguments. Changing ambient AUA configuration overrides does not
redirect the saved run. Non-configuration values such as `AUA_TOKEN` remain available from
the current environment and are not saved in the run file.

The file belongs to its original caller process and one goal. Another agent process must
initialize its own file; sharing an owner label does not transfer ownership. A missing or
mismatched goal is an explicit error. After the goal ends, review, capture access and supported
target cleanup remain available; further goal work needs a new file. `aua agent` remains a
hidden alias for `aua guide`.

## MCP: opt in on the existing connection

Call `configure` with `{"agent_response": true}`, then use `session_start` and the existing
tools as usual. The setting persists on that MCP server. `{"agent_response": false}` restores
legacy responses.

The server's existing engine retains configuration, target and caller ownership; MCP does not
create a CLI run file. Context reports the active goal session. A capture buffer's separate
`session_id` remains command result data and does not replace that goal identity.

## Read the envelope

Check CLI exit status or MCP `isError`, then the envelope's `ok`. Some transport or SDK
validation failures occur before an envelope is available. A valid JSON response or visible
controls alone do not establish success.

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` for this envelope. |
| `ok` | Whether the command and its transport succeeded. |
| `error` | Typed failure details, including the original code and hint when supplied, or `null`. |
| `observation` | The screen actually returned by this call, or `null`; the same location for analyze, actions and recovery evidence. |
| `observation_contract` | Existing evidence identity and the shared freshness, readiness and selector-reuse assessment. |
| `result` | Remaining command facts, such as recommendations, goal progress, delivery details or inventory arrays. |
| `context` | Caller and goal scope available to this transport. |

An action can return `ok: false` with `error: null` when its native result reports an unmet
condition; inspect `result` for the outcome. A typed error can still carry a useful observation.
Partial success in an attached result does not override the envelope's failure. Non-JSON or
missing CLI output is an explicit error, including an empty failed `has` response.
For unstructured CLI failures, `error.diagnostic` retains up to 2,000 characters of stderr
alongside the native exit code, so a syntax error remains distinguishable from absent UI.

Use returned controls only when the observation contract supports reuse. `readiness: ready`
confirms arrival; `not_checked` makes no destination claim. An empty element list may still
carry useful visual evidence. A summary count without actual elements supplies no addressable
controls. Requested projections stay projected; normalization cannot recover omitted fields.

When visual inspection is needed, use the returned `observation_contract.image_path`,
`observation.meta.raw_image` or MCP image block. The wrapper does not read image files, capture
another screen, maintain a second UI cache, substitute an earlier observation, or retry a
command. AUA's normal capture, journal and artifact settings still apply.

The private run file and sibling configuration directory contain configuration and identity,
not saved process environments or wrapper-cached responses. Provider secrets remain referenced
by environment-variable names; the current process supplies their values when commands run.
