# Set up AUA as an MCP server

This file is for an AI agent that installs MCP servers for a user, such as Cline. It lists what
must be on the computer, the configuration to add, and how to check that the server works.

AUA (`aua`) lets an agent see and operate an Android device or emulator, an iOS simulator, or a
web page. The agent gets the elements on screen with stable IDs and acts on an ID, not on pixel
coordinates. The server runs on the user's computer over stdio. It needs no account, API key or
environment variable.

## 1. Check what the computer needs

| Need | Check | If it is missing |
|---|---|---|
| `uv`, which provides `uvx` | `uvx --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, or `brew install uv` on macOS. Then open a new terminal. |
| `adb` (Android only) | `adb version` | `brew install android-platform-tools` on macOS. Otherwise download [platform-tools](https://developer.android.com/tools/releases/platform-tools) and add the folder to `PATH`. |
| An Android device or emulator | `adb devices` lists one line ending in `device` | Ask the user to connect a phone with USB debugging on, or to start an emulator. |

`uvx` downloads AUA and a suitable Python (3.11 or newer) by itself. Do not clone the repository
and do not run `pip install`.

## 2. Add the server

Add this entry to the client's MCP settings. In Cline that file is `cline_mcp_settings.json`.

```json
{
  "mcpServers": {
    "android-ui": {
      "command": "uvx",
      "args": ["aua", "mcp"]
    }
  }
}
```

If the client reports that `uvx` was not found, replace `"uvx"` with its full path. Print the
path with `command -v uvx`; it is usually `~/.local/bin/uvx` or `/opt/homebrew/bin/uvx`.

## 3. Check that it works

```bash
uvx aua doctor
```

The `adb` and `devices` lines must say OK. The first run downloads the package and takes longer.

Then restart the client and call the `session_start` tool with a goal, for example
`session_start(goal="read the current screen")`. A working server returns the elements on the
device's screen, each with an `id`.

## Other platforms

The default platform is Android. To use another one, change `args`:

- **iOS simulator** (macOS with Xcode 26 or newer): `["aua", "--platform", "ios", "mcp"]`. Run
  `uvx aua --platform ios doctor --fix` once to install the AXe helper, and boot a simulator.
- **Web pages**: install a browser once with `uvx --from 'aua[web]' playwright install chromium`,
  then follow [docs/web.md](docs/web.md) for the configuration file and the server arguments.

## Updating

`uvx` keeps the release it downloaded first. To start the newest release every time, use
`"args": ["aua@latest", "mcp"]`.

More detail is in the [README](README.md): [prerequisites](README.md#prerequisites),
[connecting a device](README.md#connect-a-device-or-emulator) and the
[MCP server](README.md#mcp-server) section.
