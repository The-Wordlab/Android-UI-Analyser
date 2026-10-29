# Compose previews before device deployment

`aua preview` uses the optional Google Android CLI to compile a preview target and render
Jetpack Compose `@Preview` functions on the host. No emulator or running Android Studio is
needed. It still compiles the target: this is useful before installing and launching an APK,
not a substitute for compilation or a runtime test.

Install the optional tool with `aua devtools setup`, or configure an existing installation as
described in [the Android CLI reference](android-cli-reference.md). Ordinary AUA device
automation keeps working when the optional CLI is absent.

## Discover, render, inspect

```bash
aua preview list :app --project /path/to/project
aua preview render :app --project /path/to/project \
  --fqn com.example.catalog.CatalogScreenKt.CatalogPreview \
  --output /path/to/preview-results
```

Use the exact `methodFqn` returned by `preview list`. Repeat `--fqn` for several previews in
one render. Previews with different themes, locales, device configurations or font scales
should be authored as separate `@Preview` configurations in the source. `--all` on list also
includes previews from dependent modules. `--daemon` keeps Google's renderer warm; Google
documents an idle shutdown after ten minutes.

Rendering exports the component tree by default. Its returned `imagePath` and
`componentTreePath` identify the PNG and JSON files. Visually inspect the PNG when appearance
matters. Use those returned paths; filenames depend on the selected preview configuration.
Without `--output`, AUA creates a unique directory under its preview cache. `--no-image` skips
the PNG and requires component-tree export; `--no-component-tree` keeps only the image.
Read the JSON without connecting a device:

```bash
aua preview inspect /path/to/preview-results/CatalogPreview_0.json
```

Each component includes its name, pixel bounds, derived dimensions in dp, available text,
role, test tag, source location, semantics, parameters and modifiers. The response reports
the total component count and whether the returned list was truncated; use `--max-nodes`
up to 1,000 when a larger listing is useful. Exported source locations can point into Compose
library implementation files as well as your own source.

`component:0/1/...` IDs identify nodes in that exact artifact. They are not device element IDs
and cannot be sent to `aua tap-and-analyze`. Every response identifies the scope as `rendered_preview`,
with `interactive: false` and `runtime_verified: false`.

## Deterministic artifact checks

Check explicit facts in the exported tree:

```bash
aua preview check /path/to/preview-results/CatalogPreview_0.json --checks '[
  {"selector":{"text":"Catalog"},"property":"exists","expected":true},
  {"selector":{"name":"Button"},"property":"count","expected":1},
  {"selector":{"test_tag":"continue"},"property":"within_canvas","expected":true},
  {"selector":{"test_tag":"continue"},"property":"min_height_dp","expected":48}
]'
```

Selectors match exact `id`, `name`, `text`, `role` or `test_tag` values; multiple fields are
combined. `exists` accepts a boolean, `count` a nonnegative number, and `within_canvas` a
boolean. `min_width_dp` and `min_height_dp` test recorded layout dimensions. `text` and `role`
can also be checked against an expected string. Every property except `exists` and `count`
requires exactly one matching component. A missing or ambiguous component fails the check.

The response includes each expected and actual value, matched artifact IDs, and the SHA-256
of the component-tree bytes. `verified: true` means those static assertions passed against
that exact file. It does not establish tap delivery, navigation, networking, full accessibility
semantics, or the size of an expanded runtime touch target. For those, install the build and
run an AUA session on a real device or emulator.

An unsatisfied check prints its evidence and exits with code 8. Invalid input or an unsupported
check property exits with code 2. A command exit alone does not establish runtime behavior.

## MCP equivalents

The same engine paths are exposed as `preview_list`, `preview_render`, `preview_inspect` and
`preview_check`. Use `project_dir`, `reference`, `fqns`, `path` and `checks` as appropriate.
For example, `preview_render(reference=":app", project_dir="/path/to/project",
fqns=["com.example.catalog.CatalogScreenKt.CatalogPreview"])` returns artifact paths and the
first rendered PNG as an inline image for vision-capable clients.

## Verified integration reference

The component-tree reader was checked against Google Android CLI **1.1.16717390**. A fresh
fictional `empty-activity` project produced a PNG and a component tree containing `name`,
`bounds`, `sourceLocation`, `semantics`, `modifiers` and nested `children`. Google's
[preview command reference](https://developer.android.com/tools/agents/android-cli/commands/preview)
and [render options](https://developer.android.com/tools/agents/android-cli/commands/preview_render)
describe the upstream command. A changed or malformed component-tree schema is rejected
instead of being treated as an empty passing screen.
