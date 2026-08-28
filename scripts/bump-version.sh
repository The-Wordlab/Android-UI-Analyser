#!/usr/bin/env bash
# Bump every published AUA version and promote the Unreleased changelog section.
#
# Usage:
#   scripts/bump-version.sh <new-version> [--force]
#   scripts/bump-version.sh --patch|--minor|--major [--force]
#
# Options:
#   --force   Allow a dirty tree or a branch other than main.
#   --help    Show this help.
#
# The script edits files only. It never commits, tags, or pushes.

set -euo pipefail

usage() {
  awk '
    NR == 1 { next }
    /^#/ { sub(/^# ?/, ""); print; next }
    { exit }
  ' "$0"
}

die() {
  printf 'bump-version: %s\n' "$*" >&2
  exit 1
}

force=0
target=""
for arg in "$@"; do
  case "$arg" in
    --help|-h)
      usage
      exit 0
      ;;
    --force)
      force=1
      ;;
    --patch|--minor|--major)
      [[ -z "$target" ]] || die "pass exactly one version or bump selector"
      target="$arg"
      ;;
    *)
      [[ "$arg" != -* ]] || die "unknown option: $arg"
      [[ -z "$target" ]] || die "pass exactly one version or bump selector"
      target="$arg"
      ;;
  esac
done

[[ -n "$target" ]] || { usage >&2; exit 2; }
command -v git >/dev/null 2>&1 || die "git is required"
command -v uv >/dev/null 2>&1 || die "uv is required to refresh uv.lock"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$repo_dir is not a git checkout"

if [[ "$force" -eq 0 ]]; then
  branch="$(git branch --show-current)"
  [[ "$branch" == "main" ]] || die "release from main, not ${branch:-detached HEAD}; pass --force to override"
  [[ -z "$(git status --porcelain)" ]] || die "working tree is dirty; commit or stash it, or pass --force"
fi

current="$(python3 - <<'PY'
import pathlib
import re

text = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
if match is None:
    raise SystemExit("pyproject.toml has no project version")
print(match.group(1))
PY
)"

case "$target" in
  --patch|--minor|--major)
    core="${current%%[-+]*}"
    IFS=. read -r major minor patch <<< "$core"
    case "$target" in
      --patch) target="$major.$minor.$((patch + 1))" ;;
      --minor) target="$major.$((minor + 1)).0" ;;
      --major) target="$((major + 1)).0.0" ;;
    esac
    ;;
esac

semver_re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-([0-9A-Za-z-]+\.)*[0-9A-Za-z-]+)?(\+([0-9A-Za-z-]+\.)*[0-9A-Za-z-]+)?$'
[[ "$target" =~ $semver_re ]] || die "$target is not valid SemVer (expected X.Y.Z, optionally with -prerelease or +build)"

python3 - "$current" "$target" <<'PY' || die "$target must be strictly newer than $current"
import re
import sys

pattern = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def key(value):
    match = pattern.fullmatch(value)
    if match is None:
        raise SystemExit(1)
    numbers = tuple(int(part) for part in match.group(1, 2, 3))
    pre = match.group(4)
    if pre is None:
        return numbers, 1, ()
    identifiers = tuple((0, int(p)) if p.isdigit() else (1, p) for p in pre.split("."))
    return numbers, 0, identifiers


raise SystemExit(0 if key(sys.argv[2]) > key(sys.argv[1]) else 1)
PY

release_date="$(date +%F)"
python3 - "$current" "$target" "$release_date" <<'PY'
import json
import pathlib
import re
import sys

current, target, release_date = sys.argv[1:]
root = pathlib.Path(".")
paths = {
    "pyproject.toml": root / "pyproject.toml",
    "src/android_ui_analyser/__init__.py": root / "src/android_ui_analyser/__init__.py",
    ".claude-plugin/plugin.json": root / ".claude-plugin/plugin.json",
    ".claude-plugin/marketplace.json": root / ".claude-plugin/marketplace.json",
    ".codex-plugin/plugin.json": root / ".codex-plugin/plugin.json",
    ".mcp.json": root / ".mcp.json",
}

pyproject = paths["pyproject.toml"].read_text(encoding="utf-8")
init = paths["src/android_ui_analyser/__init__.py"].read_text(encoding="utf-8")
plugin = json.loads(paths[".claude-plugin/plugin.json"].read_text(encoding="utf-8"))
marketplace = json.loads(paths[".claude-plugin/marketplace.json"].read_text(encoding="utf-8"))
codex_plugin = json.loads(paths[".codex-plugin/plugin.json"].read_text(encoding="utf-8"))
mcp = json.loads(paths[".mcp.json"].read_text(encoding="utf-8"))
mcp_args = mcp["mcpServers"]["android-ui-analyser"]["args"]
mcp_source = mcp_args[2]
mcp_tag = re.search(r"@v([^@\s]+)$", mcp_source)
pyproject_match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
init_match = re.search(r'^__version__ = "([^"]+)"$', init, re.MULTILINE)
found = {
    "pyproject.toml": pyproject_match.group(1) if pyproject_match else None,
    "src/android_ui_analyser/__init__.py": init_match.group(1) if init_match else None,
    ".claude-plugin/plugin.json": plugin.get("version"),
    ".claude-plugin/marketplace.json": marketplace.get("plugins", [{}])[0].get("version"),
    ".codex-plugin/plugin.json": codex_plugin.get("version"),
    ".mcp.json": mcp_tag.group(1) if mcp_tag else None,
}
drift = {path: version for path, version in found.items() if version != current}
if drift:
    details = ", ".join(f"{path}={version!r}" for path, version in drift.items())
    raise SystemExit(f"version files disagree with pyproject {current}: {details}")

changelog_path = root / "CHANGELOG.md"
changelog = changelog_path.read_text(encoding="utf-8")
unreleased = re.search(
    r"^## \[Unreleased\]\s*\n(?P<body>.*?)(?=^## \[|\Z)",
    changelog,
    re.MULTILINE | re.DOTALL,
)
if unreleased is None:
    raise SystemExit("CHANGELOG.md has no ## [Unreleased] section")
notes = unreleased.group("body").strip()
if not notes or notes.casefold() == "nothing yet.":
    raise SystemExit("CHANGELOG.md Unreleased section is empty; describe the release first")
if re.search(rf"^## \[{re.escape(target)}\] - ", changelog, re.MULTILINE):
    raise SystemExit(f"CHANGELOG.md already has a {target} release section")

pyproject = pyproject[: pyproject_match.start(1)] + target + pyproject[pyproject_match.end(1) :]
init = init[: init_match.start(1)] + target + init[init_match.end(1) :]
plugin["version"] = target
marketplace["plugins"][0]["version"] = target
codex_plugin["version"] = target
mcp_args[2] = re.sub(r"@v[^@\s]+$", f"@v{target}", mcp_source)

new_release = f"## [Unreleased]\n\n## [{target}] - {release_date}\n\n{notes}\n\n"
changelog = changelog[: unreleased.start()] + new_release + changelog[unreleased.end() :]
changelog = re.sub(r"^\[(?:Unreleased|" + re.escape(target) + r")\]:.*\n?", "", changelog, flags=re.MULTILINE)
changelog = changelog.rstrip() + "\n\n"
changelog += (
    f"[Unreleased]: https://github.com/The-Wordlab/Android-UI-Analyser/compare/v{target}...HEAD\n"
    f"[{target}]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v{target}\n"
)

paths["pyproject.toml"].write_text(pyproject, encoding="utf-8")
paths["src/android_ui_analyser/__init__.py"].write_text(init, encoding="utf-8")
paths[".claude-plugin/plugin.json"].write_text(json.dumps(plugin, indent=2) + "\n", encoding="utf-8")
paths[".claude-plugin/marketplace.json"].write_text(
    json.dumps(marketplace, indent=2) + "\n", encoding="utf-8"
)
paths[".codex-plugin/plugin.json"].write_text(
    json.dumps(codex_plugin, indent=2) + "\n", encoding="utf-8"
)
paths[".mcp.json"].write_text(json.dumps(mcp, indent=2) + "\n", encoding="utf-8")
changelog_path.write_text(changelog, encoding="utf-8")
PY

uv lock

printf 'Bumped AUA %s -> %s and released the Unreleased changelog as %s.\n\n' \
  "$current" "$target" "$release_date"
printf 'Review the diff, then run:\n'
printf '  git add -A && git commit -m "release: v%s"\n' "$target"
printf '  git tag -a v%s -m "v%s" && git push origin main --follow-tags\n' "$target" "$target"
