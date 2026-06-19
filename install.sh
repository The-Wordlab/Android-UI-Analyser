#!/usr/bin/env bash
#
# Bootstrap android-ui-analyser for Claude Code.
#
# Installs the `aua` CLI GLOBALLY (so it's on PATH in every project) and installs the
# Claude Code skill at USER level (~/.claude/skills) so Claude Code auto-discovers it in
# every project — not just this repo. Idempotent: safe to re-run. macOS + Linux.
#
# Usage:  ./install.sh
#
# We intentionally do NOT use `set -e`: global installs are attempted with explicit
# fallback to a project-local venv, so a failed `uv`/`pipx` step must not abort the script.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Core hierarchy analysis needs NO extras (uiautomator2 is a base dependency). OCR powers the
# vision fallback on Compose/Flutter/WebView/canvas screens the accessibility tree can't see.
case "$(uname -s)" in
  Darwin) OCR_PKGS=(pyobjc-framework-Vision pyobjc-framework-Quartz rapidocr-onnxruntime onnxruntime); EXTRA="apple,rapidocr" ;;
  *)      OCR_PKGS=(rapidocr-onnxruntime onnxruntime);                                                 EXTRA="rapidocr" ;;
esac

export PATH="$HOME/.local/bin:$PATH"   # where uv/pipx drop console scripts
AUA=""

install_global() {
  if command -v uv >/dev/null 2>&1; then
    echo "==> Installing the 'aua' CLI globally with uv tool..."
    local with=(); local p; for p in "${OCR_PKGS[@]}"; do with+=(--with "$p"); done
    if uv tool install --force "${with[@]}" "$REPO_DIR"; then
      uv tool update-shell >/dev/null 2>&1 || true
      AUA="aua"; return 0
    fi
    echo "    uv tool install failed; trying the next option..."
  fi
  if command -v pipx >/dev/null 2>&1; then
    echo "==> Installing the 'aua' CLI globally with pipx..."
    if pipx install --force "$REPO_DIR"; then
      pipx inject android-ui-analyser "${OCR_PKGS[@]}" >/dev/null 2>&1 \
        || echo "    (OCR engine not added — core CLI still works; add it later for the vision fallback)"
      pipx ensurepath >/dev/null 2>&1 || true
      AUA="aua"; return 0
    fi
    echo "    pipx install failed; falling back to a local venv..."
  fi
  return 1
}

install_venv() {
  echo "==> pipx/uv unavailable — installing into a project-local venv (.venv)."
  echo "    NOTE: this makes 'aua' available only via $REPO_DIR/.venv/bin/aua, not globally."
  echo "    For a GLOBAL 'aua', install pipx (python3 -m pip install --user pipx) and re-run."
  python3 -m venv .venv || { echo "ERROR: could not create venv"; exit 1; }
  ./.venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true
  ./.venv/bin/python -m pip install -q --upgrade pip
  ./.venv/bin/python -m pip install -q -e ".[${EXTRA}]" || { echo "ERROR: pip install failed"; exit 1; }
  AUA="$REPO_DIR/.venv/bin/aua"
}

install_global || install_venv

# Resolve a runnable aua (prefer the global one; fall back to the venv path).
if ! command -v "$AUA" >/dev/null 2>&1 && [ ! -x "$AUA" ]; then
  AUA="$REPO_DIR/.venv/bin/aua"
fi

echo "==> Installing the Claude Code skill at user level (~/.claude/skills)..."
"$AUA" guide --emit-skill "$HOME/.claude/skills/android-ui-analyser/SKILL.md"

echo
echo "==> Verifying environment (aua doctor):"
"$AUA" doctor || true

cat <<EOF

────────────────────────────────────────────────────────────────────────────
✓ Setup complete.
  • 'aua' CLI installed ($AUA)
  • Skill installed at ~/.claude/skills/android-ui-analyser/ — active in EVERY project

Next steps:
  1. Connect an Android device or emulator   (README → "Connect a device or emulator")
  2. Run 'aua doctor' until adb + devices show OK
  3. In any project, just ask Claude Code to test your Android app — the skill
     activates automatically. Operating manual: 'aua guide'.
────────────────────────────────────────────────────────────────────────────
EOF
