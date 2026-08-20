#!/usr/bin/env bash
#
# Bootstrap android-ui-analyser for Claude Code and Codex.
#
# Installs the `aua` CLI GLOBALLY (so it's on PATH in every project) and installs the
# equivalent user-level skills so agents discover it in every project. Idempotent.
#
# Usage:  ./install.sh [--with-policy[=hybrid]] [--print-plan]
#
#   --with-policy         also install the optional LOCAL POLICY runtime (see below)
#   --with-policy=hybrid  ... including the larger MLX-VLM reviewer
#   --print-plan          print what would be installed and exit, touching nothing
#   AUA_INSTALL_POLICY=1  same as --with-policy (=hybrid also accepted)
#
# We intentionally do NOT use `set -e`: global installs are attempted with explicit
# fallback to a project-local venv, so a failed `uv`/`pipx` step must not abort the script.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

usage() {
  # The header comment block *is* the help text, so the two can never disagree. Stops at the
  # first line that is not a comment, however long the block grows.
  awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

# ---------------------------------------------------------------- options
# An unknown option used to be ignored outright, so a typo'd opt-in installed a policy-less CLI
# and still printed "Setup complete".
WITH_POLICY="${AUA_INSTALL_POLICY:-}"
PRINT_PLAN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-policy)      WITH_POLICY=1 ;;
    --with-policy=*)    WITH_POLICY="${1#*=}" ;;
    --print-plan)       PRINT_PLAN=1 ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "install.sh: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# Core hierarchy analysis needs NO extras (uiautomator2 is a base dependency). OCR powers the
# vision fallback on Compose/Flutter/WebView/canvas screens the accessibility tree can't see.
case "$(uname -s)" in
  Darwin) OCR_PKGS=(pyobjc-framework-Vision pyobjc-framework-Quartz rapidocr-onnxruntime onnxruntime); EXTRA="apple,rapidocr,audio,proxy" ;;
  *)      OCR_PKGS=(rapidocr-onnxruntime onnxruntime);                                                 EXTRA="rapidocr,audio,proxy" ;;
esac
AUDIO_PKGS=(grpcio)
# `aua proxy` and `aua mock` import mitmproxy and shell out to mitmdump. They are listed by
# `aua --help` unconditionally, so omitting the extra here does not hide them — it only moves the
# failure to run time, in the global install that agents reach through PATH.
PROXY_PKGS=(mitmproxy)
FEATURE_PKGS=("${OCR_PKGS[@]}" "${AUDIO_PKGS[@]}" "${PROXY_PKGS[@]}")

# ---------------------------------------------------------------- local policy (opt-in)
# The optional local policy/autopilot needs an MLX runtime, and it is NOT in the default
# install on purpose:
#   * it is heavy — mlx alone is ~183 MB installed, plus transformers/tokenizers for the small
#     selector; the `hybrid-policy` reviewer additionally pulls opencv, fastapi/uvicorn and
#     mlx-audio. This project cares about install weight and cold-start latency.
#   * it is Apple-silicon-only, so on every other platform the extras resolve to nothing.
#   * installing it is never sufficient anyway: the lane also needs a base model the user must
#     download after accepting the Gemma terms (docs/LOCAL_POLICY_SETUP.md), so a default
#     install would add hundreds of megabytes and still not produce a working policy.
# But it must be *possible*, in the global install agents reach through PATH — before this flag
# there was no installer path that produced a runnable policy at all, and autopilot died with
# "optional dependency missing" on every step.
# `aua session autopilot` is its own opt-in at run time; this is the matching opt-in at install
# time. `aua doctor` reports a configured-but-unrunnable policy and prints this command.
POLICY_EXTRAS=()
case "$(printf '%s' "${WITH_POLICY:-}" | tr '[:upper:]' '[:lower:]')" in
  ""|0|no|false|off)   : ;;
  hybrid|all|full)     POLICY_EXTRAS=(functiongemma hybrid-policy) ;;
  1|yes|true|on)       POLICY_EXTRAS=(functiongemma) ;;
  *) echo "install.sh: --with-policy expects nothing, '1' or 'hybrid' (got '${WITH_POLICY}')" >&2
     exit 2 ;;
esac

POLICY_RUNTIME="unsupported on this platform (the MLX policy runtime is Apple-silicon-only)"
if [ "$(uname -s)" = "Darwin" ]; then
  case "$(uname -m)" in arm64|aarch64) POLICY_RUNTIME="apple silicon" ;; esac
fi

# The extras ride in the INSTALL TARGET, not in `--with`: `uv tool` records the target's extras
# in its receipt, so the next `uv tool upgrade` keeps them. Extras added to a tool environment
# any other way are silently dropped on upgrade — the trap the autopilot error message warns of.
TARGET="$REPO_DIR"
if [ "${#POLICY_EXTRAS[@]}" -gt 0 ]; then
  POLICY_EXTRA_LIST="$(IFS=,; printf '%s' "${POLICY_EXTRAS[*]}")"
  TARGET="$REPO_DIR[$POLICY_EXTRA_LIST]"
  EXTRA="$EXTRA,$POLICY_EXTRA_LIST"
else
  POLICY_EXTRA_LIST=""
fi

if [ "$PRINT_PLAN" = 1 ]; then
  echo "target: $TARGET"
  echo "extras: $EXTRA"
  echo "with: ${FEATURE_PKGS[*]}"
  echo "policy-extras: ${POLICY_EXTRA_LIST:-(none)}"
  echo "policy-runtime: $POLICY_RUNTIME"
  exit 0
fi

if [ "${#POLICY_EXTRAS[@]}" -gt 0 ]; then
  echo "==> Local policy runtime requested: ${POLICY_EXTRA_LIST}"
  case "$POLICY_RUNTIME" in
    unsupported*) echo "    NOTE: $POLICY_RUNTIME — the extras will resolve to nothing here." ;;
    *) echo "    Heavy ML dependencies; the lane also needs a base model (docs/LOCAL_POLICY_SETUP.md)." ;;
  esac
fi

export PATH="$HOME/.local/bin:$PATH"   # where uv/pipx drop console scripts
AUA=""

install_global() {
  if command -v uv >/dev/null 2>&1; then
    echo "==> Installing the 'aua' CLI globally with uv tool..."
    local with=(); local p; for p in "${FEATURE_PKGS[@]}"; do with+=(--with "$p"); done
    # --reinstall: --force alone reuses uv's cached wheel when the version string has not
    # changed, so editing the source and re-running installed the OLD code silently.
    # --editable: the console script then imports straight from this clone, so `git pull`
    # updates the CLI and the daemon can never load different bytes than the caller.
    if uv tool install --force --reinstall --editable "${with[@]}" "$TARGET"; then
      uv tool update-shell >/dev/null 2>&1 || true
      AUA="aua"; return 0
    fi
    echo "    uv tool install failed; trying the next option..."
  fi
  if command -v pipx >/dev/null 2>&1; then
    echo "==> Installing the 'aua' CLI globally with pipx..."
    if pipx install --force --editable "$TARGET"; then
      pipx inject android-ui-analyser "${FEATURE_PKGS[@]}" >/dev/null 2>&1 \
        || echo "    (optional OCR/audio dependencies not added — core CLI still works)"
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

# Codex discovers personal skills under $CODEX_HOME/skills (default ~/.codex/skills). The skill
# body is byte-identical to Claude's; agents/openai.yaml is UI metadata, generated by guide.py.
AUA_CODEX_ROOT="${CODEX_HOME:-$HOME/.codex}/skills/android-ui-analyser"
echo "==> Installing the Codex skill at user level ($AUA_CODEX_ROOT)..."
"$AUA" guide --emit-skill "$AUA_CODEX_ROOT/SKILL.md"
mkdir -p "$AUA_CODEX_ROOT/agents"
"$AUA" guide --emit-codex-metadata "$AUA_CODEX_ROOT/agents/openai.yaml"

# Optional AOT thin client — skips Python startup when talking to a warm daemon.
if command -v cc >/dev/null 2>&1 || command -v clang >/dev/null 2>&1; then
  echo "==> Building optional aua-fast (C daemon client)..."
  if make -C "$REPO_DIR/native/aua-fast" install; then
    echo "    installed to ~/.local/bin/aua-fast (use after: aua daemon start)"
  else
    echo "    (aua-fast build skipped — 'aua' still works; see native/aua-fast/README.md)"
  fi
else
  echo "==> Skipping aua-fast (no C compiler). Install later: make -C native/aua-fast install"
fi

echo
echo "==> Verifying environment (aua doctor):"
"$AUA" doctor || true

cat <<EOF

────────────────────────────────────────────────────────────────────────────
✓ Setup complete.
  • 'aua' CLI installed ($AUA)
  • Skill installed at ~/.claude/skills/android-ui-analyser/ — active in EVERY project
  • Skill installed at $AUA_CODEX_ROOT — available to Codex in every project
  • Optional: 'aua-fast' (if built) for lower latency against a warm daemon

Next steps:
  1. Connect an Android device or emulator   (README → "Connect a device or emulator")
  2. Run 'aua doctor' until adb + devices show OK
  3. In any project, ask Claude Code or Codex to test your Android app. Start with
     'aua session start --goal "<what to verify>"'. Operating manual: 'aua guide --brief'.
────────────────────────────────────────────────────────────────────────────
EOF
