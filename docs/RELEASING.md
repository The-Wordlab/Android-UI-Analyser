# Releasing — tags, versions, and what changed

## Why this exists

Before the first tagged release the only way to get `aua` was to track `main`: `git pull`, re-run
`install.sh`, hope. There was nothing to pin to, no cheap way to ask *is there something newer than
what I have*, and no page anywhere that answered *what does that newer thing do*. Anyone who wanted
to gate a CI job or an agent harness on a known-good `aua` had to read commit subjects. Tags,
releases generated from a changelog, and `aua update --check` exist so that those three questions —
what do I have, is there more, what is in it — are answerable without opening this repository.

## The versioning contract

SemVer (`MAJOR.MINOR.PATCH`), read against **the contract an agent sees**, not against the size of
the diff. Reworking the engine internals is invisible; renaming one JSON field is not.

| bump | means |
|---|---|
| **MAJOR** | The agent-visible contract broke: the stable element `id` scheme changed, a CLI command or flag was removed or renamed, a field in an `analyze` or action `observation` payload was renamed, retyped or dropped, or an MCP tool signature changed. |
| **MINOR** | Additive only: new commands, new flags, new MCP tools, new providers, new config keys with safe defaults. Everything written against the previous version still runs. |
| **PATCH** | Fixes and performance. Nothing an agent reads or calls moved. |

The line is drawn at the agent surface on purpose. Everything downstream of this tool — a generated
skill file, an MCP client, a `jq` filter in someone's pipeline — is written against those four
surfaces and nothing else. If none of them move, it is a patch however large the change.

We are on `0.x`, so the contract is still allowed to move and a breaking change ships as a MINOR
bump while the leading zero holds. That is not permission to be quiet about it: **a breaking change
is always listed under `### Breaking` in `CHANGELOG.md`**, at whatever version it lands in, and the
release page repeats it. The number tells you how careful to be; the `### Breaking` section tells
you exactly what to fix.

## Where the version lives

Eight release surfaces, and they must agree:

| file | who reads it |
|---|---|
| `pyproject.toml` → `[project] version` | the built wheel and sdist |
| `src/android_ui_analyser/__init__.py` → `__version__` | `aua --version`, and the daemon's identity string |
| `.claude-plugin/plugin.json` → `version` | the installed Claude Code plugin |
| `.claude-plugin/marketplace.json` → `plugins[].version` | what `/plugin update` compares against |
| `.codex-plugin/plugin.json` → `version` | the installed Codex plugin |
| `.mcp.json` → pinned `@vX.Y.Z` source (`env.AUA_SPEC`) | the exact AUA release both plugins start through `uvx` |
| `server.json` → `version` and `packages[0].version` | the MCP Registry listing: the PyPI release an MCP client starts from it |
| `README.md` → pinned install examples | the exact release users copy into `uvx` and clone commands |

`tests/test_the_version_is_the_same_everywhere.py` fails if any of them drift apart, because a
version that is true in five places and stale in the sixth is worse than no version at all — it
reports success while handing someone the wrong thing. The marketplace listing and MCP source both
matter: a stale marketplace never offers the update, while a stale MCP tag advertises the new
plugin but silently runs the old server.

Use `scripts/bump-version.sh`, never a hand edit. It also refreshes `uv.lock`, which pins the root
package version — and CI runs `uv sync --frozen`, which does **not** validate the lock, so a stale
lock would otherwise disagree with `pyproject.toml` silently and forever.

## Cutting a release

1. **Write the changes under `## [Unreleased]` in `CHANGELOG.md` as you merge them.** This is the
   whole habit. A changelog cannot be reconstructed from commit subjects afterwards — the subject
   says what the author changed, not what a user can now do — and every attempt to do it at release
   time produces a list nobody reads. One line per user-visible change, in the release, or it did
   not happen.
2. `scripts/bump-version.sh --minor` (or `--major` / `--patch`, or an explicit `0.14.0`). It moves
   all eight release surfaces, refreshes `uv.lock`, and promotes `## [Unreleased]` to
   `## [0.14.0] - 2026-08-28`.
3. **Read the diff.** It should be exactly the eight release surfaces, `uv.lock`, and the changelog
   heading. Anything else means the script picked up an edit you did not intend to release.
4. `git commit -m "release: v0.14.0"`
5. `git tag -a v0.14.0 -m "v0.14.0" && git push origin main --follow-tags`
6. The `release` workflow fires on the `v*` tag. It re-checks that the tag matches all eight release
   files, runs ruff, mypy and the suite (the ordinary CI workflow does **not** run on tags, so the
   release workflow runs them itself — a tag that publishes untested code is the one failure mode
   worth spending a few minutes of CI on), builds the wheel and sdist, and publishes the GitHub
   Release with the notes for that version lifted out of `CHANGELOG.md`. The next job uploads the
   same wheel and sdist to PyPI, and the last one lists the version in the MCP Registry.

**Dry run first if you are unsure.** The workflow also accepts `workflow_dispatch`: run it from the
Actions tab against a ref and it does everything except create the tag and the release, so you can
watch the verify-build-notes path succeed before anything is public.

## PyPI

The `pypi` job in `release.yml` uploads two projects on every tag, after the GitHub Release
exists:

- <https://pypi.org/project/android-ui-analyser/> holds the code: the same wheel and sdist as the
  GitHub Release.
- <https://pypi.org/project/aua/> is the short name, so `uvx aua …` and `pip install aua` work. It
  holds no code and requires the same version of `android-ui-analyser`.
  `scripts/build-aua-package.py` writes it from `pyproject.toml` during the job, so its version,
  extras and links are never edited by hand and it is not one more release surface.

The job signs in with a PyPI API token stored as the `PYPI_API_TOKEN` secret of this repository's
`pypi` environment; the token is never in the repository, and only a job that names that
environment can use it.

- **To replace the token**, create an account-wide one on pypi.org, run
  `gh secret set PYPI_API_TOKEN --env pypi`, then delete the old token on PyPI. A PyPI token covers
  one project or the whole account, and this job uploads two. A token that was deleted or has the
  wrong scope shows up as a `403` in the `pypi` job.
- **If the `pypi` job fails, re-run that job** (`gh run rerun <run-id> --failed`). The GitHub
  Release already exists and is unaffected, and files that already reached PyPI are passed over.
- **A version can be uploaded once.** PyPI never accepts the same filename twice, even after a
  delete, so a bad release is fixed by a new patch version, and the bad one is *yanked* on PyPI.
- **Each release adds about 64 MB** (30 MB wheel, 33 MB sdist) against PyPI's 10 GB per-project
  quota, which is roughly 150 releases. Nearly all of it is the bundled FunctionGemma adapter.
- **The wheel is not MIT throughout.** That adapter is distributed under the Gemma Terms of Use,
  and its directory carries the terms, prohibited-use policy and notices. `LICENSE` is the plain
  MIT text for the code, `NOTICE` names the adapter's directory and terms, and both ship in the
  wheel and the sdist. Publishing it to PyPI was decided on 2026-11-17; the reasoning sits above
  the job in `release.yml`.

Releases before 0.48.0 were never uploaded and install from their Git tag only.

## MCP Registry

The `mcp-registry` job in `release.yml` publishes `server.json` to the official registry at
<https://registry.modelcontextprotocol.io> as `io.github.The-Wordlab/aua`. MCP clients and other
directories read that registry, so one entry reaches several of them. The entry tells a client to
start the server with `uvx aua mcp` at the released version.

- **It needs no secret.** The job signs in with its GitHub OIDC token, and the registry lets a
  repository owned by The-Wordlab publish names under `io.github.The-Wordlab/`.
- **The registry checks PyPI.** It opens the page of the `aua` version named in `server.json` and
  looks for the line `<!-- mcp-name: io.github.The-Wordlab/aua -->` in the README uploaded with
  it. That line lives in the README's MCP server section, and a test fails if it goes missing.
- **If the job fails, re-run it** (`gh run rerun <run-id> --failed`), for example when PyPI has not
  caught up yet. GitHub and PyPI already have the release and are unaffected.
- **A version can be listed once**, and its entry cannot be edited afterwards. To change the
  title or description, edit `server.json` and let the next release carry it.
- **Check a change to `server.json` before tagging** with the registry's own tool:
  `mcp-publisher validate` (from the registry's
  [releases](https://github.com/modelcontextprotocol/registry/releases)).

## Consuming a release

**Ask whether there is something newer.**

```bash
aua update --check          # human-readable: current version, latest release, how to upgrade
aua update --check --json   # for automation
aua update --install        # install the latest release over a `uv tool` install
aua update --install --version 0.14.0   # or a chosen release
```

`--install` only changes a `uv tool` install of a release, reinstalling it at the tag with the
extras and `--with` packages its receipt recorded. For a clone, `uvx`, a plugin or anything else
it prints the command to run and exits `10`.

Exit codes, so a CI step can branch without parsing: `0` already on the latest release, `10` a newer
release exists, `1` the check could not run (offline, rate-limited, no network). `10` is deliberately
distinct from `1` — a CI gate must be able to tell "you are behind" from "I could not find out", and
neither should ever be reported as the other.

**No `aua` installed?** The gate is a release API call and `jq`:

```bash
curl -fsSL https://api.github.com/repos/The-Wordlab/Android-UI-Analyser/releases/latest \
  | jq -r .tag_name
```

Run a pinned release without installing it:

```bash
uvx --from \
  'android-ui-analyser[apple,rapidocr,audio] @ git+https://github.com/The-Wordlab/Android-UI-Analyser.git@v0.14.0' \
  aua --version
```

**Pin a clone to a release, and upgrade deliberately.**

```bash
git fetch --tags
git checkout v0.14.0
./install.sh
```

`install.sh` installs editable from the clone, so the checked-out ref *is* what runs — the checkout
is the upgrade, and `./install.sh` afterwards refreshes dependencies and regenerates the user-level
skill files. To go back to tracking `main`: `git checkout main && git pull && ./install.sh`. Either
way, restart any warm daemon (`aua daemon stop && aua daemon start`): the daemon's identity string
contains the version, so it retires itself after a version change anyway.

**Claude Code plugin users** take the skill update through the plugin client:

```
/plugin update android-ui-analyser@the-wordlab
```

**Codex plugin users** refresh this repository marketplace, then reinstall its current snapshot:

```bash
codex plugin marketplace upgrade the-wordlab
codex plugin add android-ui-analyser@the-wordlab
```

## What we promise

- **Every release page answers "what is new" on its own.** Release notes are generated from
  `CHANGELOG.md`, never from commit subjects, so the page is the whole story — nobody downstream
  should ever have to read the commit log to find out what they are getting.
- **A pushed tag is immutable.** Tags are never moved, re-pointed, or deleted once they are on the
  remote. Someone has already pinned to it; moving it changes what their build does without
  changing what their build says.
- **A bad release gets a new patch version, not a fixed one.** `v0.14.1` supersedes `v0.14.0`.
