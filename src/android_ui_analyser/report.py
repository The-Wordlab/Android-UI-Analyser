"""Evidence pages for people: the agent writes the HTML, AUA adds the video player and publishes it.

How a page looks is the agent's call. AUA owns the parts every report needs and nobody should
rewrite: a player that seeks on hosts that ignore Range requests (Drophere answers one with the
whole file), and a publish that sets access before the first byte is uploaded, then proves the
link works for the people meant to see it and refuses everyone else.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .errors import AuaError, ExitCode, UsageError

_DATA = Path(__file__).parent / "data"
PLAYER = _DATA / "video-player.js"
GUIDE = _DATA / "report-guide.md"
DROPHERE_API = "https://drophere.cc/api/v1"
DEFAULT_TTL = "7d"

_PLAYER_BLOCK = re.compile(r"<script data-video-player>.*?</script>\n?", re.DOTALL)
_TTL = re.compile(r"(\d+)\s*([hd])")
_CONTENT_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".pdf": "application/pdf",
}


class PublishError(AuaError):
    exit_code = ExitCode.PROVIDER
    code = "publish_failed"


def guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def add_player(pages: list[str]) -> dict[str, Any]:
    """Inline the player and image viewer before ``</body>`` of each page with a video or image."""

    if not pages:
        raise UsageError("name at least one HTML page")
    script = "<script data-video-player>\n" + PLAYER.read_text(encoding="utf-8") + "</script>\n"
    results = []
    for name in pages:
        page = Path(name).expanduser()
        if not page.is_file():
            raise UsageError(f"{page}: no such page")
        html = _PLAYER_BLOCK.sub("", page.read_text(encoding="utf-8"))
        before, body_end, after = html.rpartition("</body>")
        if not body_end:
            raise UsageError(f"{page}: no </body> to put the player before")
        if "<video" not in html and "<img" not in html:
            results.append(
                {"page": str(page), "player": "skipped", "reason": "no <video> or <img> on the page"}
            )
            continue
        page.write_text(before + script + body_end + after, encoding="utf-8")
        results.append({"page": str(page), "player": "inlined"})
    return {"ok": True, "pages": results}


@dataclass(frozen=True)
class _File:
    path: str
    body: bytes

    def manifest(self) -> dict[str, Any]:
        suffix = Path(self.path).suffix.casefold()
        content_type = _CONTENT_TYPES.get(suffix) or mimetypes.guess_type(self.path)[0]
        return {
            "path": self.path,
            "size": len(self.body),
            "contentType": content_type or "application/octet-stream",
            "hash": "sha256:" + hashlib.sha256(self.body).hexdigest(),
        }


def _bundle(folder: Path) -> list[_File]:
    if not folder.is_dir():
        raise UsageError(f"{folder}: not a folder")
    files = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        # Dot files are local state (.drophere/state.json, .DS_Store), never part of the page.
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        files.append(_File(relative.as_posix(), path.read_bytes()))
    if not any(item.path == "index.html" for item in files):
        raise UsageError(f"{folder}: no index.html at the top of the folder")
    return files


def _ttl_seconds(ttl: str) -> int:
    match = _TTL.fullmatch(ttl.strip().casefold())
    if match is None or int(match.group(1)) == 0:
        raise UsageError(f"expiry {ttl!r} is not a number of hours or days", hint="e.g. 24h or 7d")
    hours = int(match.group(1)) * (24 if match.group(2) == "d" else 1)
    return hours * 3600


def _api_key() -> str | None:
    key = os.environ.get("DROPHERE_API_KEY", "").strip()
    if key:
        return key
    credentials = Path.home() / ".drophere" / "credentials"
    if credentials.is_file():
        match = re.search(r"^API_KEY=(\S+)", credentials.read_text(encoding="utf-8"), re.M)
        if match:
            return match.group(1)
    return None


class _Drophere:
    def __init__(self, http: httpx.Client, key: str | None) -> None:
        self.http = http
        self.auth = {"Authorization": f"Bearer {key}"} if key else {}
        self.slug: str | None = None

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.http.request(method, DROPHERE_API + path, json=body, headers=self.auth)
        if response.is_error:
            hint = None
            if self.slug:
                hint = f"Artifact {self.slug} may be left pending; publish again with --slug {self.slug}."
            raise PublishError(
                f"{method} {path} -> {response.status_code} {response.text[:300]}", hint=hint
            )
        return response.json() if response.content else {}

    def upload(self, upload: dict[str, Any], body: bytes) -> None:
        headers = {**self.auth, **(upload.get("headers") or {})}
        response = self.http.request(upload.get("method", "PUT"), upload["url"], content=body,
                                     headers=headers)
        if response.is_error:
            raise PublishError(
                f"upload of {upload['path']} -> {response.status_code}",
                hint=f"Publish again with --slug {self.slug} to retry the upload.",
            )


def publish(
    folder: str,
    *,
    ttl: str | None = None,
    allow_domains: list[str] | None = None,
    allow_emails: list[str] | None = None,
    slug: str | None = None,
) -> dict[str, Any]:
    """Publish a folder to Drophere: create, restrict, upload, finalize, then verify."""

    files = _bundle(Path(folder).expanduser())
    key = _api_key()
    domains = list(allow_domains or [])
    emails = list(allow_emails or [])
    restricted = bool(domains or emails)
    if key is None and (restricted or slug or ttl):
        raise UsageError(
            "an allow-list, an expiry or --slug needs a Drophere API key",
            hint="Set DROPHERE_API_KEY or put API_KEY=... in ~/.drophere/credentials. Without a "
            "key the page is public and expires after 24 hours.",
        )
    manifest: dict[str, Any] = {"files": [item.manifest() for item in files]}
    if key:
        manifest["ttlSeconds"] = _ttl_seconds(ttl or DEFAULT_TTL)

    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        api = _Drophere(client, key)
        if slug:
            api.slug = slug
            created = api.call("PUT", f"/artifact/{slug}", manifest)
        else:
            created = api.call("POST", "/artifact", manifest)
            api.slug = created["slug"]
        if restricted:
            access: dict[str, Any] = {"visibility": "restricted"}
            if domains:
                access["allowedDomains"] = domains
            if emails:
                access["allowedEmails"] = emails
            api.call("PATCH", f"/artifact/{api.slug}/access", access)
        by_path = {item.path: item for item in files}
        for upload in created.get("uploads") or []:
            api.upload(upload, by_path[upload["path"]].body)
        finish: dict[str, Any] = {"versionId": created["versionId"]}
        if created.get("claimToken"):
            finish["claimToken"] = created["claimToken"]
        done = api.call("POST", f"/artifact/{api.slug}/finalize", finish)
        site = done.get("siteUrl") or created.get("siteUrl") or f"https://{api.slug}.drophere.cc/"
        return _verify(api, client, site.rstrip("/") + "/", files, restricted)


def _verify(
    api: _Drophere, client: httpx.Client, site: str, files: list[_File], wanted_restricted: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {"url": site, "slug": api.slug}
    if api.auth:
        info = api.call("GET", f"/artifact/{api.slug}")
        access = api.call("GET", f"/artifact/{api.slug}/access")
        result["expires_at"] = (info.get("artifact") or info).get("expiresAt")
        result["visibility"] = access.get("visibility")
        result["allowed_domains"] = access.get("allowedDomains") or []
        result["allowed_emails"] = access.get("allowedEmails") or []
    else:
        result["expires_at"] = None
        result["visibility"] = "public"
        result["note"] = "No API key: anyone with the link can open it, and it expires after 24 hours."
    # An update without an allow-list keeps the access it had, so check what is live, not what
    # this call asked for.
    restricted = result["visibility"] == "restricted"
    anonymous = client.get(site, follow_redirects=False).status_code
    result["anonymous_status"] = anonymous
    headers = api.auth if restricted else {}
    failed = []
    for item in files:
        response = client.get(site + ("" if item.path == "index.html" else item.path), headers=headers)
        if response.status_code != 200:
            failed.append(f"{item.path} {response.status_code}")
    result["files_ok"] = f"{len(files) - len(failed)}/{len(files)}"
    result["failed"] = failed
    strangers_refused = anonymous >= 300
    public_as_asked = anonymous == 200 and not wanted_restricted
    access_ok = strangers_refused if restricted else public_as_asked
    result["ok"] = access_ok and not failed
    return result
