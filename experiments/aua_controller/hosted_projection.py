"""Opt-in privacy projection for fictional-fixture AUA tool observations.

Apply after the controller's authored-answer filter, never to assistant messages
or native tool-call envelopes. This is not a general personal-data anonymizer.
Raw observations and evidence remain unchanged in private local artifacts.
"""

from __future__ import annotations

import json
import re
from typing import Any

PROFILE = "hosted-v1"
_REMOVED = {
    "serial", "deviceserial", "deviceid", "sessionid", "capturesessionid",
    "target", "targetid", "owner", "caller", "ownerid", "callerid",
    "virtualtargetdefinitionid", "virtualtargetinstancetoken", "virtualtargetstarted",
    "artifactsdir", "artifactdir", "artifactpath", "artifactpaths", "artifacts",
    "imagepath", "screenshotpath", "screenshot", "rawimage", "annotatedimage",
    "captureevidence", "capturehint", "evidenceid", "appinstall", "emulatorstarted",
    "log", "logs", "applogs", "crashevidence", "logpath", "logfile", "traceback",
    "authorization", "apikey", "accesstoken", "refreshtoken", "devicelocale",
}
_IDENTITIES = {
    "serial", "deviceserial", "deviceid", "sessionid", "capturesessionid",
    "target", "targetid", "owner", "caller", "ownerid", "callerid",
    "virtualtargetdefinitionid", "virtualtargetinstancetoken",
}
# Android resource IDs and package/activity names contain slashes but are not
# host paths: the boundary excludes their preceding word/colon characters.
_PATH = re.compile(
    r"file://[^\s\"'<>]+|(?<![\w:/])/(?!/)[^\s\"'<>]+"
    r"|(?<!\w)[A-Za-z]:\\[^\s\"'<>]+"
    r"|(?<![\w/])(?:runs|artifacts|\.worktree)/[^\s\"'<>]+"
)
_LABELLED_ID = re.compile(
    r"(?i)\b(?:device[_ -]?serial|serial|(?:capture[_ -]?)?session[_ -]?id"
    r"|target[_ -]?id|owner[_ -]?id|caller[_ -]?id)\b[\"']?\s*[:=]\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;\}\]]+)"
)
# These are AUA's published action handles and selector semantics, not metadata.
_HANDLES = {"id", "parent", "resourceid", "rid", "stablekey", "selector", "selectors"}
# URL routes the app asked its own backend for, observed at the proxy. `_PATH` cannot tell one
# from an operator's filesystem path, and scrubbing them sent the hosted model
# `GET [private-path] -> 200` -- the evidence with the evidence removed. Identity scrubbing
# still applies to them; only the host-path rule is skipped.
_URL_PATHS = {"networkcalls"}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _embedded(value: str) -> Any:
    if value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except ValueError:
            pass
    return None


def hosted_model_view(value: Any) -> Any:
    """Copy an AUA observation, dropping metadata and redacting diagnostic echoes.

    IDs needed by AUA actions remain byte-for-byte intact. Session identifiers
    are unnecessary because the harness injects them. No state or alias mapping
    persists between observations, and the input is never mutated.
    """
    identities: set[str] = set()

    def collect(item: Any, private_container: bool = False) -> None:
        if isinstance(item, dict):
            for name, child in item.items():
                normalized = _key(name)
                private = normalized in _IDENTITIES or (private_container and normalized == "id")
                if private and isinstance(child, str) and child:
                    identities.add(child)
                collect(child, private)
        elif isinstance(item, list):
            for child in item:
                collect(child, private_container)
        elif isinstance(item, str):
            nested = _embedded(item)
            if nested is not None:
                collect(nested, private_container)

    collect(value)
    ordered = sorted(identities, key=len, reverse=True)

    def project(item: Any, handle: bool = False, route: bool = False) -> Any:
        if isinstance(item, dict):
            return {
                name: project(child, _key(name) in _HANDLES, _key(name) in _URL_PATHS)
                for name, child in item.items() if _key(name) not in _REMOVED
            }
        if isinstance(item, list):
            return [project(child, handle, route) for child in item]
        if not isinstance(item, str) or handle:
            return item
        nested = _embedded(item)
        if nested is not None:
            return json.dumps(project(nested), ensure_ascii=False)
        for identity in ordered:
            item = re.sub(r"(?<![\w])" + re.escape(identity) + r"(?![\w])", "[private-id]", item)
        item = _LABELLED_ID.sub(lambda match: match.group(0)[:match.start("value") - match.start()]
                                + "[private-id]", item)
        return item if route else _PATH.sub("[private-path]", item)

    return project(value)
