"""Build, install, and control AUA's Android VPN companion over ADB."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import ssl
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from .errors import DeviceError, UsageError

PACKAGE = "dev.androiduianalyser.companion"
VERSION = "0.1.4"
TUN2PROXY_VERSION = "0.8.3"
TUN2PROXY_URL = (
    f"https://github.com/tun2proxy/tun2proxy/releases/download/v{TUN2PROXY_VERSION}/"
    "tun2proxy-android-libs.zip"
)
TUN2PROXY_SHA256 = "50706ce2b0799295b6672cf6b72ec388d02a3104ebd1195b3a9bca10d2bd80f5"


def _run(args: list[str], *, timeout: int = 180, cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise UsageError(
            f"companion command failed: {' '.join(args[:3])}: {stderr.strip() or exc}",
            hint="Install Android SDK platform-tools/build-tools and a JDK, then retry.",
        ) from exc
    return proc.stdout or ""


def _sdk_root() -> Path:
    candidates = [
        os.environ.get("ANDROID_HOME"),
        os.environ.get("ANDROID_SDK_ROOT"),
        str(Path.home() / "Library/Android/sdk"),
        str(Path.home() / "Android/Sdk"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "platforms").is_dir():
            return Path(candidate)
    raise UsageError(
        "Android SDK not found",
        hint="Set ANDROID_HOME or ANDROID_SDK_ROOT before starting the companion proxy.",
    )


def _latest_child(parent: Path) -> Path:
    children = [p for p in parent.iterdir() if p.is_dir()]
    if not children:
        raise UsageError(f"no Android SDK components installed under {parent}")

    def version(path: Path) -> tuple[int, ...]:
        return tuple(int(x) for x in re.findall(r"\d+", path.name))

    return max(children, key=version)


def _download_native_libraries(cache: Path) -> Path:
    archive = cache / f"tun2proxy-{TUN2PROXY_VERSION}-android.zip"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != TUN2PROXY_SHA256:
        cache.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(TUN2PROXY_URL, archive)
        except Exception as exc:
            raise UsageError(
                f"could not download tun2proxy {TUN2PROXY_VERSION}",
                hint=f"Download {TUN2PROXY_URL} and place it at {archive}.",
            ) from exc
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != TUN2PROXY_SHA256:
            archive.unlink(missing_ok=True)
            raise UsageError("downloaded tun2proxy archive failed its SHA-256 check")
    return archive


def build_apk(cache_dir: str | Path) -> Path:
    """Build the companion APK once using Android SDK command-line tools."""
    cache = Path(cache_dir).expanduser() / "companion"
    apk = cache / f"aua-companion-{VERSION}.apk"
    if apk.exists():
        return apk

    sdk = _sdk_root()
    platform = _latest_child(sdk / "platforms")
    tools = _latest_child(sdk / "build-tools")
    android_jar = platform / "android.jar"
    source = Path(__file__).with_name("companion_app")
    work = cache / "build"
    classes = work / "classes"
    dex = work / "dex"
    native = work / "native"
    for directory in (classes, dex, native):
        directory.mkdir(parents=True, exist_ok=True)

    archive = _download_native_libraries(cache)
    with zipfile.ZipFile(archive) as zf:
        for abi in ("arm64-v8a", "armeabi-v7a", "x86", "x86_64"):
            name = f"tun2proxy-android-libs/{abi}/libtun2proxy.so"
            target = native / abi / "libtun2proxy.so"
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    java_files = sorted(str(p) for p in (source / "java").rglob("*.java"))
    _run(
        [
            "javac",
            "-source",
            "8",
            "-target",
            "8",
            "-bootclasspath",
            str(android_jar),
            "-d",
            str(classes),
            *java_files,
        ]
    )
    unsigned = work / "unsigned.apk"
    compiled_resources = work / "compiled-resources.zip"
    _run(
        [
            str(tools / "aapt2"),
            "compile",
            "--dir",
            str(source / "res"),
            "-o",
            str(compiled_resources),
        ]
    )
    _run(
        [
            str(tools / "aapt2"),
            "link",
            "-I",
            str(android_jar),
            "--manifest",
            str(source / "AndroidManifest.xml"),
            "--min-sdk-version",
            "26",
            "--target-sdk-version",
            "35",
            "--version-code",
            "1",
            "--version-name",
            VERSION,
            "-o",
            str(unsigned),
            str(compiled_resources),
        ]
    )
    class_files = sorted(str(p) for p in classes.rglob("*.class"))
    _run([str(tools / "d8"), "--lib", str(android_jar), "--output", str(dex), *class_files])
    with zipfile.ZipFile(unsigned, "a", compression=zipfile.ZIP_DEFLATED) as out:
        out.write(dex / "classes.dex", "classes.dex")
        for library in native.glob("*/libtun2proxy.so"):
            out.write(library, f"lib/{library.parent.name}/libtun2proxy.so")

    aligned = work / "aligned.apk"
    _run([str(tools / "zipalign"), "-f", "4", str(unsigned), str(aligned)])
    keystore = cache / "debug.keystore"
    if not keystore.exists():
        _run(
            [
                "keytool",
                "-genkeypair",
                "-keystore",
                str(keystore),
                "-storepass",
                "android",
                "-alias",
                "androiddebugkey",
                "-keypass",
                "android",
                "-dname",
                "CN=Android Debug,O=Android,C=US",
                "-keyalg",
                "RSA",
                "-validity",
                "10000",
            ]
        )
    _run(
        [
            str(tools / "apksigner"),
            "sign",
            "--ks",
            str(keystore),
            "--ks-pass",
            "pass:android",
            "--key-pass",
            "pass:android",
            "--out",
            str(apk),
            str(aligned),
        ]
    )
    return apk


def start(
    serial: str,
    cache_dir: str | Path,
    port: int,
    *,
    target_package: str | None,
    ca_pem: Path | None,
) -> dict[str, object]:
    apk = build_apk(cache_dir)
    installed = _run(["adb", "-s", serial, "shell", "dumpsys", "package", PACKAGE])
    if f"versionName={VERSION}" not in installed:
        try:
            _run(["adb", "-s", serial, "install", "-r", str(apk)], timeout=300)
        except UsageError as exc:
            if "INSTALL_FAILED_UPDATE_INCOMPATIBLE" not in str(exc):
                raise
            _run(["adb", "-s", serial, "uninstall", PACKAGE], timeout=60)
            _run(["adb", "-s", serial, "install", str(apk)], timeout=300)

    extras = ["--ei", "proxy_port", str(port)]
    if target_package and target_package != PACKAGE:
        extras += ["--es", "target_package", target_package]
    ca_sha: str | None = None
    if ca_pem and ca_pem.exists():
        ca_sha = hashlib.sha256(ca_pem.read_bytes()).hexdigest()
        der = base64.b64decode(ssl.PEM_cert_to_DER_cert(ca_pem.read_text(encoding="ascii")))
        extras += [
            "--es",
            "ca_base64",
            base64.b64encode(der).decode("ascii"),
            "--es",
            "ca_sha",
            ca_sha,
        ]
    _run(
        [
            "adb",
            "-s",
            serial,
            "shell",
            "am",
            "start",
            "-n",
            f"{PACKAGE}/.MainActivity",
            *extras,
        ]
    )
    return {
        "package": PACKAGE,
        "version": VERSION,
        "apk": str(apk),
        "target_package": target_package,
        "ca_sha256": ca_sha,
        "permission_pending": True,
    }


def stop(serial: str) -> bool:
    try:
        _run(["adb", "-s", serial, "shell", "am", "force-stop", PACKAGE], timeout=30)
        return True
    except (UsageError, DeviceError):
        return False
