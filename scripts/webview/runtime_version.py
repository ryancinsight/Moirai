"""Query availability through the SDK selected by the native Windows build."""

import ctypes
import json
from pathlib import Path
import subprocess
import tomllib


def native_sdk(records):
    """Join public Cargo records by opaque package ID, without cache guesses.

    The SDK build script copies its loader into the emitted native link-search
    directory. Exactly one successful native build, SDK library, and matching
    build-script record establish which loader and manifest own this query.
    """
    finished = [item for item in records if item.get("reason") == "build-finished"]
    if len(finished) != 1 or finished[0].get("success") is not True:
        raise RuntimeError("expected one successful native SDK check")
    artifacts = [
        item
        for item in records
        if item.get("reason") == "compiler-artifact"
        and item.get("target", {}).get("name") == "webview2_com_sys"
        and item["target"].get("kind") == ["lib"]
    ]
    if len(artifacts) != 1:
        raise RuntimeError("expected one checked native WebView2 SDK library")
    artifact = artifacts[0]
    builds = [
        item
        for item in records
        if item.get("reason") == "build-script-executed"
        and item.get("package_id") == artifact["package_id"]
    ]
    if len(builds) != 1:
        raise RuntimeError("expected one matching native SDK build-script record")
    native = [
        Path(path.removeprefix("native="))
        for path in builds[0]["linked_paths"]
        if path.startswith("native=")
    ]
    if len(native) != 1 or not native[0].is_absolute():
        raise RuntimeError("expected one absolute SDK native link-search directory")
    manifest = Path(artifact["manifest_path"])
    if not manifest.is_absolute():
        raise RuntimeError("expected an absolute checked SDK manifest path")
    with manifest.open("rb") as source:
        package = tomllib.load(source)["package"]
    version = package.get("version")
    if (
        package.get("name") != "webview2-com-sys"
        or not isinstance(version, str)
        or not version.strip()
    ):
        raise RuntimeError("checked SDK manifest omits its package identity/version")
    return native[0] / "WebView2Loader.dll", version


def main():
    # Prior native tests already compile this exact locked feature graph.
    # Offline native checking emits authoritative artifacts without asking
    # metadata to fetch unrelated Apple/Linux/WASI source packages.
    output = subprocess.check_output(
        [
            "cargo", "check", "--locked", "--offline", "-p", "moirai-pal", "--lib",
            "--no-default-features", "--features", "webview2",
            "--message-format=json",
        ],
        timeout=60,
        encoding="utf-8",
    )
    loader_path, sdk_version = native_sdk(
        [json.loads(line) for line in output.splitlines()]
    )
    loader = ctypes.WinDLL(str(loader_path))
    available = loader.GetAvailableCoreWebView2BrowserVersionString
    available.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
    available.restype = ctypes.c_long
    allocator = ctypes.WinDLL("ole32")
    allocator.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    allocator.CoTaskMemFree.restype = None
    version_pointer = ctypes.c_void_p()
    result = available(None, ctypes.byref(version_pointer))
    try:
        if result < 0:
            raise OSError(f"native WebView2 version query failed: HRESULT {result:#x}")
        if not version_pointer.value:
            raise RuntimeError("native WebView2 version query returned no runtime")
        version = ctypes.wstring_at(version_pointer.value)
        if not version:
            raise RuntimeError("native WebView2 version query returned an empty version")
        print(json.dumps({
            "native_api": "GetAvailableCoreWebView2BrowserVersionString",
            "runtime_version": version,
            "sdk_version": sdk_version,
            "loader_path": str(loader_path),
        }, sort_keys=True))
    finally:
        allocator.CoTaskMemFree(version_pointer)


if __name__ == "__main__":
    main()
