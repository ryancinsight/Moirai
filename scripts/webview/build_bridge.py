#!/usr/bin/env python3
"""Produce original Rust bridge fixture bytes before the native runtime test."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import tomllib

from verify_browser import file_digest, run_logged, text_command


EXAMPLE = "webview_bridge_fixture"
BUILD_TIMEOUT = 600  # The existing verify_browser compilation owner bound.
SOURCE_PATHS = (
    "Cargo.lock", "Cargo.toml", "moirai-pal/Cargo.toml",
    "moirai-pal/examples/webview_bridge_fixture.rs",
    "moirai-pal/examples/webview_bridge_fixture/browser.rs",
    "moirai-pal/examples/webview_bridge_fixture/protocol.rs",
    "moirai-pal/src/wasm/webview.rs",
    "moirai-pal/src/wasm/webview/listener.rs",
    "moirai-pal/src/wasm/webview/endpoint.rs",
)


def actual_module(log, manifest, target):
    candidates = []
    source = manifest.parent / "moirai-pal/examples/webview_bridge_fixture.rs"
    for line in log.splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if (value.get("reason") != "compiler-artifact"
                or value["target"]["name"] != EXAMPLE):
            continue
        if (value["target"]["kind"] != ["example"]
                or value["target"]["crate_types"] != ["cdylib"]
                or value["profile"]["test"]
                or not Path(value["target"]["src_path"]).samefile(source)):
            raise ValueError("Cargo example roles/source differ from the actual Rust fixture")
        candidates.extend(Path(name).resolve(strict=True) for name in value["filenames"]
                          if Path(name).suffix == ".wasm")
    if len(candidates) != 1:
        raise ValueError("Cargo must report one actual Rust fixture WASM module")
    module = candidates[0]
    if not module.is_file() or not any(parent.samefile(target) for parent in module.parents):
        raise ValueError("Actual Rust fixture module lies outside the requested target owner")
    return module


def build(args):
    manifest = args.manifest_path.resolve(strict=True)
    root = manifest.parent
    target = args.target_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    payload.mkdir()
    sources = {name: file_digest(root / name) for name in SOURCE_PATHS}
    lock = (root / "Cargo.lock").read_bytes()
    versions = [p["version"] for p in tomllib.loads(lock.decode())["package"]
                if p["name"] == "wasm-bindgen"]
    if len(versions) != 1:
        raise ValueError("Canonical lock must identify one wasm-bindgen compiler")
    selected = shutil.which("wasm-bindgen")
    if selected is None:
        raise ValueError("The lock-matched actual wasm-bindgen compiler is required")
    compiler = Path(selected).resolve(strict=True)
    observed = text_command([str(compiler), "--version"], output)
    if re.findall(r"\d+\.\d+\.\d+", observed) != versions:
        raise ValueError(f"Actual wasm-bindgen version differs from the lock: {observed}")
    compiler_hash = file_digest(compiler)
    with tempfile.TemporaryDirectory(prefix="moirai-native-bridge-build-") as temporary:
        neutral = Path(temporary)
        environment = dict(os.environ)
        environment["RUSTFLAGS"] = (environment.get("RUSTFLAGS", "")
                                    + ' --cfg getrandom_backend="wasm_js"').strip()
        command = ["cargo", "+1.97.0", "build", "--manifest-path", str(manifest),
                   "--locked", "--target-dir", str(target), "-p", "moirai-pal",
                   "--no-default-features", "--example", EXAMPLE,
                   "--target", "wasm32-unknown-unknown", "--message-format=json"]
        log, seconds = run_logged(command, neutral, environment,
                                  output / "build.jsonl", BUILD_TIMEOUT)
        module = actual_module(log, manifest, target)
        bindgen = [str(compiler), str(module), "--target", "web", "--out-dir",
                   str(payload), "--out-name", EXAMPLE, "--no-typescript"]
        _, bindgen_seconds = run_logged(bindgen, neutral, environment,
                                       output / "bindgen.log", BUILD_TIMEOUT)
        # This is only the ABI loader. Every fixture operation is authored Rust.
        (payload / "bootstrap.js").write_text(
            "import init from './webview_bridge_fixture.js';\nawait init();\n")
        (payload / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><body>'
            '<p id="fixture-status">Rust bridge fixture loading</p>'
            '<script type="module" src="./bootstrap.js"></script></body>')
        if sources != {name: file_digest(root / name) for name in SOURCE_PATHS}:
            raise ValueError("Rust fixture source/lock changes during actual compilation")
        if file_digest(compiler) != compiler_hash:
            raise ValueError("The selected wasm-bindgen executable changes during compilation")
        files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": file_digest(p)}
                 for p in sorted(payload.iterdir())]
        if [p["path"] for p in files] != ["bootstrap.js", "index.html",
                                           EXAMPLE + ".js", EXAMPLE + "_bg.wasm"]:
            raise ValueError("Actual compiler output contains unexpected fixture payload members")
        evidence = {"schema": 1, "kind": "actual-rust-native-bridge-payload",
                    "source_commit": text_command(["git", "-C", str(root), "rev-parse", "HEAD"], neutral),
                    "source_dirty": bool(text_command(["git", "-C", str(root), "status", "--porcelain"], neutral)),
                    "sources": sources, "command": command, "bindgen_command": bindgen,
                    "rustc": text_command(["rustup", "run", "1.97.0", "rustc", "-vV"], neutral),
                    "compiler_version": observed, "compiler_sha256": compiler_hash,
                    "lock_sha256": hashlib.sha256(lock).hexdigest(),
                    "module_sha256": file_digest(module), "files": files,
                    "build_seconds": seconds, "bindgen_seconds": bindgen_seconds,
                    "build_timeout_seconds": BUILD_TIMEOUT, "payload": str(payload)}
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
