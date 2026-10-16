#!/usr/bin/env python3
"""Produce a selected Rust fixture before its separate native runtime test.

Only closed fixture identity, source ownership and authored markup vary. One
Cargo JSON/module-admission/bindgen path owns both cases. The bridge default
retains its source set, authored bytes, output names and compilation contract.
No generated payload or build receipt establishes native fixture acceptance.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import tomllib
from types import MappingProxyType

from verify_browser import file_digest, run_logged, text_command


BUILD_TIMEOUT = 600  # The existing verify_browser compilation owner bound.
FAVICON = "moirai-pal/examples/webview_bridge_fixture/favicon.svg"
HTML_PREFIX = ('<!doctype html><meta charset="utf-8">'
               '<link rel="icon" type="image/svg+xml" href="./favicon.svg"><body>')
HTML_SUFFIX = '<script type="module" src="./bootstrap.js"></script></body>'


@dataclass(frozen=True)
class Fixture:
    example: str
    body: str
    kind: str

    @property
    def source_paths(self):
        source = f"moirai-pal/examples/{self.example}"
        return ("Cargo.lock", "Cargo.toml", "moirai-pal/Cargo.toml",
                source + ".rs", source + "/browser.rs", source + "/protocol.rs", FAVICON,
                "moirai-pal/src/wasm/webview.rs", "moirai-pal/src/wasm/webview/listener.rs",
                "moirai-pal/src/wasm/webview/endpoint.rs")

    @property
    def payload_names(self):
        return ("bootstrap.js", "favicon.svg", "index.html",
                self.example + ".js", self.example + "_bg.wasm")


FIXTURES = MappingProxyType({
    "bridge": Fixture("webview_bridge_fixture",
                      '<p id="fixture-status">Rust bridge fixture loading</p>',
                      "actual-rust-native-bridge-payload"),
    "keyboard": Fixture("webview_keyboard_fixture",
                        '<input id="keyboard-target" autofocus>'
                        '<p id="fixture-status">Rust keyboard fixture loading</p>',
                        "actual-rust-native-keyboard-payload"),
})


def write_payload(root, payload, fixture):
    # This is only the ABI loader. Every fixture operation is authored Rust.
    (payload / "bootstrap.js").write_text(
        f"import init from './{fixture.example}.js';\nawait init();\n")
    shutil.copyfile(root / FAVICON, payload / "favicon.svg")
    (payload / "index.html").write_text(HTML_PREFIX + fixture.body + HTML_SUFFIX)


def actual_module(log, manifest, target, fixture):
    candidates = []
    source = manifest.parent / f"moirai-pal/examples/{fixture.example}.rs"
    for line in log.splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if (value.get("reason") != "compiler-artifact"
                or value["target"]["name"] != fixture.example):
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
    fixture = FIXTURES[args.fixture]
    manifest = args.manifest_path.resolve(strict=True)
    root = manifest.parent
    target = args.target_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    payload.mkdir()
    sources = {name: file_digest(root / name) for name in fixture.source_paths}
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
                   "--no-default-features", "--example", fixture.example,
                   "--target", "wasm32-unknown-unknown", "--message-format=json"]
        log, seconds = run_logged(command, neutral, environment,
                                  output / "build.jsonl", BUILD_TIMEOUT)
        module = actual_module(log, manifest, target, fixture)
        bindgen = [str(compiler), str(module), "--target", "web", "--out-dir",
                   str(payload), "--out-name", fixture.example, "--no-typescript"]
        _, bindgen_seconds = run_logged(bindgen, neutral, environment,
                                       output / "bindgen.log", BUILD_TIMEOUT)
        write_payload(root, payload, fixture)
        if sources != {name: file_digest(root / name) for name in fixture.source_paths}:
            raise ValueError("Rust fixture source/lock changes during actual compilation")
        if file_digest(compiler) != compiler_hash:
            raise ValueError("The selected wasm-bindgen executable changes during compilation")
        files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": file_digest(p)}
                 for p in sorted(payload.iterdir())]
        if tuple(p["path"] for p in files) != fixture.payload_names:
            raise ValueError("Actual compiler output contains unexpected fixture payload members")
        evidence = {"schema": 1, "kind": fixture.kind,
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


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", choices=FIXTURES, default="bridge")
    return parser.parse_args(argv)


def main():
    build(arguments())


if __name__ == "__main__":
    main()
