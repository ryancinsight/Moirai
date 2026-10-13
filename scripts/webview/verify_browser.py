#!/usr/bin/env python3
"""Run actual PAL WASM browser contracts with a matching official Chrome driver.

Chrome's M115+ selection contract admits matching MAJOR.MINOR.BUILD:
https://developer.chrome.com/docs/chromedriver/downloads/version-selection
Compilation and runner setup are separate from the strict 60-second browser
execution. Every exported wasm-bindgen test runs; absence/ignored tests fail.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
import urllib.request
import zipfile


DOWNLOAD_LIMIT = 64 * 1024 * 1024
EVIDENCE_LIMIT = 1024 * 1024
TEST_TIMEOUT = 60
VERSION = re.compile(r"\b\d+\.\d+\.\d+\.\d+\b")
REQUIRED_DOMAINS = ("wasm::webview::tests::", "wasm::dom::event::listener::tests::", "wasm::number::tests::")


def text_command(command, cwd, timeout=10):
    return subprocess.check_output(command, cwd=cwd, text=True, timeout=timeout).strip()


def download(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(DOWNLOAD_LIMIT + 1)
    if len(data) > DOWNLOAD_LIMIT:
        raise ValueError("Official browser tool response exceeds the download bound")
    return data


def driver_platform():
    system, machine = platform.system(), platform.machine()
    choices = {("Darwin", "arm64"): "mac-arm64", ("Darwin", "x86_64"): "mac-x64",
               ("Linux", "x86_64"): "linux64", ("Linux", "aarch64"): "linux-arm64"}
    if (system, machine) not in choices:
        raise ValueError(f"Unsupported Chrome driver platform: {system}/{machine}")
    return choices[(system, machine)]


def installed_driver(browser, directory):
    versions = VERSION.findall(text_command([str(browser), "--version"], directory))
    if len(versions) != 1:
        raise ValueError("Installed Chrome does not report one authoritative version")
    browser_version = versions[0]
    build = browser_version.rsplit(".", 1)[0]
    manifest_url = "https://googlechromelabs.github.io/chrome-for-testing/latest-patch-versions-per-build-with-downloads.json"
    manifest = json.loads(download(manifest_url))
    entry = manifest["builds"][build]
    version = entry["version"]
    if version.rsplit(".", 1)[0] != build:
        raise ValueError("Official Chrome driver build differs from the installed browser")
    selected_platform = driver_platform()
    records = [item for item in entry["downloads"]["chromedriver"] if item["platform"] == selected_platform]
    expected_url = f"https://storage.googleapis.com/chrome-for-testing-public/{version}/{selected_platform}/chromedriver-{selected_platform}.zip"
    if len(records) != 1 or records[0]["url"] != expected_url:
        raise ValueError("Official driver manifest does not identify the expected exact asset")
    archive = download(expected_url)
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        member = source.getinfo(f"chromedriver-{selected_platform}/chromedriver")
        if member.file_size > DOWNLOAD_LIMIT:
            raise ValueError("Official Chrome driver executable exceeds its allocation bound")
        executable = source.read(member)
    driver = directory / "chromedriver"
    driver.write_bytes(executable)
    driver.chmod(0o755)
    observed = text_command([str(driver), "--version"], directory)
    if VERSION.findall(observed) != [version]:
        raise ValueError("Downloaded driver version differs from its official manifest")
    return driver, {"browser_version": browser_version, "driver_version": observed,
                    "matching_build": build, "manifest_url": manifest_url, "driver_url": expected_url,
                    "archive_sha256": hashlib.sha256(archive).hexdigest(),
                    "driver_sha256": hashlib.sha256(executable).hexdigest()}


def run_logged(command, cwd, environment, output, timeout):
    started = time.monotonic()
    with output.open("wb") as stream:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if output.stat().st_size > EVIDENCE_LIMIT:
        raise ValueError(f"Verification log exceeds its evidence bound: {output}")
    text = output.read_text()
    if result.returncode:
        raise RuntimeError(f"Verification exits {result.returncode}; complete bounded log: {output}\n{text}")
    return text, time.monotonic() - started


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_module(log, target):
    candidates = []
    for line in log.splitlines():
        if not line.startswith("{"):
            continue  # Cargo's human-readable status messages are on stderr.
        record = json.loads(line)
        if (record.get("reason") == "compiler-artifact" and record["target"]["name"] == "moirai_pal"
                and record["profile"]["test"]):
            candidates.extend(Path(name).resolve() for name in record["filenames"] if name.endswith(".wasm"))
    if len(candidates) != 1 or not candidates[0].is_relative_to(target):
        raise ValueError("Cargo does not report exactly one PAL test module in the requested target directory")
    return candidates[0]


def verify(args):
    manifest = args.manifest_path.resolve(strict=True)
    root = manifest.parent
    browser = args.browser.resolve(strict=True)
    target = args.target_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_bytes = (root / "Cargo.lock").read_bytes()
    packages = tomllib.loads(lock_bytes.decode())["package"]
    versions = [package["version"] for package in packages if package["name"] == "wasm-bindgen"]
    if len(versions) != 1:
        raise ValueError("Canonical lock must identify exactly one wasm-bindgen compiler version")
    runner_path = shutil.which("wasm-bindgen-test-runner")
    if runner_path is None:
        raise ValueError("The locked matching wasm-bindgen-test-runner is not installed")
    runner = Path(runner_path).resolve(strict=True)
    observed = text_command([str(runner), "--version"], output)
    if re.findall(r"\d+\.\d+\.\d+", observed) != versions:
        raise ValueError(f"Installed WASM runner {observed} differs from locked compiler {versions}")
    with tempfile.TemporaryDirectory(prefix="moirai-browser-contract-") as temporary:
        neutral = Path(temporary)
        driver, browser_record = installed_driver(browser, neutral)
        capabilities = {"goog:chromeOptions": {"binary": str(browser),
                        "args": [f"--user-data-dir={neutral / 'profile'}"]}}
        configuration = neutral / "webdriver.json"
        configuration.write_text(json.dumps(capabilities))
        environment = dict(os.environ)
        # The actual browser crypto backend preserves property-test randomness;
        # no OS fork or dummy entropy backend is introduced on wasm32.
        environment["RUSTFLAGS"] = (environment.get("RUSTFLAGS", "") + ' --cfg getrandom_backend="wasm_js"').strip()
        environment["CHROMEDRIVER"] = str(driver)
        environment["WASM_BINDGEN_TEST_WEBDRIVER_JSON"] = str(configuration)
        environment["WASM_BINDGEN_TEST_TIMEOUT"] = str(TEST_TIMEOUT)
        environment["WASM_BINDGEN_TEST_DRIVER_TIMEOUT"] = str(TEST_TIMEOUT)
        command = ["cargo", "+1.97.0", "test", "--manifest-path", str(manifest), "--locked",
                   "--target-dir", str(target), "-p", "moirai-pal", "--lib", "--no-default-features",
                   "--target", "wasm32-unknown-unknown", "--no-run", "--message-format=json"]
        build, build_seconds = run_logged(command, neutral, environment, output / "build.jsonl", 600)
        module = test_module(build, target)
        listed = text_command([str(runner), str(module), "--list"], neutral, TEST_TIMEOUT)
        names = [line.removesuffix(": test") for line in listed.splitlines() if line.endswith(": test")]
        if not names or any(not any(domain in name for name in names) for domain in REQUIRED_DOMAINS):
            raise ValueError(f"Compiled real browser test inventory misses a required domain: {listed}")
        log, test_seconds = run_logged([str(runner), str(module), "--nocapture"], neutral,
                                      environment, output / "browser.log", TEST_TIMEOUT)
        summary = re.search(r"test result: ok\. (\d+) passed; 0 failed; 0 ignored; 0 filtered out", log)
        if summary is None or int(summary[1]) != len(names):
            raise ValueError("Real browser summary does not prove every compiled test passes without skips")
        if (root / "Cargo.lock").read_bytes() != lock_bytes:
            raise ValueError("Canonical lock changed during verification")
        evidence = {"status": "passed", **browser_record, "runner_version": observed,
                    "runner_sha256": file_digest(runner),
                    "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
                    "module_sha256": file_digest(module),
                    "source_commit": text_command(["git", "-C", str(root), "rev-parse", "HEAD"], neutral),
                    "source_dirty": bool(text_command(["git", "-C", str(root), "status", "--porcelain"], neutral)),
                    "tests": names, "passed": len(names), "failed": 0, "ignored": 0,
                    "build_seconds": build_seconds, "test_seconds": test_seconds,
                    "test_timeout_seconds": TEST_TIMEOUT, "isolated_profile": True}
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
