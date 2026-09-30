#!/usr/bin/env python3
"""Run the local benchmark suite under a committed wall-clock budget.

Benchmarks are local measurement instruments (performance policy): CI only
smoke-runs each once, and the timing run executes here, on the controlled host.
A suite that cannot finish inside its budget is an oversized instrument, not a
slow machine, so the run stops at the first breach and names the target that
crossed the line.

The runner builds every `moirai-benchmarks` bench binary once with
`cargo bench --no-run`, then runs each executable with `--bench` in turn and
times only the run, so compile time never counts against the budget. Cargo runs
from a temporary directory outside the Atlas stack, for the reason
`scripts/lockfile.py` documents: inside the stack its `[patch]` overlay would
rewrite `Cargo.lock`.

Usage:

    scripts/bench_suite.py                  # 300 s total budget
    scripts/bench_suite.py --budget 600     # a different committed bound
    scripts/bench_suite.py --only channel_matrix --only spsc_throughput

Exit status is 0 when every target finished inside the remaining budget, 1 when
one crossed it or failed, and 2 when the benchmark binaries could not be built.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "Cargo.toml"
PACKAGE = "moirai-benchmarks"
DEFAULT_BUDGET_SECONDS = 300.0


def build_executables(cwd: Path) -> dict[str, Path]:
    """Builds every bench binary and returns `{target name: executable}`."""
    command = [
        "cargo",
        "bench",
        "--manifest-path",
        str(MANIFEST),
        "--locked",
        "-p",
        PACKAGE,
        "--no-run",
        "--message-format=json-render-diagnostics",
    ]
    completed = subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE, text=True, check=False
    )
    if completed.returncode != 0:
        raise SystemExit(2)
    executables: dict[str, Path] = {}
    for line in completed.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact" or not message.get("executable"):
            continue
        target = message["target"]
        if "bench" in target["kind"]:
            executables[target["name"]] = Path(message["executable"])
    return executables


def kill_tree(process: subprocess.Popen[bytes]) -> None:
    """Stops a benchmark and every process it started."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.kill()
    process.wait()


def run_target(
    command: list[str], timeout: float, workdir: Path
) -> tuple[float, str]:
    """Runs one bench command; returns its wall time and `ok`/`timeout`/`failed`."""
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(process)
        return time.monotonic() - start, "timeout"
    elapsed = time.monotonic() - start
    return elapsed, "ok" if process.returncode == 0 else "failed"


def run_suite(
    commands: dict[str, list[str]], budget: float, workdir: Path
) -> int:
    """Runs the commands in name order inside one wall-clock budget."""
    remaining = budget
    total = 0.0
    for name in sorted(commands):
        elapsed, outcome = run_target(commands[name], remaining, workdir)
        total += elapsed
        remaining -= elapsed
        print(f"{name:<48} {elapsed:8.1f} s  {outcome}")
        if outcome == "timeout":
            print(
                f"BUDGET BREACH: {name} was still running when the "
                f"{budget:.0f} s suite budget ended ({total:.1f} s used)",
                file=sys.stderr,
            )
            return 1
        if outcome == "failed":
            print(f"FAILED: {name} exited with an error", file=sys.stderr)
            return 1
    print(f"{'total':<48} {total:8.1f} s of {budget:.0f} s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--only", action="append", default=[], metavar="TARGET")
    arguments = parser.parse_args()

    if not os.environ.get("CARGO_TARGET_DIR"):
        print(
            "CARGO_TARGET_DIR is unset: building outside the stack would fork "
            "the shared build cache; point it at the shared target directory",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="moirai-bench-") as scratch:
        scratch_path = Path(scratch)
        executables = build_executables(scratch_path)
        names = arguments.only or list(executables)
        unknown = [name for name in names if name not in executables]
        if unknown:
            print(f"unknown bench target(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        commands = {name: [str(executables[name]), "--bench"] for name in names}
        return run_suite(commands, arguments.budget, scratch_path)


if __name__ == "__main__":
    sys.exit(main())
