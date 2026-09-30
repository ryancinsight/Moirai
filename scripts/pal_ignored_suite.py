#!/usr/bin/env python3
"""Run the ignored `moirai-pal` WebView2 suite from a private executable copy.

The ignored suite drives real browser controls and runs for minutes. Run in
place, it keeps `moirai_pal-*.exe` in the shared target directory open, and on
Windows every other tree's clippy or nextest of `moirai-pal` then fails with
`failed to remove file` when it tries to replace that executable. The suite
therefore runs from a `cargo nextest archive` extracted to a temporary
directory: the shared cache is read once to build the archive and never held
open afterward.

Cargo runs from a temporary directory outside the Atlas stack, for the reason
`scripts/lockfile.py` documents: inside the stack its `[patch]` overlay would
rewrite `Cargo.lock`.

Usage:

    scripts/pal_ignored_suite.py                  # every ignored test
    scripts/pal_ignored_suite.py -- webview       # nextest filter expressions

Exit status is nextest's, or 2 when the archive could not be built.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "Cargo.toml"
PACKAGE = "moirai-pal"
FEATURES = "webview2"


def main() -> int:
    if not os.environ.get("CARGO_TARGET_DIR"):
        print(
            "CARGO_TARGET_DIR is unset: building outside the stack would fork "
            "the shared build cache; point it at the shared target directory",
            file=sys.stderr,
        )
        return 2

    filters = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    # WebView2 browser processes outlive the test binary for a moment and keep
    # the extracted copy open on Windows; a directory that cannot be removed yet
    # is left for the temp cleaner rather than failing a passed run.
    with tempfile.TemporaryDirectory(
        prefix="moirai-pal-ignored-", ignore_cleanup_errors=True
    ) as scratch:
        scratch_path = Path(scratch)
        archive = scratch_path / "pal.tar.zst"
        build = subprocess.run(
            [
                "cargo", "nextest", "archive",
                "--manifest-path", str(MANIFEST),
                "--locked",
                "-p", PACKAGE,
                "--features", FEATURES,
                "--archive-file", str(archive),
            ],
            cwd=scratch_path,
            check=False,
        )
        if build.returncode != 0:
            return 2
        extracted = scratch_path / "extracted"
        extracted.mkdir()
        run = subprocess.run(
            [
                "cargo", "nextest", "run",
                "--archive-file", str(archive),
                "--extract-to", str(extracted),
                "--workspace-remap", str(REPO),
                "--run-ignored", "all",
                *filters,
            ],
            cwd=scratch_path,
            check=False,
        )
        return run.returncode


if __name__ == "__main__":
    sys.exit(main())
