#!/usr/bin/env python3
"""Regenerate or check tracked lockfiles against the source set CI resolves.

# The trap this exists for

This repository is normally worked on inside the Atlas stack, whose
`.cargo/config.toml` carries a `[patch]` overlay redirecting every first-party
dependency to a local working tree. Cargo discovers that config by walking up
from the *current directory*, so any `cargo` command run from inside the stack
picks it up -- including anything that rewrites the lock.

A lock written with the overlay active has every `source = "git+..."` line
**stripped**, because those dependencies resolved to local paths rather than to
git. Committing it replaces all 87 git sources with nothing. CI has no overlay,
so it re-resolves, and every `--locked` job fails with

    error: cannot update the lock file ... because --locked was passed

which names neither the cause nor the fix. That message is also what a merely
*stale* lock produces -- one pinning first-party revisions whose versions no
longer satisfy the manifests -- so the two failures are indistinguishable from
the log alone (KW-CI-087).

This is not limited to deliberate regeneration. *Any* cargo invocation that
updates the lock while the overlay is active flattens it -- an ordinary
`cargo check` inside the stack is enough, which is how it happens in practice:
nobody sets out to rewrite the lock. Treat a modified `Cargo.lock` after routine
work as suspect and run `--check` before staging it.

Both are fixed the same way: regenerate from outside the overlay. This script
does that by running cargo from a temporary directory that is not underneath the
stack root, which is the whole mechanism -- there is no flag that disables config
discovery.

# Usage

    scripts/lockfile.py --check         # verify committed locks, offline
    scripts/lockfile.py --check-staged  # fast index-only check, for pre-commit
    scripts/lockfile.py --regenerate    # rewrite it correctly (needs network)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
LOCK_TARGETS = (
    (
        "workspace",
        REPOSITORY / "Cargo.lock",
        REPOSITORY / "Cargo.toml",
        "Cargo.lock",
    ),
    (
        "fuzz",
        REPOSITORY / "fuzz" / "Cargo.lock",
        REPOSITORY / "fuzz" / "Cargo.toml",
        "fuzz/Cargo.lock",
    ),
)

# Any first-party dependency resolves through one of these. A lock with none of
# them has been flattened by the overlay.
FIRST_PARTY_SOURCE = re.compile(r'^source = "git\+https://github\.com/ryancinsight/', re.M)


def run_outside_the_overlay(
    arguments: list[str], manifest: Path
) -> subprocess.CompletedProcess[str]:
    """Run cargo with a working directory outside the stack root.

    Cargo resolves `.cargo/config.toml` by walking up from the working
    directory, never from `--manifest-path`, so this is what excludes the
    overlay. Running from the repository itself would silently include it.
    """
    with tempfile.TemporaryDirectory() as neutral_directory:
        return subprocess.run(
            ["cargo", *arguments, "--manifest-path", str(manifest)],
            cwd=neutral_directory,
            capture_output=True,
            # `text=True` alone decodes with the locale codepage. Cargo emits
            # UTF-8, so on a Windows console (cp1252) subprocess's reader thread
            # dies on the first byte it cannot map and the captured stream is
            # lost. The verdict survives -- it comes from `returncode` -- but the
            # message explaining a failure does not, which is the one moment it
            # is needed.
            encoding="utf-8",
            errors="replace",
            check=False,
        )


def check() -> int:
    status = 0
    for label, lockfile, manifest, _relative in LOCK_TARGETS:
        if not lockfile.is_file():
            print(f"error: {label} lockfile {lockfile} does not exist", file=sys.stderr)
            status = 1
            continue

        sources = len(FIRST_PARTY_SOURCE.findall(lockfile.read_text(encoding="utf-8")))
        if sources == 0:
            print(
                f"error: {label} lockfile contains no first-party git sources.\n"
                "\n"
                "It was regenerated with the Atlas stack overlay active, which\n"
                "resolves those dependencies to local paths and drops their git\n"
                "sources. CI has no overlay and will fail every --locked job.\n"
                "\n"
                "Fix: scripts/lockfile.py --regenerate",
                file=sys.stderr,
            )
            status = 1
            continue

        completed = run_outside_the_overlay(
            ["metadata", "--locked", "--format-version", "1", "--all-features"],
            manifest,
        )
        if completed.returncode != 0:
            print(
                f"error: the committed {label} lockfile does not resolve under "
                f"--locked ({sources} first-party git sources present, so it is "
                f"stale rather than flattened).\n"
                "\n"
                "The pinned first-party revisions no longer satisfy the manifests'\n"
                "version requirements, so cargo must re-resolve and --locked\n"
                "refuses.\n"
                "\n"
                "Fix: scripts/lockfile.py --regenerate\n"
                f"\ncargo said:\n{completed.stderr.strip()}",
                file=sys.stderr,
            )
            status = 1
            continue

        print(f"{label} lockfile resolves under --locked; {sources} first-party git sources.")
    return status


def check_staged() -> int:
    """Structural check of the *staged* `Cargo.lock`, for use from `pre-commit`.

    Deliberately does not run cargo. A pre-commit hook has to be fast enough that
    nobody reaches for `--no-verify`, and the flattened lock has an unmistakable
    signature -- zero first-party git sources -- that a text scan settles
    instantly. Staleness, the other failure `--check` detects, needs real
    resolution and stays a pre-push concern.

    Checking the *staged blob* rather than the working file is the point: the
    working copy may already have been repaired while the poisoned version sits
    in the index, and it is the index that becomes the commit.
    """
    for label, _lockfile, _manifest, relative in LOCK_TARGETS:
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--", relative],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if staged.returncode != 0 or not staged.stdout.strip():
            continue

        blob = subprocess.run(
            ["git", "show", f":{relative}"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if blob.returncode != 0 or len(FIRST_PARTY_SOURCE.findall(blob.stdout)) > 0:
            continue

        print(
            f"error: the staged {label} lockfile contains no first-party git sources.\n"
            "\n"
            "A cargo command run against a tree under the Atlas stack root rewrote\n"
            "it with the overlay active, which resolves those dependencies to local\n"
            "paths and drops their git sources. Committing it now is what turns a\n"
            "working branch into one that can never be pushed.\n"
            "\n"
            "Fix: scripts/lockfile.py --regenerate, then stage the result.\n"
            "To commit anyway: SKIP_LOCKFILE_CHECK=1 git commit",
            file=sys.stderr,
        )
        return 1
    return 0


def regenerate() -> int:
    for label, _lockfile, manifest, _relative in LOCK_TARGETS:
        completed = run_outside_the_overlay(["generate-lockfile"], manifest)
        if completed.returncode != 0:
            print(
                f"error: {label} lockfile regeneration failed:\n{completed.stderr.strip()}",
                file=sys.stderr,
            )
            return 1
        print(f"{label} lockfile regenerated outside the overlay.")
    return check()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify the committed lock")
    mode.add_argument("--regenerate", action="store_true", help="rewrite the lock correctly")
    mode.add_argument(
        "--check-staged",
        action="store_true",
        help="fast structural check of the staged lock, for pre-commit",
    )
    arguments = parser.parse_args()
    if arguments.regenerate:
        return regenerate()
    if arguments.check_staged:
        return check_staged()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
