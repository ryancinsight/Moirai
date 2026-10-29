"""Verify the release graph at its declared direct dependency floors."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

COMMAND_TIMEOUT_SECONDS = 600
NIGHTLY_TOOLCHAIN = "nightly-2026-08-01"
MSRV_TOOLCHAIN = "1.95.0"


class GateFailure(RuntimeError):
    """Report a dependency-floor contract violation."""


def _node_name(dependency: dict[str, Any]) -> str:
    return (dependency.get("rename") or dependency["name"]).replace("-", "_")


def direct_registry_dependencies(metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Return resolved direct registry edges for every workspace member."""
    packages = {package["id"]: package for package in metadata["packages"]}
    resolve = metadata.get("resolve")
    if resolve is None:
        raise GateFailure("locked metadata omitted its resolve graph")
    nodes = {node["id"]: node for node in resolve["nodes"]}
    selected: list[dict[str, str]] = []
    for member_id in sorted(metadata["workspace_members"]):
        member = packages[member_id]
        node = nodes.get(member_id)
        if node is None:
            raise GateFailure(f"workspace member has no resolve node: {member_id}")
        for dependency in member["dependencies"]:
            source = dependency.get("source") or ""
            if not source.startswith("registry+"):
                continue
            alias = _node_name(dependency)
            kind = dependency.get("kind") or "normal"
            target = dependency.get("target") or ""
            matches = []
            for resolved in node["deps"]:
                if resolved["name"] != alias:
                    continue
                package = packages[resolved["pkg"]]
                if not (package.get("source") or "").startswith("registry+"):
                    continue
                if any(
                    (dep_kind.get("kind") or "normal") == kind
                    and (dep_kind.get("target") or "") == target
                    for dep_kind in resolved["dep_kinds"]
                ):
                    matches.append(package)
            if not matches:
                raise GateFailure(
                    f"direct registry edge did not resolve: {member['name']}:{alias}"
                )
            for package in matches:
                selected.append(
                    {
                        "member": member["name"],
                        "alias": alias,
                        "package": dependency["name"],
                        "kind": kind,
                        "target": target,
                        "requirement": dependency["req"],
                        "version": package["version"],
                        "source": package["source"],
                        "package_id": package["id"],
                    }
                )
    return sorted(
        selected,
        key=lambda item: (
            item["member"],
            item["alias"],
            item["kind"],
            item["target"],
            item["package_id"],
        ),
    )


def _edge_key(dependency: dict[str, str]) -> tuple[str, ...]:
    return (
        dependency["member"],
        dependency["alias"],
        dependency["package"],
        dependency["kind"],
        dependency["target"],
        dependency["requirement"],
        dependency["source"],
    )


def assert_direct_edges_preserved(
    locked: list[dict[str, str]], minimum: list[dict[str, str]]
) -> None:
    """Prove the minimum graph resolves every unchanged direct declaration."""
    before = sorted(_edge_key(dependency) for dependency in locked)
    after = sorted(_edge_key(dependency) for dependency in minimum)
    if before != after:
        raise GateFailure(
            "direct-minimal resolution changed the direct registry edge set: "
            f"locked={before!r}, minimum={after!r}"
        )


def assert_minimum_versions_at_declared_floors(
    dependencies: list[dict[str, str]],
) -> None:
    """Reject a minimum graph that did not select each declared caret floor."""
    mismatches = []
    for dependency in dependencies:
        expected = f"^{dependency['version']}"
        if dependency["requirement"] != expected:
            mismatches.append(
                f"{dependency['member']}:{dependency['alias']} declares "
                f"{dependency['requirement']}, selected {dependency['version']}"
            )
    if mismatches:
        raise GateFailure(
            "direct-minimal resolution did not select declared floors:\n"
            + "\n".join(mismatches)
        )


def git_sources(lock: bytes) -> list[tuple[str, str, str]]:
    """Extract every Git package identity from a Cargo lockfile."""
    parsed = tomllib.loads(lock.decode("utf-8"))
    return sorted(
        (package["name"], package["version"], package["source"])
        for package in parsed.get("package", [])
        if package.get("source", "").startswith("git+")
    )


def assert_git_sources_unchanged(before: bytes, after: bytes) -> None:
    """Reject any resolver movement of a committed Git dependency."""
    locked = git_sources(before)
    minimum = git_sources(after)
    if locked != minimum:
        raise GateFailure(
            "direct-minimal resolution changed Git package identities: "
            f"locked={locked!r}, minimum={minimum!r}"
        )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def candidate_receipt(candidate: Path, excluded: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in sorted(candidate.rglob("*")):
        if not path.is_file() or path.is_relative_to(excluded):
            continue
        relative = path.relative_to(candidate).as_posix()
        files[relative] = _sha256(path.read_bytes())
    digest = hashlib.sha256()
    for relative, content_hash in files.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "files": files}


def _load_process_tree(root: Path) -> Any:
    if not (root / "process_tree.py").is_file() or not (
        root / "windows_process.py"
    ).is_file():
        raise GateFailure(f"process-tree runner is incomplete: {root}")
    sys.path.insert(0, str(root))
    return importlib.import_module("process_tree")


def _capture(
    process_tree: Any, command: list[str], cwd: Path, log: Path
) -> str:
    try:
        result = process_tree.run(
            command,
            cwd=str(cwd),
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except Exception as error:
        raise GateFailure(
            f"command runner failed: {' '.join(command)}: {error}"
        ) from error
    stdout = (result.stdout or b"").decode("utf-8", errors="replace")
    stderr = (result.stderr or b"").decode("utf-8", errors="replace")
    log.write_text(stdout + stderr, encoding="utf-8")
    if result.returncode != 0:
        raise GateFailure(f"command failed ({result.returncode}): {' '.join(command)}")
    return stdout


def _run(process_tree: Any, command: list[str], cwd: Path, log: Path) -> None:
    try:
        result = process_tree.run(
            command,
            cwd=str(cwd),
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except Exception as error:
        raise GateFailure(
            f"command runner failed: {' '.join(command)}: {error}"
        ) from error
    stdout = result.stdout or b""
    stderr = result.stderr or b""
    log.write_bytes(stdout + stderr)
    if result.returncode != 0:
        raise GateFailure(f"command failed ({result.returncode}): {' '.join(command)}")


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def verify(
    candidate: Path, cargo_config: Path, runner_root: Path, output_root: Path
) -> dict[str, Any]:
    """Resolve direct floors, preserve Git pins, and check the graph at MSRV."""
    candidate = candidate.resolve(strict=True)
    cargo_config = cargo_config.resolve(strict=True)
    runner_root = runner_root.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
    working_directory = Path.cwd().resolve(strict=True)
    if not _within(working_directory, temporary_root):
        raise GateFailure("the gate must start from an operating-system scratch directory")
    if not _within(candidate, working_directory) or candidate == working_directory:
        raise GateFailure("candidate must be an explicit export below the scratch directory")
    if (candidate / ".git").exists():
        raise GateFailure("candidate must be an export, not a Git checkout")
    if not _within(output_root, candidate):
        raise GateFailure("output root must be inside the candidate export")
    manifest = candidate / "Cargo.toml"
    lock_path = candidate / "Cargo.lock"
    if not manifest.is_file() or not lock_path.is_file():
        raise GateFailure("candidate export lacks Cargo.toml or Cargo.lock")

    receipt = candidate_receipt(candidate, output_root)
    committed_lock = lock_path.read_bytes()
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "committed-Cargo.lock").write_bytes(committed_lock)
    process_tree = _load_process_tree(runner_root)

    nightly = ["cargo", f"+{NIGHTLY_TOOLCHAIN}", "--config", str(cargo_config)]
    msrv = ["cargo", f"+{MSRV_TOOLCHAIN}", "--config", str(cargo_config)]
    nightly_version = _capture(
        process_tree,
        [*nightly, "--version"],
        candidate,
        output_root / "nightly-version.log",
    ).strip()
    msrv_version = _capture(
        process_tree,
        [*msrv, "--version"],
        candidate,
        output_root / "msrv-version.log",
    ).strip()
    metadata_command = [
        *nightly,
        "metadata",
        "--locked",
        "--all-features",
        "--format-version",
        "1",
        "--manifest-path",
        str(manifest),
    ]
    locked_text = _capture(
        process_tree,
        metadata_command,
        candidate,
        output_root / "locked-metadata.log",
    )
    locked_metadata = json.loads(locked_text)
    locked_dependencies = direct_registry_dependencies(locked_metadata)
    package_ids = sorted(
        {dependency["package_id"] for dependency in locked_dependencies}
    )
    if not package_ids:
        raise GateFailure("workspace has no direct registry dependencies")

    update_command = [
        *nightly,
        "update",
        "-Zdirect-minimal-versions",
        "--manifest-path",
        str(manifest),
    ]
    for package_id in package_ids:
        update_command.extend(("-p", package_id))
    _run(
        process_tree,
        update_command,
        candidate,
        output_root / "direct-minimal-update.log",
    )
    minimum_lock = lock_path.read_bytes()
    assert_git_sources_unchanged(committed_lock, minimum_lock)
    (output_root / "minimum-Cargo.lock").write_bytes(minimum_lock)

    minimum_text = _capture(
        process_tree,
        metadata_command,
        candidate,
        output_root / "minimum-metadata.log",
    )
    minimum_metadata = json.loads(minimum_text)
    minimum_dependencies = direct_registry_dependencies(minimum_metadata)
    assert_direct_edges_preserved(locked_dependencies, minimum_dependencies)
    assert_minimum_versions_at_declared_floors(minimum_dependencies)

    check_command = [
        *msrv,
        "check",
        "--locked",
        "--workspace",
        "--all-features",
        "--all-targets",
        "--manifest-path",
        str(manifest),
    ]
    _run(
        process_tree,
        check_command,
        candidate,
        output_root / "msrv-check.log",
    )
    (output_root / "locked-metadata.json").write_text(
        locked_text, encoding="utf-8"
    )
    (output_root / "minimum-metadata.json").write_text(
        minimum_text, encoding="utf-8"
    )
    provenance = {
        "candidate": receipt,
        "cargo_config_sha256": _sha256(cargo_config.read_bytes()),
        "process_runner_sha256": {
            name: _sha256((runner_root / name).read_bytes())
            for name in ("process_tree.py", "windows_process.py")
        },
        "committed_lock_sha256": _sha256(committed_lock),
        "minimum_lock_sha256": _sha256(minimum_lock),
        "nightly": nightly_version,
        "msrv": msrv_version,
        "command_timeout_seconds": COMMAND_TIMEOUT_SECONDS,
        "committed_direct_registry_dependencies": locked_dependencies,
        "minimum_direct_registry_dependencies": minimum_dependencies,
        "git_sources": [
            {"name": name, "version": version, "source": source}
            for name, version, source in git_sources(committed_lock)
        ],
        "commands": {
            "locked_metadata": metadata_command,
            "direct_minimum_update": update_command,
            "minimum_metadata": metadata_command,
            "msrv_check": check_command,
        },
        "gates": {
            "committed_direct_registry_edges_enumerated": True,
            "direct_minimum_update_completed": True,
            "git_sources_unchanged": True,
            "minimum_graph_resolves_declared_direct_edges": True,
            "minimum_versions_equal_declared_floors": True,
            "msrv_workspace_all_features_all_targets": True,
        },
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--cargo-config", required=True, type=Path)
    parser.add_argument("--runner-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        verify(
            arguments.candidate,
            arguments.cargo_config,
            arguments.runner_root,
            arguments.output_root,
        )
    except (GateFailure, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"minimum-version gate failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
