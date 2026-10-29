#!/usr/bin/env python3
"""Plan a registry release and compare independently produced release evidence.

The release workflow invokes this from a clean checkout.  It deliberately
queries the sparse index instead of inferring publication from Git tags: an
absent crate is recorded as bootstrap work. The checked cohort always contains
every publishable workspace package, including versions already indexed, so an
interrupted upload cannot narrow package, SBOM or reproducibility coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path
from typing import Any

import release_sbom


REGISTRY_READER = Path(".github/actions/crate-release-identity/release_identity.py")
ARTIFACT_EVIDENCE = Path(".github/actions/crates-pending/artifact_evidence.py")


def cargo_metadata(manifest: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1",
             "--manifest-path", str(manifest)],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"cargo metadata exceeded 60 seconds for {manifest}") from error
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "cargo metadata failed")
    return json.loads(result.stdout)


def load_atlas_module(root: Path, relative: Path, module_name: str, member: str) -> Any:
    """Load a required contract owner from an explicit Atlas checkout root."""
    source = (root / relative).resolve()
    if not source.is_file():
        raise RuntimeError(f"Atlas contract owner does not exist: {source}")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Atlas contract owner cannot be loaded: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    if not callable(getattr(module, member, None)):
        sys.modules.pop(spec.name, None)
        raise RuntimeError(f"Atlas contract owner lacks {member}: {source}")
    return module


def load_registry_reader(root: Path) -> Any:
    """Load the pinned Atlas registry reader from its explicit checkout root."""
    return load_atlas_module(
        root, REGISTRY_READER, "atlas_crate_release_identity", "sparse_index_versions"
    )


def is_publishable(package: dict[str, Any]) -> bool:
    registries = package.get("publish")
    return registries is None or (isinstance(registries, list) and "crates-io" in registries)


def cohort_entry(package: dict[str, Any]) -> dict[str, Any]:
    return {"name": package["name"], "version": package["version"]}


def select_cohort(packages: list[dict[str, Any]], versions_for: Any) -> dict[str, Any]:
    release = sorted(
        (package for package in packages if is_publishable(package)),
        key=lambda package: package["name"],
    )
    if not release:
        raise RuntimeError("workspace release has no publishable members")
    versions = {package["version"] for package in release}
    if len(versions) != 1:
        raise RuntimeError("workspace release members have different versions")
    names = [package["name"] for package in release]
    if len(names) != len(set(names)):
        raise RuntimeError("workspace release has duplicate package names")
    first: list[str] = []
    pending: list[str] = []
    for package in release:
        versions = versions_for(package["name"])
        if versions is None:
            first.append(package["name"])
        if versions is None or package["version"] not in versions:
            pending.append(package["name"])
    return {
        "version": release[0]["version"],
        "packages": names,
        "bootstrap": first,
        "pending": pending,
        "cohort": [cohort_entry(package) for package in release],
    }


def plan(manifest: Path, versions_for: Any) -> dict[str, Any]:
    metadata = cargo_metadata(manifest)
    members = set(metadata["workspace_members"])
    packages = [package for package in metadata["packages"] if package["id"] in members]
    if {package["id"] for package in packages} != members:
        raise RuntimeError("Cargo metadata omits a workspace member")
    return select_cohort(packages, versions_for)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(context: dict[str, str], artifacts: dict[str, str], cohort: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(context, dict) or not isinstance(artifacts, dict) or not isinstance(cohort, list):
        raise RuntimeError("malformed release evidence")
    if not artifacts:
        raise RuntimeError("release evidence contains no artifacts")
    required = {"candidate", "toolchain", "target", "profile"}
    missing = required - set(context)
    if missing:
        raise RuntimeError("release evidence lacks context: " + ", ".join(sorted(missing)))
    if any(not isinstance(context[key], str) or not context[key] for key in required):
        raise RuntimeError("release evidence has malformed context")
    if not cohort:
        raise RuntimeError("release evidence has no planned cohort")
    names = []
    for item in cohort:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "version"}
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not isinstance(item.get("version"), str)
            or not item["version"]
        ):
            raise RuntimeError("release evidence has malformed cohort")
        names.append((item["name"], item["version"]))
    if len(names) != len(set(names)):
        raise RuntimeError("release evidence has duplicate cohort members")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for name, value in artifacts.items()
    ):
        raise RuntimeError("release evidence has invalid SHA-256")
    return {"context": context, "cohort": cohort, "artifacts": artifacts}


def record(paths: list[Path], output: Path, context: dict[str, str], cohort: list[dict[str, Any]]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("missing release evidence: " + ", ".join(map(str, missing)))
    artifacts = {str(path.name): sha256(path) for path in sorted(paths)}
    if len(artifacts) != len(paths):
        raise RuntimeError("package archive set has duplicate basenames")
    evidence(context, artifacts, cohort)
    expected = {f"{item['name']}-{item['version']}.crate" for item in cohort}
    if set(artifacts) != expected:
        raise RuntimeError(f"package archive coverage differs: expected {sorted(expected)}, got {sorted(artifacts)}")
    for item in cohort:
        archive = next(path for path in paths if path.name == f"{item['name']}-{item['version']}.crate")
        prefix = f"{item['name']}-{item['version']}/"
        with tarfile.open(archive, "r:gz") as packed:
            manifest = packed.extractfile(prefix + "Cargo.toml")
            if manifest is None:
                raise RuntimeError(f"{archive.name} lacks normalized Cargo identity")
            identity = tomllib.loads(manifest.read().decode("utf-8")).get("package", {})
            if identity.get("name") != item["name"] or identity.get("version") != item["version"]:
                raise RuntimeError(f"{archive.name} has a mismatched package identity")
    output.write_text(
        json.dumps(evidence(context, artifacts, cohort), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_archive_evidence(value: dict[str, Any]) -> None:
    context, cohort, artifacts = value.get("context"), value.get("cohort"), value.get("artifacts")
    if not isinstance(context, dict) or not isinstance(cohort, list) or not isinstance(artifacts, dict):
        raise RuntimeError("malformed release evidence")
    evidence(context, artifacts, cohort)
    expected = {f"{item['name']}-{item['version']}.crate" for item in cohort}
    if set(artifacts) != expected:
        raise RuntimeError("release evidence has incomplete or unknown package archives")


def tool_version(manifest: Path, tool: str) -> str:
    tools = tomllib.loads(manifest.read_text(encoding="utf-8")).get("tools", {})
    version = tools.get(tool)
    if not isinstance(version, str):
        raise RuntimeError(f"no version pinned for {tool} in {manifest}")
    return version


def compare(left: Path, right: Path, expected_cohort: list[dict[str, Any]]) -> None:
    first = json.loads(left.read_text(encoding="utf-8"))
    second = json.loads(right.read_text(encoding="utf-8"))
    validate_archive_evidence(first)
    validate_archive_evidence(second)
    if first["cohort"] != expected_cohort or second["cohort"] != expected_cohort:
        raise RuntimeError("release evidence cohort differs from the release plan")
    if not first["artifacts"] or not second["artifacts"] or first != second:
        raise RuntimeError(
            "independent release artifacts differ: "
            + json.dumps({"first": first, "second": second}, sort_keys=True)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)
    plan_parser = subcommands.add_parser("plan")
    plan_parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    plan_parser.add_argument("--github-output", type=Path)
    plan_parser.add_argument("--registry-reader-root", required=True, type=Path)
    record_parser = subcommands.add_parser("record")
    record_parser.add_argument("--output", required=True, type=Path)
    record_parser.add_argument("--context", required=True)
    record_parser.add_argument("--cohort", required=True)
    record_parser.add_argument("paths", nargs="+", type=Path)
    tool_parser = subcommands.add_parser("tool-version")
    tool_parser.add_argument("tool")
    tool_parser.add_argument("--manifest", type=Path, default=Path(".config/release-tools.toml"))
    sbom_parser = subcommands.add_parser("validate-sbom")
    sbom_parser.add_argument("--cohort", required=True)
    sbom_parser.add_argument("--metadata", type=Path)
    sbom_parser.add_argument("paths", nargs="+", type=Path)
    compare_parser = subcommands.add_parser("compare")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--expected-cohort", required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            reader = load_registry_reader(args.registry_reader_root)
            result = plan(args.manifest, reader.sparse_index_versions)
            encoded = ",".join(result["packages"])
            print(encoded)
            if args.github_output:
                bootstrap = ",".join(result["bootstrap"])
                cohort = json.dumps(result["cohort"], separators=(",", ":"))
                args.github_output.write_text(
                    f"version={result['version']}\npackages={encoded}\nbootstrap={bootstrap}\ncohort={cohort}\n", encoding="utf-8"
                )
        elif args.command == "record":
            record(args.paths, args.output, json.loads(args.context), json.loads(args.cohort))
        elif args.command == "tool-version":
            print(tool_version(args.manifest, args.tool))
        elif args.command == "validate-sbom":
            if args.metadata is None:
                raise RuntimeError("SBOM validation requires locked cargo metadata")
            release_sbom.validate(args.paths, json.loads(args.cohort), release_sbom.Graph(args.metadata))
        else:
            compare(args.left, args.right, json.loads(args.expected_cohort))
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
