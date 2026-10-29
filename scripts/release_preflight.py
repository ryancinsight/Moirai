#!/usr/bin/env python3
"""Plan a registry release and compare independently produced release evidence.

The release workflow invokes this from a clean checkout.  It deliberately
queries the sparse index instead of inferring publication from Git tags: an
absent crate is recorded as bootstrap work while remaining in the checked
release cohort, so package, minimum-version, SBOM, and reproducibility gates
cover first publications before the authorised registry upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


USER_AGENT = "moirai-release-preflight"


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


def index_path(name: str) -> str:
    lowered = name.lower()
    if len(lowered) <= 2:
        return f"{len(lowered)}/{lowered}"
    if len(lowered) == 3:
        return f"3/{lowered[0]}/{lowered}"
    return f"{lowered[:2]}/{lowered[2:4]}/{lowered}"


def published_versions(name: str) -> list[str] | None:
    request = urllib.request.Request(
        f"https://index.crates.io/{index_path(name)}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"registry index returned HTTP {response.status} for {name}")
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"registry index returned HTTP {error.code} for {name}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"registry index request failed for {name}: {error}") from error
    return [json.loads(line)["vers"] for line in body.splitlines() if line]


def is_publishable(package: dict[str, Any]) -> bool:
    registries = package.get("publish")
    return registries is None or (isinstance(registries, list) and "crates-io" in registries)


def cohort_entry(package: dict[str, Any]) -> dict[str, Any]:
    libraries = [target["name"] for target in package["targets"] if {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"} & set(target["kind"])]
    return {"name": package["name"], "version": package["version"], "libraries": sorted(libraries)}


def select_pending(packages: list[dict[str, Any]], versions_for: Any) -> dict[str, Any]:
    release = [package for package in packages if is_publishable(package)]
    first: list[str] = []
    pending: list[dict[str, Any]] = []
    for package in release:
        versions = versions_for(package["name"])
        if versions is None:
            first.append(package["name"])
        if versions is None or package["version"] not in versions:
            pending.append(cohort_entry(package))
    return {"packages": [item["name"] for item in sorted(pending, key=lambda item: item["name"])], "bootstrap": sorted(first), "cohort": sorted(pending, key=lambda item: item["name"])}


def plan(manifest: Path) -> dict[str, Any]:
    return select_pending(cargo_metadata(manifest)["packages"], published_versions)


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
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not isinstance(item.get("version"), str)
            or not item["version"]
            or not isinstance(item.get("libraries"), list)
            or any(not isinstance(library, str) or not library for library in item["libraries"])
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


def sbom_components(value: dict[str, Any]) -> set[tuple[str, str]]:
    components: set[tuple[str, str]] = set()

    def collect(component: Any) -> None:
        if not isinstance(component, dict):
            return
        name, version = component.get("name"), component.get("version")
        if isinstance(name, str) and isinstance(version, str):
            components.add((name, version))
        for child in component.get("components", []):
            collect(child)

    collect(value.get("metadata", {}).get("component"))
    for component in value.get("components", []):
        collect(component)
    return components


def locked_dependency_closure(metadata: Path, cohort: list[dict[str, Any]]) -> set[tuple[str, str]]:
    value = json.loads(metadata.read_text(encoding="utf-8"))
    packages = value.get("packages")
    resolve = value.get("resolve")
    if not isinstance(packages, list) or not isinstance(resolve, dict) or not isinstance(resolve.get("nodes"), list):
        raise RuntimeError("locked cargo metadata lacks a resolved package graph")
    identities: dict[str, tuple[str, str]] = {}
    ids_by_identity: dict[tuple[str, str], str] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise RuntimeError("locked cargo metadata has a malformed package")
        package_id, name, version = package.get("id"), package.get("name"), package.get("version")
        if not isinstance(package_id, str) or not package_id or not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise RuntimeError("locked cargo metadata has a malformed package identity")
        if package_id in identities or (name, version) in ids_by_identity:
            raise RuntimeError("locked cargo metadata has an ambiguous package identity")
        identities[package_id] = (name, version)
        ids_by_identity[(name, version)] = package_id
    roots = {(item["name"], item["version"]) for item in cohort}
    missing_roots = roots - set(ids_by_identity)
    if missing_roots:
        raise RuntimeError("locked cargo metadata omits planned packages")
    dependencies: dict[str, list[str]] = {}
    for node in resolve["nodes"]:
        if not isinstance(node, dict):
            raise RuntimeError("locked cargo metadata has a malformed resolved node")
        package_id, direct = node.get("id"), node.get("dependencies")
        if not isinstance(package_id, str) or not package_id or not isinstance(direct, list):
            raise RuntimeError("locked cargo metadata has a malformed resolved node")
        if package_id not in identities or package_id in dependencies:
            raise RuntimeError("locked cargo metadata has an invalid resolved node")
        if any(not isinstance(dependency, str) or not dependency for dependency in direct):
            raise RuntimeError("locked cargo metadata has a malformed dependency")
        dependencies[package_id] = direct
    pending = [ids_by_identity[root] for root in roots]
    seen: set[str] = set()
    while pending:
        package_id = pending.pop()
        if package_id in seen:
            continue
        if package_id not in dependencies:
            raise RuntimeError(f"locked cargo metadata lacks a resolved node for {package_id}")
        seen.add(package_id)
        pending.extend(dependencies[package_id])
    closure = {identities[package_id] for package_id in seen}
    return closure


def validate_sbom(paths: list[Path], cohort: list[dict[str, Any]], metadata: Path | None = None) -> None:
    if not paths:
        raise RuntimeError("no CycloneDX documents generated")
    primary = set()
    components = set()
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("bomFormat") != "CycloneDX":
            raise RuntimeError(f"{path} is not a CycloneDX document")
        component = document.get("metadata", {}).get("component", {})
        name, version = component.get("name"), component.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeError(f"{path} lacks a primary component name/version")
        primary.add((name, version))
        components.update(sbom_components(document))
    missing = [f"{item['name']}@{item['version']}" for item in cohort if (item["name"], item["version"]) not in primary]
    if missing:
        raise RuntimeError("SBOM omits planned packages: " + ", ".join(missing))
    if metadata is not None:
        absent = sorted(locked_dependency_closure(metadata, cohort) - components)
        if absent:
            rendered = ", ".join(f"{name}@{version}" for name, version in absent)
            raise RuntimeError("SBOM omits locked dependency components: " + rendered)


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
            result = plan(args.manifest)
            encoded = ",".join(result["packages"])
            print(encoded)
            if args.github_output:
                bootstrap = ",".join(result["bootstrap"])
                cohort = json.dumps(result["cohort"], separators=(",", ":"))
                args.github_output.write_text(
                    f"packages={encoded}\nbootstrap={bootstrap}\ncohort={cohort}\n", encoding="utf-8"
                )
        elif args.command == "record":
            record(args.paths, args.output, json.loads(args.context), json.loads(args.cohort))
        elif args.command == "tool-version":
            print(tool_version(args.manifest, args.tool))
        elif args.command == "validate-sbom":
            validate_sbom(args.paths, json.loads(args.cohort), args.metadata)
        else:
            compare(args.left, args.right, json.loads(args.expected_cohort))
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
