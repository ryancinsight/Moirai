#!/usr/bin/env python3
"""Extend cargo-cyclonedx output with locked development dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


class Graph:
    def __init__(self, metadata: Path) -> None:
        value = read_json(metadata)
        packages, resolve = value.get("packages"), value.get("resolve")
        if not isinstance(packages, list) or not isinstance(resolve, dict) or not isinstance(resolve.get("nodes"), list):
            raise RuntimeError("locked cargo metadata lacks a resolved package graph")
        self.packages: dict[str, dict[str, Any]] = {}
        self.identities: dict[tuple[str, str], str] = {}
        for package in packages:
            if not isinstance(package, dict):
                raise RuntimeError("locked cargo metadata has a malformed package")
            package_id, name, version = package.get("id"), package.get("name"), package.get("version")
            if not all(isinstance(value, str) and value for value in (package_id, name, version)):
                raise RuntimeError("locked cargo metadata has a malformed package identity")
            if package_id in self.packages or (name, version) in self.identities:
                raise RuntimeError("locked cargo metadata has an ambiguous package identity")
            self.packages[package_id] = package
            self.identities[(name, version)] = package_id
        self.edges: dict[str, list[tuple[str, set[str]]]] = {}
        for node in resolve["nodes"]:
            if not isinstance(node, dict) or not isinstance(node.get("id"), str):
                raise RuntimeError("locked cargo metadata has a malformed resolved node")
            package_id = node["id"]
            if package_id not in self.packages or package_id in self.edges:
                raise RuntimeError("locked cargo metadata has an invalid resolved node")
            deps = node.get("deps")
            if not isinstance(deps, list):
                raise RuntimeError("locked cargo metadata lacks typed dependency edges")
            parsed: list[tuple[str, set[str]]] = []
            for dependency in deps:
                if not isinstance(dependency, dict) or not isinstance(dependency.get("pkg"), str) or dependency["pkg"] not in self.packages:
                    raise RuntimeError("locked cargo metadata has a malformed dependency")
                kinds = dependency.get("dep_kinds")
                if not isinstance(kinds, list) or not kinds:
                    raise RuntimeError("locked cargo metadata has a malformed dependency kind")
                kinds_seen: set[str] = set()
                for kind in kinds:
                    if not isinstance(kind, dict) or kind.get("kind") not in (None, "build", "dev"):
                        raise RuntimeError("locked cargo metadata has a malformed dependency kind")
                    kinds_seen.add("normal" if kind["kind"] is None else kind["kind"])
                parsed.append((dependency["pkg"], kinds_seen))
            self.edges[package_id] = parsed

    def root(self, name: str, version: str) -> str:
        try:
            package_id = self.identities[(name, version)]
        except KeyError as error:
            raise RuntimeError(f"locked cargo metadata omits planned package {name}@{version}") from error
        if package_id not in self.edges:
            raise RuntimeError(f"locked cargo metadata lacks a resolved node for {package_id}")
        return package_id

    def closure(self, root: str, allowed: set[str]) -> tuple[set[str], set[tuple[str, str]]]:
        pending = [root]
        nodes: set[str] = set()
        edges: set[tuple[str, str]] = set()
        while pending:
            package_id = pending.pop()
            if package_id in nodes:
                continue
            if package_id not in self.edges:
                raise RuntimeError(f"locked cargo metadata lacks a resolved node for {package_id}")
            nodes.add(package_id)
            for dependency, kinds in self.edges[package_id]:
                if kinds & allowed:
                    edges.add((package_id, dependency))
                    pending.append(dependency)
        return nodes, edges

    def component(self, package_id: str) -> dict[str, Any]:
        package = self.packages[package_id]
        name, version = package["name"], package["version"]
        source = package.get("source")
        purl = f"pkg:cargo/{quote(name, safe='')}@{quote(version, safe='.+-') }"
        if isinstance(source, str) and source:
            if source.startswith("git+"):
                base, _, fragment = source.partition("#")
                base = base.split("?", 1)[0]
                value = f"{base}@{fragment}" if fragment else base
                purl += "?vcs_url=" + quote(value, safe="")
            elif not source.startswith("registry+https://github.com/rust-lang/crates.io-index") and not source.startswith("sparse+https://index.crates.io/"):
                _, _, registry = source.partition("+")
                purl += "?repository_url=" + quote(registry, safe="")
        return {"type": "library", "bom-ref": purl, "name": name, "version": version, "purl": purl}


def document_components(document: dict[str, Any]) -> list[dict[str, Any]]:
    components = document.get("components", [])
    if not isinstance(components, list) or any(not isinstance(component, dict) for component in components):
        raise RuntimeError("CycloneDX document has malformed components")
    return components


def all_components(document: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []

    def collect(component: dict[str, Any]) -> None:
        values.append(component)
        children = component.get("components", [])
        if not isinstance(children, list) or any(not isinstance(child, dict) for child in children):
            raise RuntimeError("CycloneDX document has malformed components")
        for child in children:
            collect(child)

    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict) or not isinstance(metadata.get("component"), dict):
        raise RuntimeError("CycloneDX document lacks a primary component")
    collect(metadata["component"])
    for component in document_components(document):
        collect(component)
    return values


def component_ref(component: dict[str, Any]) -> str:
    reference = component.get("bom-ref", component.get("purl"))
    if not isinstance(reference, str) or not reference:
        raise RuntimeError("CycloneDX component lacks a bom-ref or purl")
    return reference


def component_index(document: dict[str, Any]) -> dict[tuple[str, str], str]:
    values: dict[tuple[str, str], str] = {}
    references: dict[str, tuple[str, str]] = {}
    for component in all_components(document):
        reference = component_ref(component)
        if reference in references:
            raise RuntimeError("CycloneDX document reuses a bom-ref")
        name, version = component.get("name"), component.get("version")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise RuntimeError("CycloneDX component lacks a name/version")
        references[reference] = (name, version)
    metadata = document["metadata"]
    for component in [metadata["component"], *document_components(document)]:
        name, version = component["name"], component["version"]
        identity = (name, version)
        reference = component_ref(component)
        if identity in values and values[identity] != reference:
            raise RuntimeError("CycloneDX document has ambiguous component identity")
        values[identity] = reference
    return values


def root_for(document: dict[str, Any], graph: Graph) -> str:
    metadata = document["metadata"]
    component = metadata["component"]
    name, version = component.get("name"), component.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise RuntimeError("CycloneDX document lacks a primary component name/version")
    return graph.root(name, version)


def dependency_index(document: dict[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    dependencies = document.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise RuntimeError("CycloneDX document has malformed dependency edges")
    for dependency in dependencies:
        if not isinstance(dependency, dict) or not isinstance(dependency.get("ref"), str) or not isinstance(dependency.get("dependsOn", []), list):
            raise RuntimeError("CycloneDX document has malformed dependency edges")
        refs = dependency.get("dependsOn", [])
        if any(not isinstance(reference, str) or not reference for reference in refs):
            raise RuntimeError("CycloneDX document has malformed dependency edges")
        values.setdefault(dependency["ref"], set()).update(refs)
    return values


def assert_known_references(dependencies: dict[str, set[str]], references: set[str]) -> None:
    dangling = set(dependencies) - references
    dangling.update(reference for children in dependencies.values() for reference in children if reference not in references)
    if dangling:
        raise RuntimeError("CycloneDX document has dangling dependency references")


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def annotate_postprocessor(document: dict[str, Any]) -> None:
    metadata = document["metadata"]
    if not isinstance(metadata, dict):
        raise RuntimeError("CycloneDX document has malformed metadata")
    digest = source_hash()
    tool = {"vendor": "Moirai", "name": "scripts/release_sbom.py", "version": digest, "hashes": [{"alg": "SHA-256", "content": digest}]}
    tools = metadata.setdefault("tools", [])
    if isinstance(tools, list):
        tools.append(tool)
    elif isinstance(tools, dict) and isinstance(tools.get("components"), list):
        tools["components"].append({"type": "application", "name": tool["name"], "version": tool["version"], "hashes": tool["hashes"]})
    else:
        raise RuntimeError("CycloneDX document has unsupported metadata.tools")


def producer_coverage(document: dict[str, Any], graph: Graph, root: str) -> tuple[set[str], set[tuple[str, str]]]:
    nodes, edges = graph.closure(root, {"normal", "build"})
    components = component_index(document)
    absent = [(graph.packages[node]["name"], graph.packages[node]["version"]) for node in nodes if (graph.packages[node]["name"], graph.packages[node]["version"]) not in components]
    if absent:
        raise RuntimeError("producer SBOM omits normal/build components: " + ", ".join(f"{name}@{version}" for name, version in sorted(absent)))
    refs = {node: components[(graph.packages[node]["name"], graph.packages[node]["version"])] for node in nodes}
    dependencies = dependency_index(document)
    assert_known_references(dependencies, {component_ref(component) for component in all_components(document)})
    missing_edges = [(refs[parent], refs[child]) for parent, child in edges if refs[child] not in dependencies.get(refs[parent], set())]
    if missing_edges:
        raise RuntimeError("producer SBOM omits normal/build dependency edges")
    return nodes, edges


def extend(path: Path, graph: Graph) -> None:
    document = read_json(path)
    if document.get("bomFormat") != "CycloneDX":
        raise RuntimeError(f"{path} is not a CycloneDX document")
    root = root_for(document, graph)
    producer_coverage(document, graph, root)
    all_nodes, all_edges = graph.closure(root, {"normal", "build", "dev"})
    components = component_index(document)
    output_components = document_components(document)
    refs: dict[str, str] = {}
    for node in all_nodes:
        package = graph.packages[node]
        identity = (package["name"], package["version"])
        if identity not in components:
            injected = graph.component(node)
            output_components.append(injected)
            components[identity] = injected["bom-ref"]
        refs[node] = components[identity]
    document["components"] = output_components
    dependencies = dependency_index(document)
    for parent, child in all_edges:
        dependencies.setdefault(refs[parent], set()).add(refs[child])
    document["dependencies"] = [{"ref": reference, "dependsOn": sorted(children)} for reference, children in sorted(dependencies.items())]
    annotate_postprocessor(document)
    document.setdefault("properties", []).append({"name": "org.moirai.release-preflight:development-closure", "value": "cargo-metadata --locked --all-features"})
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def validate(paths: list[Path], cohort: list[dict[str, Any]], graph: Graph) -> None:
    expected = {(item["name"], item["version"]) for item in cohort}
    seen: set[tuple[str, str]] = set()
    for path in paths:
        document = read_json(path)
        if document.get("bomFormat") != "CycloneDX":
            raise RuntimeError(f"{path} is not a CycloneDX document")
        root = root_for(document, graph)
        package = graph.packages[root]
        identity = (package["name"], package["version"])
        if identity in seen:
            raise RuntimeError("SBOM has duplicate primary package identity")
        seen.add(identity)
        nodes, edges = graph.closure(root, {"normal", "build", "dev"})
        refs = component_index(document)
        absent = [(graph.packages[node]["name"], graph.packages[node]["version"]) for node in nodes if (graph.packages[node]["name"], graph.packages[node]["version"]) not in refs]
        if absent:
            raise RuntimeError("SBOM omits locked dependency components")
        mapped = {node: refs[(graph.packages[node]["name"], graph.packages[node]["version"])] for node in nodes}
        dependencies = dependency_index(document)
        assert_known_references(dependencies, {component_ref(component) for component in all_components(document)})
        if any(mapped[child] not in dependencies.get(mapped[parent], set()) for parent, child in edges):
            raise RuntimeError("SBOM omits locked dependency edges")
    if seen != expected:
        raise RuntimeError("SBOM omits planned packages")


def stage(paths: list[Path], output: Path, manifest: Path, cohort: list[dict[str, Any]], graph: Graph) -> None:
    output.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    expected = {(item["name"], item["version"]) for item in cohort}
    selected: set[tuple[str, str]] = set()
    for path in paths:
        document = read_json(path)
        if document.get("bomFormat") != "CycloneDX":
            raise RuntimeError(f"{path} is not a CycloneDX document")
        root = root_for(document, graph)
        package = graph.packages[root]
        name, version = package["name"], package["version"]
        if (name, version) not in expected:
            continue
        if (name, version) in selected:
            raise RuntimeError(f"duplicate staged SBOM primary {name}@{version}")
        destination = output / f"{name}-{version}.cdx.json"
        if destination.exists():
            raise RuntimeError(f"duplicate staged SBOM {destination.name}")
        shutil.move(path, destination)
        staged.append(str(destination))
        selected.add((name, version))
    if selected != expected:
        raise RuntimeError("producer SBOM omits planned primary packages")
    manifest.write_text(json.dumps(staged, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("--output", type=Path, required=True)
    stage_parser.add_argument("--manifest", type=Path, required=True)
    stage_parser.add_argument("--metadata", type=Path, required=True)
    stage_parser.add_argument("--cohort", required=True)
    stage_parser.add_argument("paths", nargs="+", type=Path)
    commands.add_parser("source-hash")
    for name in ("extend", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--metadata", type=Path, required=True)
        command.add_argument("--cohort", required=True)
        command.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "stage":
            stage(args.paths, args.output, args.manifest, json.loads(args.cohort), Graph(args.metadata))
        elif args.command == "source-hash":
            print(source_hash())
        else:
            graph = Graph(args.metadata)
            cohort = json.loads(args.cohort)
            if args.command == "extend":
                for path in args.paths:
                    extend(path, graph)
            else:
                validate(args.paths, cohort, graph)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
