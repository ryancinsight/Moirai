"""Value-semantic tests for release preflight planning and artifact comparison."""

import json
import io
import gzip
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_preflight
import release_sbom


class ReleasePreflightTests(unittest.TestCase):
    @staticmethod
    def registry_reader():
        root = os.environ.get("ATLAS_REGISTRY_READER_ROOT")
        if root is None:
            raise RuntimeError("ATLAS_REGISTRY_READER_ROOT must name the pinned Atlas checkout")
        return release_preflight.load_registry_reader(Path(root))

    @staticmethod
    def context():
        return {"candidate": "a", "toolchain": "1.97.0", "target": "linux", "profile": "release"}

    @staticmethod
    def cohort(name="demo", version="1.0.0"):
        return [{"name": name, "version": version}]

    @staticmethod
    def write_crate(path, name="demo", version="1.0.0", prefix=None, epoch=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = f'[package]\nname = "{name}"\nversion = "{version}"\n'.encode()
        with path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    member = tarfile.TarInfo(f"{prefix or f'{name}-{version}'}/Cargo.toml")
                    member.size = len(contents)
                    member.mtime = epoch
                    archive.addfile(member, io.BytesIO(contents))

    def test_plan_keeps_an_unpublished_workspace_crate_in_the_cohort(self):
        packages = [{"name": "root", "version": "1.0.0", "publish": None, "targets": []}]
        self.assertEqual(
            release_preflight.select_cohort(packages, lambda _name: None),
            {"packages": ["root"], "bootstrap": ["root"], "pending": ["root"], "cohort": [{"name": "root", "version": "1.0.0"}]},
        )

    def test_plan_preserves_indexed_versions_in_the_checked_cohort(self):
        packages = [
            {"name": "ready", "version": "1.0.0", "publish": None, "targets": []},
            {"name": "pending", "version": "1.1.0", "publish": ["crates-io"], "targets": []},
            {"name": "private", "version": "1.0.0", "publish": False, "targets": []},
        ]
        def versions(name):
            return {"1.0.0": "0" * 64} if name != "private" else None
        self.assertEqual(
            release_preflight.select_cohort(packages, versions),
            {"packages": ["pending", "ready"], "bootstrap": [], "pending": ["pending"], "cohort": [{"name": "pending", "version": "1.1.0"}, {"name": "ready", "version": "1.0.0"}]},
        )

    def test_plan_uses_the_shared_registry_reader_and_propagates_its_error(self):
        reader = self.registry_reader()
        checksum = "0" * 64

        def present(_url, _limit):
            record = json.dumps({"name": "root", "vers": "1.0.0", "cksum": checksum})
            return 200, record.encode("utf-8")

        def fail(_url, _limit):
            raise reader.IdentityError("shared registry failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "Cargo.toml"
            manifest.write_text('[package]\nname = "root"\nversion = "1.0.0"\nedition = "2024"\n')
            (root / "Cargo.lock").write_text(
                'version = 4\n\n[[package]]\nname = "root"\nversion = "1.0.0"\n'
            )
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text("")

            def versions(name):
                return reader.sparse_index_versions(name, fetch=present)

            self.assertEqual(
                release_preflight.plan(manifest, versions),
                {"packages": ["root"], "bootstrap": [], "pending": [], "cohort": [{"name": "root", "version": "1.0.0"}]},
            )

            def failing_versions(name):
                return reader.sparse_index_versions(name, fetch=fail)

            with self.assertRaisesRegex(reader.IdentityError, "shared registry failure"):
                release_preflight.plan(manifest, failing_versions)

    def test_upload_progress_cannot_change_the_checked_archive_set(self):
        packages = [
            {"name": "root", "version": "1.0.0", "publish": None},
            {"name": "leaf", "version": "1.0.0", "publish": ["crates-io"]},
            {"name": "private", "version": "1.0.0", "publish": []},
        ]
        cohort = [{"name": name, "version": "1.0.0"} for name in ("leaf", "root")]
        initial_evidence = None
        for published in (frozenset(), frozenset({"leaf"}), frozenset({"leaf", "root"})):
            with self.subTest(published=sorted(published)), tempfile.TemporaryDirectory() as directory:
                queries = []

                def versions(name):
                    queries.append(name)
                    return {"1.0.0": "0" * 64} if name in published else None

                result = release_preflight.select_cohort(list(reversed(packages)), versions)
                self.assertEqual(queries, ["leaf", "root"])
                self.assertEqual(result["packages"], ["leaf", "root"])
                self.assertEqual(result["cohort"], cohort)
                self.assertEqual(result["pending"], [name for name in queries if name not in published])
                self.assertEqual(result["bootstrap"], result["pending"])
                root = Path(directory)
                archives = [root / f"{name}-1.0.0.crate" for name in queries]
                for name, archive in zip(queries, archives):
                    self.write_crate(archive, name=name)
                receipt = root / "evidence.json"
                release_preflight.record(archives, receipt, self.context(), result["cohort"])
                current_evidence = json.loads(receipt.read_text(encoding="utf-8"))
                if initial_evidence is None:
                    initial_evidence = current_evidence
                else:
                    self.assertEqual(current_evidence, initial_evidence)
                self.assertEqual(
                    sorted(current_evidence["artifacts"]),
                    ["leaf-1.0.0.crate", "root-1.0.0.crate"],
                )

    def test_evidence_matches_the_shared_atlas_artifact_schema(self):
        root = Path(os.environ["ATLAS_REGISTRY_READER_ROOT"])
        owner = release_preflight.load_atlas_module(
            root,
            release_preflight.ARTIFACT_EVIDENCE,
            "atlas_artifact_evidence",
            "parse_expected_artifacts",
        )
        context = self.context() | {
            "candidate": "a" * 40,
            "profile": "cargo-package-verify",
        }
        value = release_preflight.evidence(
            context,
            {"demo-1.0.0.crate": "0" * 64},
            self.cohort(),
        )
        parsed = owner.parse_expected_artifacts(json.dumps(value))
        self.assertEqual(parsed.cohort, (("demo", "1.0.0"),))
        self.assertEqual(parsed.artifacts, {"demo-1.0.0.crate": "0" * 64})

    def test_record_and_compare_accept_complete_identical_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "demo-1.0.0.crate"
            second = root / "second" / "demo-1.0.0.crate"
            self.write_crate(first)
            self.write_crate(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            left = root / "left.json"
            right = root / "right.json"
            release_preflight.record([first], left, self.context(), self.cohort())
            release_preflight.record([second], right, self.context(), self.cohort())
            release_preflight.compare(left, right, self.cohort())

    def test_compare_rejects_controlled_archive_epoch_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first" / "demo-1.0.0.crate", root / "second" / "demo-1.0.0.crate"
            self.write_crate(first, epoch=1)
            self.write_crate(second, epoch=2)
            self.assertNotEqual(first.read_bytes(), second.read_bytes())
            left = root / "left.json"
            right = root / "right.json"
            release_preflight.record([first], left, self.context(), self.cohort())
            release_preflight.record([second], right, self.context(), self.cohort())
            with self.assertRaisesRegex(RuntimeError, "differ"):
                release_preflight.compare(left, right, self.cohort())

    def test_record_rejects_missing_artifacts_and_compare_rejects_empty_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self.context()
            cohort = self.cohort()
            with self.assertRaisesRegex(RuntimeError, "missing release evidence"):
                release_preflight.record([root / "missing.rlib"], root / "out.json", context, cohort)
            empty = root / "empty.json"
            empty.write_text(json.dumps({"context": context, "cohort": cohort, "artifacts": {}}))
            with self.assertRaisesRegex(RuntimeError, "no artifacts"):
                release_preflight.compare(empty, empty, cohort)

    def test_tool_version_reads_the_committed_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "release-tools.toml"
            manifest.write_text("[tools]\ncargo-cyclonedx = \"0.5.9\"\n")
            self.assertEqual(
                release_preflight.tool_version(manifest, "cargo-cyclonedx"), "0.5.9"
            )

    def test_compare_checkout_precedes_evidence_download(self):
        workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release-preflight.yml").read_text(encoding="utf-8")
        compare = workflow.split("  compare:\n", 1)[1].split("\n  sbom:\n", 1)[0]
        self.assertLess(
            compare.index("actions/checkout@"),
            compare.index("actions/download-artifact@"),
        )

    def test_provenance_uses_postprocessor_byte_hash(self):
        workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release-preflight.yml").read_text(encoding="utf-8")
        self.assertIn('export POSTPROCESSOR_SHA256="$(python3 scripts/release_sbom.py source-hash)"', workflow)
        self.assertNotIn("hashFiles('scripts/release_sbom.py')", workflow)
        self.assertEqual(release_sbom.source_hash(), __import__("hashlib").sha256((Path(__file__).resolve().parents[1] / "release_sbom.py").read_bytes()).hexdigest())

    @staticmethod
    def graph_metadata(root):
        value = {
            "packages": [
                {"id": "demo", "name": "demo", "version": "1.0.0", "source": None},
                {"id": "runtime", "name": "runtime", "version": "1.0.0", "source": "registry+https://github.com/rust-lang/crates.io-index"},
                {"id": "build", "name": "build", "version": "1.0.0", "source": "registry+https://github.com/rust-lang/crates.io-index"},
                {"id": "dev", "name": "dev", "version": "1.0.0", "source": "git+https://example.invalid/dev?rev=abc#deadbeef"},
                {"id": "dev-child", "name": "dev-child", "version": "1.0.0", "source": "registry+https://github.com/rust-lang/crates.io-index"},
                {"id": "unused", "name": "unused", "version": "1.0.0", "source": None},
            ],
            "resolve": {"nodes": [
                {"id": "demo", "deps": [
                    {"pkg": "runtime", "dep_kinds": [{"kind": None}]},
                    {"pkg": "build", "dep_kinds": [{"kind": "build"}]},
                    {"pkg": "dev", "dep_kinds": [{"kind": "dev"}]},
                ]},
                {"id": "runtime", "deps": []},
                {"id": "build", "deps": []},
                {"id": "dev", "deps": [{"pkg": "dev-child", "dep_kinds": [{"kind": None}, {"kind": "dev"}]}]},
                {"id": "dev-child", "deps": [{"pkg": "dev", "dep_kinds": [{"kind": "dev"}]}]},
                {"id": "unused", "deps": []},
            ]},
        }
        root.write_text(json.dumps(value))

    @staticmethod
    def producer_bom(root):
        value = {
            "bomFormat": "CycloneDX",
            "metadata": {"component": {"bom-ref": "demo", "name": "demo", "version": "1.0.0", "purl": "pkg:cargo/demo@1.0.0"}},
            "components": [
                {"bom-ref": "runtime", "name": "runtime", "version": "1.0.0", "purl": "pkg:cargo/runtime@1.0.0", "licenses": [{"license": {"id": "MIT"}}]},
                {"bom-ref": "build", "name": "build", "version": "1.0.0", "purl": "pkg:cargo/build@1.0.0"},
            ],
            "dependencies": [{"ref": "demo", "dependsOn": ["runtime", "build"]}],
        }
        root.write_text(json.dumps(value))

    def test_compare_rejects_identical_missing_archive_against_external_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"context": self.context(), "cohort": self.cohort(), "artifacts": {"demo-1.0.0.crate": "0" * 64}}))
            with self.assertRaisesRegex(RuntimeError, "cohort differs"):
                release_preflight.compare(path, path, [{"name": "other", "version": "1.0.0"}])

    def test_record_rejects_duplicate_archive_basenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one" / "demo-1.0.0.crate", root / "two" / "demo-1.0.0.crate"
            self.write_crate(first)
            self.write_crate(second)
            with self.assertRaisesRegex(RuntimeError, "duplicate basenames"):
                release_preflight.record([first, second], root / "evidence.json", self.context(), self.cohort())

    def test_record_rejects_filename_internal_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo-1.0.0.crate"
            self.write_crate(path, version="2.0.0", prefix="demo-1.0.0")
            with self.assertRaisesRegex(RuntimeError, "mismatched package identity"):
                release_preflight.record([path], Path(directory) / "evidence.json", self.context(), self.cohort())

    def test_evidence_rejects_malformed_context_cohort_and_hash(self):
        with self.assertRaisesRegex(RuntimeError, "malformed context"):
            release_preflight.evidence({"candidate": "a", "toolchain": 1, "target": "x", "profile": "p"}, {"demo-1.0.0.crate": "0" * 64}, self.cohort())
        with self.assertRaisesRegex(RuntimeError, "duplicate cohort"):
            release_preflight.evidence(self.context(), {"demo-1.0.0.crate": "0" * 64}, self.cohort() * 2)
        with self.assertRaisesRegex(RuntimeError, "invalid SHA"):
            release_preflight.evidence(self.context(), {"demo-1.0.0.crate": "not-a-hash"}, self.cohort())

    def test_compare_rejects_context_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "demo-1.0.0.crate"
            self.write_crate(archive)
            left, right = root / "left.json", root / "right.json"
            release_preflight.record([archive], left, self.context(), self.cohort())
            different = self.context() | {"candidate": "b"}
            release_preflight.record([archive], right, different, self.cohort())
            with self.assertRaisesRegex(RuntimeError, "differ"):
                release_preflight.compare(left, right, self.cohort())

    def test_sbom_extension_adds_only_locked_development_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom = root / "demo.cdx.json"
            metadata = root / "metadata.json"
            self.graph_metadata(metadata)
            self.producer_bom(bom)
            graph = release_sbom.Graph(metadata)
            release_sbom.extend(bom, graph)
            release_sbom.validate([bom], self.cohort(), graph)
            document = json.loads(bom.read_text())
            components = {(item["name"], item["version"]) for item in document["components"]}
            self.assertEqual(components, {("runtime", "1.0.0"), ("build", "1.0.0"), ("dev", "1.0.0"), ("dev-child", "1.0.0")})
            self.assertNotIn(("unused", "1.0.0"), components)
            self.assertEqual(document["components"][0]["licenses"], [{"license": {"id": "MIT"}}])
            injected = next(item for item in document["components"] if item["name"] == "dev")
            self.assertEqual(injected["bom-ref"], injected["purl"])
            self.assertTrue(injected["purl"].startswith("pkg:cargo/dev@1.0.0?vcs_url="))
            tool = document["metadata"]["tools"][-1]
            self.assertEqual(tool["name"], "scripts/release_sbom.py")
            self.assertEqual(tool["version"], release_sbom.source_hash())
            self.assertEqual(tool["hashes"], [{"alg": "SHA-256", "content": release_sbom.source_hash()}])
            document["components"] = [item for item in document["components"] if item["name"] != "dev-child"]
            bom.write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "dependency components"):
                release_sbom.validate([bom], self.cohort(), graph)

    def test_sbom_validator_rejects_removed_development_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom, metadata = root / "demo.cdx.json", root / "metadata.json"
            self.graph_metadata(metadata)
            self.producer_bom(bom)
            graph = release_sbom.Graph(metadata)
            release_sbom.extend(bom, graph)
            document = json.loads(bom.read_text())
            dev_ref = next(item["bom-ref"] for item in document["components"] if item["name"] == "dev")
            document["dependencies"] = [item for item in document["dependencies"] if item["ref"] != "demo"]
            document["dependencies"].append({"ref": "demo", "dependsOn": ["runtime", "build"]})
            bom.write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "dependency edges"):
                release_sbom.validate([bom], self.cohort(), graph)
            self.assertTrue(dev_ref)

    def test_sbom_validator_rejects_duplicate_primary_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom, duplicate, metadata = root / "demo.cdx.json", root / "duplicate.cdx.json", root / "metadata.json"
            self.graph_metadata(metadata)
            self.producer_bom(bom)
            graph = release_sbom.Graph(metadata)
            release_sbom.extend(bom, graph)
            duplicate.write_bytes(bom.read_bytes())
            with self.assertRaisesRegex(RuntimeError, "duplicate primary"):
                release_sbom.validate([bom, duplicate], self.cohort(), graph)

    def test_sbom_rejects_colliding_references_and_dangling_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom, metadata = root / "demo.cdx.json", root / "metadata.json"
            self.graph_metadata(metadata)
            self.producer_bom(bom)
            document = json.loads(bom.read_text())
            document["components"][1]["bom-ref"] = "runtime"
            bom.write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "reuses a bom-ref"):
                release_sbom.extend(bom, release_sbom.Graph(metadata))
            self.producer_bom(bom)
            document = json.loads(bom.read_text())
            document["dependencies"].append({"ref": "demo", "dependsOn": ["missing"]})
            bom.write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "dangling dependency"):
                release_sbom.extend(bom, release_sbom.Graph(metadata))

    def test_nested_target_keeps_distinct_component_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom, metadata = root / "demo.cdx.json", root / "metadata.json"
            self.graph_metadata(metadata)
            self.producer_bom(bom)
            document = json.loads(bom.read_text())
            document["metadata"]["component"]["components"] = [{"type": "library", "bom-ref": "demo-target", "name": "demo", "version": "1.0.0"}]
            bom.write_text(json.dumps(document))
            graph = release_sbom.Graph(metadata)
            release_sbom.extend(bom, graph)
            release_sbom.validate([bom], self.cohort(), graph)

    def test_stage_selects_only_planned_primaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, planned, private = root / "metadata.json", root / "planned.cdx.json", root / "private.cdx.json"
            self.graph_metadata(metadata)
            self.producer_bom(planned)
            private.write_text(json.dumps({"bomFormat": "CycloneDX", "metadata": {"component": {"bom-ref": "unused", "name": "unused", "version": "1.0.0"}}, "components": [], "dependencies": []}))
            output, manifest = root / "output", root / "documents.json"
            release_sbom.stage([planned, private], output, manifest, self.cohort(), release_sbom.Graph(metadata))
            paths = json.loads(manifest.read_text())
            self.assertEqual([Path(path).name for path in paths], ["demo-1.0.0.cdx.json"])
            self.assertTrue(private.exists())

    def test_sbom_extension_rejects_missing_producer_runtime_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom, metadata = root / "demo.cdx.json", root / "metadata.json"
            self.graph_metadata(metadata)
            self.producer_bom(bom)
            document = json.loads(bom.read_text())
            document["components"] = document["components"][1:]
            bom.write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "normal/build components"):
                release_sbom.extend(bom, release_sbom.Graph(metadata))

    def test_sbom_graph_rejects_malformed_or_missing_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.json"
            self.graph_metadata(metadata)
            value = json.loads(metadata.read_text())
            value["resolve"]["nodes"] = value["resolve"]["nodes"][:1]
            metadata.write_text(json.dumps(value))
            graph = release_sbom.Graph(metadata)
            with self.assertRaisesRegex(RuntimeError, "lacks a resolved node"):
                graph.closure("demo", {"normal", "build", "dev"})
            self.graph_metadata(metadata)
            value = json.loads(metadata.read_text())
            value["resolve"]["nodes"][0]["deps"][0]["dep_kinds"] = []
            metadata.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "malformed dependency kind"):
                release_sbom.Graph(metadata)


if __name__ == "__main__":
    unittest.main()
