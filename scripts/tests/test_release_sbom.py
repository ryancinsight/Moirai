"""CycloneDX graph and publication-cohort regression cases."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_sbom

COHORT = [{"name": "demo", "version": "1.0.0"}]


class ReleaseSbomTests(unittest.TestCase):
    def test_provenance_uses_postprocessor_byte_hash(self):
        workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release-preflight.yml").read_text(encoding="utf-8")
        self.assertIn('POSTPROCESSOR_SHA256="$(python3 scripts/release_sbom.py source-hash)"\n          export POSTPROCESSOR_SHA256', workflow)
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


    def test_sbom_extension_adds_only_locked_development_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom = root / "demo.cdx.json"
            metadata = root / "metadata.json"
            self.graph_metadata(metadata)
            self.producer_bom(bom)
            graph = release_sbom.Graph(metadata)
            release_sbom.extend(bom, graph)
            release_sbom.validate([bom], COHORT, graph)
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
                release_sbom.validate([bom], COHORT, graph)


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
                release_sbom.validate([bom], COHORT, graph)
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
                release_sbom.validate([bom, duplicate], COHORT, graph)


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
            release_sbom.validate([bom], COHORT, graph)


    def test_stage_selects_only_planned_primaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, planned, private = root / "metadata.json", root / "planned.cdx.json", root / "private.cdx.json"
            self.graph_metadata(metadata)
            self.producer_bom(planned)
            private.write_text(json.dumps({"bomFormat": "CycloneDX", "metadata": {"component": {"bom-ref": "unused", "name": "unused", "version": "1.0.0"}}, "components": [], "dependencies": []}))
            output, manifest = root / "output", root / "documents.json"
            release_sbom.stage([planned, private], output, manifest, COHORT, release_sbom.Graph(metadata))
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
