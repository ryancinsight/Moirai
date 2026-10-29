"""Value-semantic tests for release preflight planning and artifact comparison."""

import json
import io
import gzip
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_preflight


class ReleasePreflightTests(unittest.TestCase):
    @staticmethod
    def context():
        return {"candidate": "a", "toolchain": "1.97.0", "target": "linux", "profile": "release"}

    @staticmethod
    def cohort(name="demo", version="1.0.0"):
        return [{"name": name, "version": version, "libraries": []}]

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
            release_preflight.select_pending(packages, lambda _name: None),
            {"packages": ["root"], "bootstrap": ["root"], "cohort": [{"name": "root", "version": "1.0.0", "libraries": []}]},
        )

    def test_plan_selects_only_registry_missing_versions(self):
        packages = [
            {"name": "ready", "version": "1.0.0", "publish": None, "targets": []},
            {"name": "pending", "version": "1.1.0", "publish": ["crates-io"], "targets": []},
            {"name": "private", "version": "1.0.0", "publish": False, "targets": []},
        ]
        def versions(name):
            return ["1.0.0"] if name != "private" else None
        self.assertEqual(
            release_preflight.select_pending(packages, versions),
            {"packages": ["pending"], "bootstrap": [], "cohort": [{"name": "pending", "version": "1.1.0", "libraries": []}]},
        )

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

    def test_sbom_requires_each_cohort_member_as_primary_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom = root / "demo.cdx.json"
            bom.write_text(json.dumps({"bomFormat": "CycloneDX", "metadata": {"component": {"name": "demo", "version": "1.0.0"}}}))
            cohort = [{"name": "demo", "version": "1.0.0", "libraries": []}]
            release_preflight.validate_sbom([bom], cohort)
            with self.assertRaisesRegex(RuntimeError, "omits"):
                release_preflight.validate_sbom([bom], cohort + [{"name": "other", "version": "1.0.0", "libraries": []}])

    def test_compare_rejects_identical_missing_archive_against_external_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"context": self.context(), "cohort": self.cohort(), "artifacts": {"demo-1.0.0.crate": "0" * 64}}))
            with self.assertRaisesRegex(RuntimeError, "cohort differs"):
                release_preflight.compare(path, path, [{"name": "other", "version": "1.0.0", "libraries": []}])

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

    def test_sbom_requires_locked_dependency_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom = root / "demo.cdx.json"
            bom.write_text(json.dumps({"bomFormat": "CycloneDX", "metadata": {"component": {"name": "demo", "version": "1.0.0"}}, "components": [{"name": "dep", "version": "2.0.0"}]}))
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"packages": [{"id": "demo 1.0.0", "name": "demo", "version": "1.0.0"}, {"id": "dep 2.0.0", "name": "dep", "version": "2.0.0"}], "resolve": {"nodes": [{"id": "demo 1.0.0", "dependencies": ["dep 2.0.0"]}, {"id": "dep 2.0.0", "dependencies": []}]}}))
            release_preflight.validate_sbom([bom], self.cohort(), metadata)
            bom.write_text(json.dumps({"bomFormat": "CycloneDX", "metadata": {"component": {"name": "demo", "version": "1.0.0"}}}))
            with self.assertRaisesRegex(RuntimeError, "locked dependency"):
                release_preflight.validate_sbom([bom], self.cohort(), metadata)

    def test_locked_dependency_closure_rejects_missing_root_node(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.json"
            metadata.write_text(json.dumps({"packages": [{"id": "demo 1.0.0", "name": "demo", "version": "1.0.0"}], "resolve": {"nodes": []}}))
            with self.assertRaisesRegex(RuntimeError, "lacks a resolved node"):
                release_preflight.locked_dependency_closure(metadata, self.cohort())

    def test_locked_dependency_closure_rejects_missing_dependency_node(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.json"
            metadata.write_text(json.dumps({"packages": [{"id": "demo 1.0.0", "name": "demo", "version": "1.0.0"}, {"id": "dep 2.0.0", "name": "dep", "version": "2.0.0"}], "resolve": {"nodes": [{"id": "demo 1.0.0", "dependencies": ["dep 2.0.0"]}]}}))
            with self.assertRaisesRegex(RuntimeError, "lacks a resolved node"):
                release_preflight.locked_dependency_closure(metadata, self.cohort())

    def test_locked_dependency_closure_rejects_malformed_node(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.json"
            metadata.write_text(json.dumps({"packages": [{"id": "demo 1.0.0", "name": "demo", "version": "1.0.0"}], "resolve": {"nodes": [{"id": "demo 1.0.0", "dependencies": [1]}]}}))
            with self.assertRaisesRegex(RuntimeError, "malformed dependency"):
                release_preflight.locked_dependency_closure(metadata, self.cohort())


if __name__ == "__main__":
    unittest.main()
