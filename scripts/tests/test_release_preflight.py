"""Value-semantic tests for release preflight planning and artifact comparison."""

import json
import io
import gzip
import hashlib
import importlib.util
import os
import py_compile
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_preflight


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
            {"version": "1.0.0", "packages": ["root"], "bootstrap": ["root"], "pending": ["root"], "cohort": [{"name": "root", "version": "1.0.0"}]},
        )

    def test_plan_preserves_indexed_versions_in_the_checked_cohort(self):
        packages = [
            {"name": "ready", "version": "1.0.0", "publish": None, "targets": []},
            {"name": "pending", "version": "1.0.0", "publish": ["crates-io"], "targets": []},
            {"name": "private", "version": "1.0.0", "publish": False, "targets": []},
        ]
        def versions(name):
            return {"1.0.0" if name == "ready" else "0.9.0": "0" * 64}
        self.assertEqual(
            release_preflight.select_cohort(packages, versions),
            {"version": "1.0.0", "packages": ["pending", "ready"], "bootstrap": [], "pending": ["pending"], "cohort": [{"name": "pending", "version": "1.0.0"}, {"name": "ready", "version": "1.0.0"}]},
        )

    def test_workspace_unit_rejects_missing_or_conflicting_members_before_registry_reads(self):
        cases = [
            ([], "no publishable members"),
            ([{"name": "private", "version": "1.0.0", "publish": []}], "no publishable members"),
            ([{"name": "root", "version": "1.0.0"}, {"name": "leaf", "version": "1.1.0"}], "different versions"),
            ([{"name": "root", "version": "1.0.0"}] * 2, "duplicate package names"),
        ]
        for packages, message in cases:
            with self.subTest(message=message):
                queries = []
                with self.assertRaisesRegex(RuntimeError, message):
                    release_preflight.select_cohort(packages, queries.append)
                self.assertEqual(queries, [])

    def test_native_workspace_plan_fixes_membership_and_one_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "Cargo.toml"
            manifest.write_text(
                '[workspace]\nmembers = ["leaf", "root", "private"]\nresolver = "3"\n'
                '[workspace.package]\nversion = "1.0.0"\nedition = "2024"\n',
                encoding="utf-8",
            )
            for name in ("leaf", "root", "private"):
                member = root / name
                (member / "src").mkdir(parents=True)
                (member / "src/lib.rs").write_text("", encoding="utf-8")
                (member / "Cargo.toml").write_text(
                    f'[package]\nname = "{name}"\nversion.workspace = true\nedition.workspace = true\n'
                    + ("publish = false\n" if name == "private" else ""),
                    encoding="utf-8",
                )
            (root / "Cargo.lock").write_text(
                "version = 4\n" + "".join(
                    f'\n[[package]]\nname = "{name}"\nversion = "1.0.0"\n'
                    for name in ("leaf", "private", "root")
                ), encoding="utf-8",
            )
            queries = []
            registry = {"leaf": {"1.0.0": "0" * 64}, "root": None}

            def versions(name):
                queries.append(name)
                return registry[name]

            self.assertEqual(
                release_preflight.plan(manifest, versions),
                {"version": "1.0.0", "packages": ["leaf", "root"],
                 "bootstrap": ["root"], "pending": ["root"],
                 "cohort": [{"name": "leaf", "version": "1.0.0"},
                            {"name": "root", "version": "1.0.0"}]},
            )
            self.assertEqual(queries, ["leaf", "root"])
            member = root / "leaf/Cargo.toml"
            member.write_text(
                member.read_text(encoding="utf-8").replace('version.workspace = true', 'version = "1.1.0"'),
                encoding="utf-8",
            )
            lock = root / "Cargo.lock"
            lock.write_text(
                lock.read_text(encoding="utf-8").replace('name = "leaf"\nversion = "1.0.0"', 'name = "leaf"\nversion = "1.1.0"'),
                encoding="utf-8",
            )
            queries.clear()
            with self.assertRaisesRegex(RuntimeError, "different versions"):
                release_preflight.plan(manifest, versions)
            self.assertEqual(queries, [])

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
                {"version": "1.0.0", "packages": ["root"], "bootstrap": [], "pending": [], "cohort": [{"name": "root", "version": "1.0.0"}]},
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
                self.assertEqual(result["version"], "1.0.0")
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

    def test_contract_owner_executes_source_despite_valid_stale_bytecode(self):
        prefix = "import hashlib\ndef digest(value):\n    return hashlib.sha256("
        source_bytes = (prefix + "value" + ").hexdigest()\n").encode()
        stale_bytes = (prefix + "value[::-1]" + ").hexdigest()\n").encode()
        source_bytes = source_bytes.ljust(len(stale_bytes), b" ")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "owner.py"
            source.write_bytes(stale_bytes)
            timestamp = source.stat().st_mtime_ns
            bytecode = Path(py_compile.compile(str(source), doraise=True))
            source.write_bytes(source_bytes)
            os.utime(source, ns=(timestamp, timestamp))
            control_spec = importlib.util.spec_from_file_location("stale_owner_control", source)
            control = importlib.util.module_from_spec(control_spec)
            control_spec.loader.exec_module(control)
            payload = b"release-candidate"
            self.assertEqual(control.digest(payload), hashlib.sha256(payload[::-1]).hexdigest())
            module_name = "release_preflight_source_owner"
            try:
                owner = release_preflight.load_atlas_module(root, Path("owner.py"), module_name, "digest")
                self.assertEqual(owner.digest(payload), hashlib.sha256(payload).hexdigest())
                self.assertEqual(Path(owner.__file__).resolve(), source.resolve())
                self.assertEqual(source.read_bytes(), source_bytes)
                self.assertEqual(bytecode.read_bytes()[:4], importlib.util.MAGIC_NUMBER)
            finally:
                sys.modules.pop(module_name, None)

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


if __name__ == "__main__":
    unittest.main()
