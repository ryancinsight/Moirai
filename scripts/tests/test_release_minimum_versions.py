from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import release_minimum_versions as minimum


def registry_id(name: str, version: str) -> str:
    return f"registry+https://github.com/rust-lang/crates.io-index#{name}@{version}"


class SelectionTests(unittest.TestCase):
    def metadata(self) -> dict:
        workspace = "path+file:///candidate/member#0.1.0"
        direct_one = registry_id("demo", "1.2.3")
        direct_two = registry_id("demo", "2.3.4")
        transitive = registry_id("transitive", "9.0.0")
        git = "git+https://example.invalid/provider#provider@1.0.0"
        path = "path+file:///candidate/path-dependency#0.1.0"
        return {
            "workspace_members": [workspace],
            "packages": [
                {
                    "id": workspace,
                    "name": "member",
                    "dependencies": [
                        {
                            "name": "demo",
                            "rename": "first-demo",
                            "source": "registry+https://github.com/rust-lang/crates.io-index",
                            "req": "^1.2.3",
                            "kind": None,
                            "target": None,
                        },
                        {
                            "name": "demo",
                            "rename": "second-demo",
                            "source": "registry+https://github.com/rust-lang/crates.io-index",
                            "req": "^2.3.4",
                            "kind": "dev",
                            "target": "cfg(unix)",
                        },
                        {
                            "name": "provider",
                            "rename": None,
                            "source": "git+https://example.invalid/provider",
                            "req": "^1.0.0",
                            "kind": None,
                            "target": None,
                        },
                        {
                            "name": "path-dependency",
                            "rename": None,
                            "source": None,
                            "req": "*",
                            "kind": None,
                            "target": None,
                        },
                    ],
                },
                {
                    "id": direct_one,
                    "name": "demo",
                    "version": "1.2.3",
                    "source": "registry+https://github.com/rust-lang/crates.io-index",
                    "dependencies": [],
                },
                {
                    "id": direct_two,
                    "name": "demo",
                    "version": "2.3.4",
                    "source": "registry+https://github.com/rust-lang/crates.io-index",
                    "dependencies": [],
                },
                {
                    "id": transitive,
                    "name": "transitive",
                    "version": "9.0.0",
                    "source": "registry+https://github.com/rust-lang/crates.io-index",
                    "dependencies": [],
                },
                {
                    "id": git,
                    "name": "provider",
                    "version": "1.0.0",
                    "source": "git+https://example.invalid/provider",
                    "dependencies": [],
                },
                {
                    "id": path,
                    "name": "path-dependency",
                    "version": "0.1.0",
                    "source": None,
                    "dependencies": [],
                },
            ],
            "resolve": {
                "nodes": [
                    {
                        "id": workspace,
                        "deps": [
                            {
                                "name": "first_demo",
                                "pkg": direct_one,
                                "dep_kinds": [{"kind": None, "target": None}],
                            },
                            {
                                "name": "second_demo",
                                "pkg": direct_two,
                                "dep_kinds": [
                                    {"kind": "dev", "target": "cfg(unix)"}
                                ],
                            },
                            {
                                "name": "provider",
                                "pkg": git,
                                "dep_kinds": [{"kind": None, "target": None}],
                            },
                            {
                                "name": "path_dependency",
                                "pkg": path,
                                "dep_kinds": [{"kind": None, "target": None}],
                            },
                        ],
                    },
                    {
                        "id": direct_one,
                        "deps": [
                            {
                                "name": "transitive",
                                "pkg": transitive,
                                "dep_kinds": [{"kind": None, "target": None}],
                            }
                        ],
                    },
                ]
            },
        }

    def test_selects_direct_registry_aliases_at_distinct_versions(self) -> None:
        dependencies = minimum.direct_registry_dependencies(self.metadata())
        self.assertEqual(
            [(item["alias"], item["version"]) for item in dependencies],
            [("first_demo", "1.2.3"), ("second_demo", "2.3.4")],
        )

    def test_compatible_lock_advance_does_not_raise_declared_floor(self) -> None:
        locked = minimum.direct_registry_dependencies(self.metadata())
        selected = [dict(dependency) for dependency in locked]
        selected[0]["version"] = "1.2.3"
        selected[0]["package_id"] = registry_id("demo", "1.2.3")
        locked[0]["version"] = "1.2.9"
        locked[0]["package_id"] = registry_id("demo", "1.2.9")

        minimum.assert_direct_edges_preserved(locked, selected)
        minimum.assert_minimum_versions_at_declared_floors(selected)
        self.assertEqual(locked[0]["requirement"], "^1.2.3")
        self.assertEqual(selected[0]["version"], "1.2.3")

    def test_rejects_changed_direct_declaration(self) -> None:
        locked = minimum.direct_registry_dependencies(self.metadata())
        selected = [dict(dependency) for dependency in locked]
        selected[0]["requirement"] = "^1"
        with self.assertRaisesRegex(minimum.GateFailure, "changed the direct registry"):
            minimum.assert_direct_edges_preserved(locked, selected)

    def test_rejects_minimum_graph_above_declared_floor(self) -> None:
        selected = minimum.direct_registry_dependencies(self.metadata())
        selected[0]["version"] = "1.2.9"
        with self.assertRaisesRegex(minimum.GateFailure, "did not select declared floors"):
            minimum.assert_minimum_versions_at_declared_floors(selected)


class LockIdentityTests(unittest.TestCase):
    def lock(self, revision: str) -> bytes:
        return (
            "version = 4\n\n[[package]]\nname = \"provider\"\nversion = \"1.0.0\"\n"
            f"source = \"git+https://example.invalid/provider?rev={revision}#{revision}\"\n"
        ).encode()

    def test_rejects_changed_git_revision(self) -> None:
        with self.assertRaisesRegex(minimum.GateFailure, "changed Git package"):
            minimum.assert_git_sources_unchanged(self.lock("a" * 40), self.lock("b" * 40))


@unittest.skipUnless(
    os.environ.get("MOIRAI_RUN_CARGO_CONTROL") == "1",
    "set MOIRAI_RUN_CARGO_CONTROL=1 for the real Cargo resolver control",
)
class CargoControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = Path(__file__).resolve().parents[2]
        runner_root = Path(os.environ["ATLAS_PROCESS_TREE_ROOT"])
        sys.path.insert(0, str(runner_root))
        cls.process_tree = importlib.import_module("process_tree")
        cls.cargo_config = Path(os.environ["MOIRAI_CARGO_CONFIG"])

    def resolve(self, floor: str) -> tuple[int, bytes, str | None]:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            (candidate / "src").mkdir()
            (candidate / "src" / "lib.rs").write_text("pub fn value() {}\n", encoding="utf-8")
            (candidate / "Cargo.toml").write_text(
                "[package]\nname = \"minimum-control\"\nversion = \"0.1.0\"\n"
                "edition = \"2024\"\n\n[dependencies]\n"
                f"tracing = {{ version = \"{floor}\", default-features = false, features = [\"std\"] }}\n",
                encoding="utf-8",
            )
            shutil.copyfile(self.repository / "Cargo.lock", candidate / "Cargo.lock")
            command = [
                "cargo",
                f"+{minimum.NIGHTLY_TOOLCHAIN}",
                "--config",
                str(self.cargo_config),
                "update",
                "-Zdirect-minimal-versions",
                "--manifest-path",
                str(candidate / "Cargo.toml"),
                "-p",
                "tracing@0.1.44",
            ]
            result = self.process_tree.run(
                command,
                cwd=str(candidate),
                timeout=minimum.COMMAND_TIMEOUT_SECONDS,
            )
            output = (result.stdout or b"") + (result.stderr or b"")
            if result.returncode != 0:
                return result.returncode, output, None
            lock = tomllib.loads((candidate / "Cargo.lock").read_text(encoding="utf-8"))
            versions = [
                package["version"]
                for package in lock["package"]
                if package["name"] == "tracing"
            ]
            self.assertEqual(len(versions), 1)
            return result.returncode, output, versions[0]

    def test_declared_floor_controls_direct_minimum_selection(self) -> None:
        positive = self.resolve("0.1.44")
        self.assertEqual(positive[0], 0, positive[1])
        self.assertEqual(positive[2], "0.1.44")
        advanced_lock = self.resolve("0.1.40")
        self.assertEqual(advanced_lock[0], 0, advanced_lock[1])
        self.assertEqual(advanced_lock[2], "0.1.40")
        negative = self.resolve("0.1")
        self.assertEqual(negative[0], 101, negative[1])
        self.assertIn(b"versions that meet the requirements `^0.1` are: 0.1.0", negative[1])
        self.assertIn(b"`tracing` does not have that feature", negative[1])


if __name__ == "__main__":
    unittest.main()
