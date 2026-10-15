"""The native loader must belong to the SDK that Cargo actually checks."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "webview" / "runtime_version.py"
SPEC = importlib.util.spec_from_file_location("webview_runtime_version", SCRIPT)
runtime_version = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_version)


class NativeSdkSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.manifest = root / "sdk" / "Cargo.toml"
        self.manifest.parent.mkdir()
        self.manifest.write_text(
            '[package]\nname = "webview2-com-sys"\nversion = "0.39.1"\n',
            encoding="utf-8",
        )
        self.native = root / "compiler-reported-native-directory"
        self.native.mkdir()
        # These are Cargo's public JSON records, with the package ID treated
        # as opaque. No SDK or runtime API is replaced by these parser inputs.
        self.records = [
            {
                "reason": "compiler-artifact",
                "package_id": "opaque-sdk-package-id",
                "manifest_path": str(self.manifest),
                "target": {"kind": ["lib"], "name": "webview2_com_sys"},
            },
            {
                "reason": "build-script-executed",
                "package_id": "opaque-sdk-package-id",
                "linked_paths": [f"native={self.native}"],
            },
            {"reason": "build-finished", "success": True},
        ]

    def test_native_build_identity_controls_loader_and_version(self):
        unrelated = {
            "reason": "build-script-executed",
            "package_id": "other-platform-package",
            "linked_paths": ["native=/unrelated/WebView2Loader.dll"],
        }
        path, version = runtime_version.native_sdk([unrelated, *self.records])
        self.assertEqual(path, self.native / "WebView2Loader.dll")
        self.assertEqual(version, "0.39.1")

    def test_missing_ambiguous_or_failed_native_build_is_rejected(self):
        cases = []
        for index in range(len(self.records)):
            records = copy.deepcopy(self.records)
            del records[index]
            cases.append(records)
        for index in (0, 1):
            records = copy.deepcopy(self.records)
            records.append(copy.deepcopy(records[index]))
            cases.append(records)
        records = copy.deepcopy(self.records)
        records[-1]["success"] = False
        cases.append(records)
        records = copy.deepcopy(self.records)
        records[1]["package_id"] = "different-sdk-package-id"
        cases.append(records)
        records = copy.deepcopy(self.records)
        records[1]["linked_paths"].append("native=/ambiguous-native-directory")
        cases.append(records)
        records = copy.deepcopy(self.records)
        records[0]["target"]["kind"] = ["custom-build"]
        cases.append(records)
        for records in cases:
            with self.subTest(records=records), self.assertRaises(RuntimeError):
                runtime_version.native_sdk(records)

    def test_manifest_identity_and_version_are_required(self):
        for package in (
            'name = "other-package"\nversion = "0.39.1"',
            'name = "webview2-com-sys"',
            'name = "webview2-com-sys"\nversion = ""',
        ):
            self.manifest.write_text(f"[package]\n{package}\n", encoding="utf-8")
            with self.subTest(package=package), self.assertRaises(RuntimeError):
                runtime_version.native_sdk(self.records)


if __name__ == "__main__":
    unittest.main()
