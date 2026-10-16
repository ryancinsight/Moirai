"""Closed fixture selection preserves the original authored bridge payload."""

import hashlib
from html.parser import HTMLParser
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/webview/build_bridge.py"
sys.path.insert(0, str(SCRIPT.parent))
import build_bridge


class Markup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.script_depth = 0
        self.script_text = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))
        if tag == "script":
            self.script_depth += 1

    def handle_endtag(self, tag):
        if tag == "script":
            self.script_depth -= 1

    def handle_data(self, text):
        if self.script_depth:
            self.script_text.append(text)


class FixtureSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="moirai-payload-患者-🦀-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.required = ["--manifest-path", str(ROOT / "Cargo.toml"),
                         "--target-dir", str(self.root / "target"),
                         "--output", str(self.root / "output")]

    def materialize(self, name):
        fixture = build_bridge.FIXTURES[name]
        payload = self.root / name
        payload.mkdir()
        build_bridge.write_payload(ROOT, payload, fixture)
        self.assertEqual(sorted(p.name for p in payload.iterdir()),
                         ["bootstrap.js", "favicon.svg", "index.html"])
        self.assertEqual((payload / "favicon.svg").read_bytes(),
                         (ROOT / "moirai-pal/examples/webview_bridge_fixture/favicon.svg").read_bytes())
        return fixture, payload

    def test_default_bridge_keeps_original_static_bytes_and_names(self):
        args = build_bridge.arguments(self.required)
        self.assertEqual(args.fixture, "bridge")
        fixture, payload = self.materialize(args.fixture)
        # These hashes come from the unchanged baseline's two authored literals.
        expected = {
            "bootstrap.js": "09eb8ba91395fbd7627f7cb575dc109ad6d18a27427465319a248d61606515fd",
            "index.html": "7a94fe3272b87d4a63c194f35f7e6a3cad6c5b8c67e6f039cafc525a2de90fb5",
        }
        for name, digest in expected.items():
            original = (payload / name).read_bytes()
            self.assertEqual(hashlib.sha256(original.replace(b"\r\n", b"\n")).hexdigest(), digest)
        self.assertEqual((payload / "bootstrap.js").read_bytes(),
                         b"import init from './webview_bridge_fixture.js';\nawait init();\n"
                         .replace(b"\n", os.linesep.encode("ascii")))
        self.assertEqual(fixture.example, "webview_bridge_fixture")
        self.assertEqual(fixture.payload_names, ("bootstrap.js", "favicon.svg", "index.html",
                         "webview_bridge_fixture.js", "webview_bridge_fixture_bg.wasm"))
        self.assertEqual(fixture.kind, "actual-rust-native-bridge-payload")

    def test_keyboard_selection_has_a_real_input_and_no_inline_behavior(self):
        args = build_bridge.arguments([*self.required, "--fixture", "keyboard"])
        self.assertEqual(args.fixture, "keyboard")
        fixture, payload = self.materialize(args.fixture)
        markup = Markup()
        markup.feed((payload / "index.html").read_text(encoding="utf-8"))
        inputs = [attrs for tag, attrs in markup.elements if tag == "input"]
        self.assertEqual(inputs, [{"id": "keyboard-target", "autofocus": None}])
        self.assertEqual([attrs for tag, attrs in markup.elements if tag == "p"],
                         [{"id": "fixture-status"}])
        self.assertEqual([attrs for tag, attrs in markup.elements if tag == "script"],
                         [{"type": "module", "src": "./bootstrap.js"}])
        self.assertEqual(markup.script_text, [])
        self.assertFalse(any(name.startswith("on") for _, attrs in markup.elements for name in attrs))
        self.assertEqual((payload / "bootstrap.js").read_bytes(),
                         b"import init from './webview_keyboard_fixture.js';\nawait init();\n"
                         .replace(b"\n", os.linesep.encode("ascii")))
        self.assertEqual(fixture.payload_names, ("bootstrap.js", "favicon.svg", "index.html",
                         "webview_keyboard_fixture.js", "webview_keyboard_fixture_bg.wasm"))

    def test_unknown_fixture_is_rejected_by_the_real_cli_before_build(self):
        result = subprocess.run([sys.executable, str(SCRIPT), *self.required,
                                 "--fixture", "unowned"], capture_output=True, timeout=5, check=False)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(b"invalid choice", result.stderr)
        self.assertIn(b"bridge", result.stderr)
        self.assertIn(b"keyboard", result.stderr)
        self.assertFalse((self.root / "output").exists())

    def test_examples_share_the_original_compile_configuration(self):
        examples = tomllib.loads((ROOT / "moirai-pal/Cargo.toml").read_text(encoding="utf-8"))["example"]
        entries = {entry["name"]: {key: value for key, value in entry.items() if key != "name"}
                   for entry in examples}
        self.assertEqual(entries["webview_keyboard_fixture"], entries["webview_bridge_fixture"])
        self.assertEqual(entries["webview_keyboard_fixture"], {"crate-type": ["cdylib"]})
        for name, fixture in build_bridge.FIXTURES.items():
            with self.subTest(fixture=name):
                self.assertEqual(len(fixture.source_paths), 10)
                self.assertIn(f"moirai-pal/examples/{fixture.example}.rs", fixture.source_paths)
                self.assertIn(f"moirai-pal/examples/{fixture.example}/browser.rs", fixture.source_paths)
                self.assertIn(f"moirai-pal/examples/{fixture.example}/protocol.rs", fixture.source_paths)
                self.assertEqual(len(set(fixture.source_paths)), len(fixture.source_paths))


if __name__ == "__main__":
    unittest.main()
