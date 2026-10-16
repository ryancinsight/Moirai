"""Output-only policy/JSON/filesystem checks; no native event is simulated."""

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "webview/keyboard_evidence.py"
SPEC = importlib.util.spec_from_file_location("keyboard_evidence", SCRIPT)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


class KeyboardEvidenceTests(unittest.TestCase):
    def test_every_three_cell_report_pattern_has_the_specified_scope(self):
        expected = {
            (False, False, True): "BOUNDED_PROPERTY_SPECIFIC_DIFFERENCE",
            (False, True, True): "BOUNDED_SETTER_PATH_DIFFERENCE",
            (True, True, True): "NOT_REPRODUCED",
        }
        actual = {}
        for pattern in itertools.product((False, True), repeat=3):
            with self.subTest(pattern=pattern):
                actual[pattern] = evidence.classify(pattern)
                self.assertEqual(actual[pattern], expected.get(pattern, "INCONCLUSIVE"))
        self.assertEqual(sum(value == "INCONCLUSIVE" for value in actual.values()), 5)
        for invalid in ((0, False, True), (False, True), [False, False, True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evidence.classify(invalid)

    def test_actual_json_file_hash_and_original_duplicate_rejection(self):
        with tempfile.TemporaryDirectory(prefix="keyboard-evidence-患者-🦀-") as temporary:
            path = Path(temporary) / "actual metadata.json"
            raw = '{"schema":1,"text":"患者-🦀"}'.encode("utf-8")
            path.write_bytes(raw)
            value, identity = evidence.read_json(path)
            self.assertEqual(value, {"schema": 1, "text": "患者-🦀"})
            self.assertEqual(identity, {"path": str(path.resolve()), "bytes": len(raw),
                                       "sha256": hashlib.sha256(raw).hexdigest()})
            path.write_bytes(b'{"schema":1,"schema":1}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                evidence.read_json(path)
            with self.assertRaises(FileNotFoundError) as caught:
                evidence.read_json(Path(temporary) / "missing.json")
            self.assertEqual(caught.exception.errno, 2)

    def test_raw_document_byte_budget_is_inclusive_before_decode(self):
        # Deliberately invalid whitespace-only JSON tests byte admission, not
        # an invented Ready/Key/Closed event or a successful native cell.
        for size, expected in ((65_535, json.JSONDecodeError), (65_536, json.JSONDecodeError),
                               (65_537, ValueError)):
            raw = " " * size
            value = {"rows": [{}, {}], "originals": [raw, ""], "original_bytes": size}
            with self.subTest(size=size), self.assertRaises(expected) as caught:
                evidence.document(value)
            if size > 65_536:
                self.assertEqual(str(caught.exception), "actual SDK original bytes exceed aggregate cap")
        for invalid in ({}, {"rows": [], "originals": [], "original_bytes": 0},
                        {"rows": [{}, {}], "originals": [True, ""], "original_bytes": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evidence.document(invalid)

    def test_output_is_utf8_hashed_and_never_overwrites_an_existing_receipt(self):
        with tempfile.TemporaryDirectory(prefix="keyboard-result-患者-🦀-") as temporary:
            path = Path(temporary) / "comparison.json"
            # This is a publisher representation vector, not a native receipt.
            value = {"classification": evidence.classify((True, True, True)),
                     "label": "患者-🦀", "delivery_absence": None}
            identity = evidence.publish(value, path)
            raw = path.read_bytes()
            self.assertEqual(json.loads(raw), value)
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(identity, {"output": str(path.resolve()), "bytes": len(raw),
                                       "sha256": hashlib.sha256(raw).hexdigest(),
                                       "classification": "NOT_REPRODUCED"})
            with self.assertRaises(FileExistsError):
                evidence.publish({"classification": "INCONCLUSIVE"}, path)
            self.assertEqual(path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
