"""Failure classification for the controlled-profile gate."""

from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_profiles import admitted_case, admitted_cases, classify_failures, main


def record(name="delay-10ms-loss-1pct", streams=16, size=65560, error=None):
    result = {"profile": {"name": name}, "repetition": 0,
              "streams": streams, "bytes": size}
    if error is not None:
        result["error"] = error
    return result


TIMEOUT = "TimeoutExpired: Command 'quic_load' timed out after 59.98 seconds"


class FailureClassification(unittest.TestCase):
    def test_admitted_case_selection_preserves_declared_domain(self):
        cases = admitted_cases(True)
        selected = admitted_case('delay-10ms-loss-1pct:16:65560', cases)
        self.assertEqual((selected[0].name, selected[1], selected[2]),
                         ('delay-10ms-loss-1pct', 16, 65_560))
        with self.assertRaisesRegex(ValueError, 'one member'):
            admitted_case('delay-10ms-loss-1pct:16:65559', cases)

    def test_invalid_case_fails_before_network_construction(self):
        arguments = ['quic_profiles.py', '--binary', 'missing', '--output', 'unused.json',
                     '--case', 'delay-10ms-loss-1pct:16:65559']
        with mock.patch.object(sys, 'argv', arguments), \
                mock.patch('quic_profiles.Network') as network, \
                self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)
        network.assert_not_called()

    def test_full_matrix_timeout_blocks(self):
        failure = record(error=TIMEOUT)
        self.assertEqual(classify_failures([failure, record()]), [failure])

    def test_diagnostic_case_timeout_blocks(self):
        failure = record(error=TIMEOUT)
        self.assertEqual(classify_failures([failure]), [failure])

    def test_every_error_value_and_case_blocks(self):
        failures = [
            record(error="CalledProcessError: exit 1"),
            record(name="delay-50ms", error=TIMEOUT),
            record(streams=4, error=TIMEOUT),
            record(size=4096, error=TIMEOUT),
        ]
        self.assertEqual(classify_failures(failures), failures)

    def test_successful_records_are_not_failures(self):
        self.assertEqual(classify_failures([record(), record(name="delay-50ms")]), [])


if __name__ == "__main__":
    unittest.main()
