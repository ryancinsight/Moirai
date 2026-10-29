"""Failure classification for the controlled-profile gate."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_profiles import classify_failures


def record(name="delay-10ms-loss-1pct", streams=16, size=65560, error=None):
    result = {"profile": {"name": name}, "repetition": 0,
              "streams": streams, "bytes": size}
    if error is not None:
        result["error"] = error
    return result


TIMEOUT = "TimeoutExpired: Command 'quic_load' timed out after 59.98 seconds"


class FailureClassification(unittest.TestCase):
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
