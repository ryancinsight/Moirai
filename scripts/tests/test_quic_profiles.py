"""Known deadline-limit classification for the controlled-profile gate."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_profiles import KNOWN_DEADLINE_LIMITS, classify_failures


def record(name="delay-10ms-loss-1pct", streams=16, size=65560, error=None):
    result = {"profile": {"name": name}, "repetition": 0,
              "streams": streams, "bytes": size}
    if error is not None:
        result["error"] = error
    return result


TIMEOUT = "TimeoutExpired: Command 'quic_load' timed out after 59.98 seconds"


class KnownDeadlineLimit(unittest.TestCase):
    def test_recorded_case_timeout_is_a_known_limit_in_the_full_matrix(self):
        blocking, known = classify_failures(
            [record(error=TIMEOUT), record()], diagnostic=False)
        self.assertEqual(blocking, [])
        self.assertEqual(len(known), 1)

    def test_diagnostic_selection_never_tolerates_the_failure(self):
        blocking, known = classify_failures([record(error=TIMEOUT)], diagnostic=True)
        self.assertEqual(len(blocking), 1)
        self.assertEqual(known, [])

    def test_other_error_kinds_and_cases_still_block(self):
        runs = [
            record(error="CalledProcessError: exit 1"),
            record(name="delay-50ms", error=TIMEOUT),
            record(streams=4, error=TIMEOUT),
            record(size=4096, error=TIMEOUT),
        ]
        blocking, known = classify_failures(runs, diagnostic=False)
        self.assertEqual(len(blocking), 4)
        self.assertEqual(known, [])

    def test_known_limits_match_the_recorded_domain(self):
        self.assertEqual(KNOWN_DEADLINE_LIMITS, {("delay-10ms-loss-1pct", 16, 65560)})


if __name__ == "__main__":
    unittest.main()
