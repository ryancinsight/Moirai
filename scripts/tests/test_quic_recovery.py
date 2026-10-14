"""Analytical inventory and real archived measurement verification."""

import copy
import gzip
import itertools
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_profiles import FRAME_BYTES, MESSAGES, PROFILES, REPETITIONS, STREAMS, verify
from quic_recovery import POLICIES, matrix_cases


class RecoveryContract(unittest.TestCase):
    def test_every_policy_retains_the_original_cartesian_domain(self):
        expected = set(itertools.product([p.name for p in PROFILES], STREAMS,
                                         FRAME_BYTES, range(REPETITIONS)))
        actual = {policy: [] for policy in POLICIES}
        for repetition in range(REPETITIONS):
            cases = list(matrix_cases(repetition))
            self.assertEqual([case[3] for case in cases[:3]],
                             list(POLICIES[repetition:] + POLICIES[:repetition]))
            for profile, streams, size, policy in cases:
                actual[policy].append((profile.name, streams, size, repetition))
        for policy, cases in actual.items():
            with self.subTest(policy=policy):
                self.assertEqual(len(cases), 108)
                self.assertEqual(set(cases), expected)
                self.assertEqual(sum(s * MESSAGES * b for _, s, b, _ in cases), 2_247_598_080)

    def test_real_archive_rates_percentiles_and_corruption_rejection(self):
        root = Path(__file__).resolve().parents[2]
        archive = root / 'moirai-quic/measurements/controlled/2026-10-13/8168e21f-matrix.json.gz'
        evidence = json.loads(gzip.decompress(archive.read_bytes()))
        successes = [run for run in evidence['runs'] if 'error' not in run]
        self.assertEqual(len(successes), 105)
        self.assertEqual(sum(run['streams'] * MESSAGES * run['bytes'] for run in successes),
                         1_844_797_440)
        for run in successes:
            for role, responder in [('client', False), ('server', True)]:
                report = run['reports'][role]
                verify(report, responder, run['streams'], run['bytes'])
                for field in ['goodput_bits_per_second', 'p50_ns', 'p95_ns',
                              'verified_payload_bytes_per_direction']:
                    corrupted = copy.deepcopy(report)
                    corrupted[field] += 1
                    with self.subTest(role=role, field=field), self.assertRaises(AssertionError):
                        verify(corrupted, responder, run['streams'], run['bytes'])


if __name__ == '__main__':
    unittest.main()
