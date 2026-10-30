"""Analytical inventory and real archived measurement verification."""

import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_profiles import FRAME_BYTES, MESSAGES, PROFILES, REPETITIONS, STREAMS, verify
from quic_recovery import POLICIES, main, matrix_cases, probe_identity
from quic_recovery_audit import audit, fraction, progress, queue_evidence, sampled_fairness
from fractions import Fraction


class RecoveryContract(unittest.TestCase):
    def test_real_recovery_archives_retain_full_inventory_and_failed_experiments(self):
        directory = Path(__file__).resolve().parents[2] / 'moirai-quic/measurements/recovery/2026-10-13'
        paths = [*directory.glob('3e353caf-matrix-*.json.gz'),
                 *directory.glob('3e353caf-fairness-*.json.gz')]
        evidence = [json.loads(gzip.decompress(p.read_bytes())) for p in paths]
        result = audit(evidence)
        self.assertEqual(result['counts'], {
            'cubic': {'pairs': 108, 'passed': 105, 'failed': 3,
                      'complete_payload_bytes_per_direction': 1_844_797_440},
            'newreno': {'pairs': 108, 'passed': 104, 'failed': 4,
                        'complete_payload_bytes_per_direction': 1_844_785_152},
            'bbr-experimental': {'pairs': 108, 'passed': 107, 'failed': 1,
                                 'complete_payload_bytes_per_direction': 2_247_595_008}})
        self.assertEqual(len(result['matrix']), 324)
        self.assertEqual(len(result['competing']), 30)
        self.assertEqual(sum(f['status'] == 'failed' for r in result['competing'] for f in r['flows']), 21)
        record = next(r for s in evidence for r in s['runs'] if r['profile']['name'] == 'delay-10ms')
        for name, mutate in [
            ('seed', lambda r: r['before'][0]['qdisc'][0]['options'].__setitem__('seed', 1)),
            ('delay', lambda r: r['before'][0]['qdisc'][0]['options']['delay'].__setitem__('delay', 0.05)),
            ('rate', lambda r: r['before'][0]['qdisc'][0]['options']['rate'].__setitem__('rate', 100_000_000)),
            ('payload coverage', lambda r: r['after'][0]['qdisc'][0].__setitem__('bytes', r['before'][0]['qdisc'][0]['bytes']))]:
            corrupted = copy.deepcopy(record)
            mutate(corrupted)
            with self.subTest(name=name), self.assertRaises(ValueError):
                queue_evidence(corrupted)
        evidence[0]['runs'].pop()
        with self.assertRaises(ValueError):
            audit(evidence)

    def test_sampled_fairness_retains_exact_ratio_boundary_ages_and_failed_status(self):
        def trace(counts):
            # Analytical counters at whole-second observations: flow two
            # validates twice as many complete messages as flow one.
            return '\n'.join(
                f'observed_unix_ns={second * 10**9} exchange_elapsed_ns={(second - 1) * 10**9} '
                f'completed_messages={count} completed_cycle_p95_ns=Some(137) '
                'lost_packets=1 congestion_events=1 congestion_window_bytes=12000 '
                'round_trip_ns=20000000 path_mtu=1200 verified progress and recovery snapshot'
                for second, count in enumerate(counts, 1))
        flows = [{'error': 'unchanged deadline', 'stderr': trace([3, 5, 7])},
                 {'error': 'unchanged deadline', 'stderr': trace([6, 10, 14])}]
        evidence = sampled_fairness(flows, 65_560)
        self.assertEqual(evidence['verified_counter_byte_increments'], [262_240, 524_480])
        self.assertEqual(evidence['jain_sampled_progress'], fraction(Fraction(9, 10)))
        self.assertEqual(evidence['boundary_sample_ages'],
                         [{'start_ns': 0, 'end_ns': 0}, {'start_ns': 0, 'end_ns': 0}])
        self.assertEqual([p['completed_messages'] for p in progress(flows[0]['stderr'])], [3, 5, 7])
        self.assertTrue(all('reports' not in flow and 'error' in flow for flow in flows))
        stalled = [{'error': 'deadline', 'stderr': trace([3, 3, 3])} for _ in range(2)]
        self.assertEqual(sampled_fairness(stalled, 65_560)['jain_sampled_progress'], None)
        self.assertIn('unavailable', sampled_fairness([{'error': 'deadline'}, flows[1]], 65_560))
        with self.assertRaisesRegex(ValueError, 'nonmonotonic completed_messages'):
            progress(trace([7, 5, 3]))

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

    def test_exact_cubic_case_selects_one_original_case_per_repetition(self):
        from quic_profiles import admitted_case, admitted_cases
        selected = admitted_case('delay-10ms-loss-1pct:16:65560', admitted_cases(True))
        cases = [case for repetition in range(REPETITIONS)
                 for case in matrix_cases(repetition, selected)]
        self.assertEqual(len(cases), 3)
        self.assertEqual(
            [(profile.name, streams, size, policy) for profile, streams, size, policy in cases],
            [('delay-10ms-loss-1pct', 16, 65_560, 'cubic')] * REPETITIONS)
        workflow = (Path(__file__).resolve().parents[2]
                    / '.github/workflows/quic-recovery.yml').read_text(encoding='utf-8')
        self.assertIn('default: moirai\n        type: choice\n        options:\n'
                      '          - moirai\n          - tokio', workflow)
        self.assertIn("RUNTIME: ${{ inputs.runtime || 'moirai' }}", workflow)
        self.assertIn('moirai) BINARY=target/release/examples/quic_load ;;', workflow)
        self.assertIn('tokio) BINARY=target/release/examples/quic_load_tokio ;;', workflow)
        self.assertEqual(workflow.count('--binary "$BINARY"'), 1)
        self.assertIn('--case "$CASE"', workflow)
        self.assertIn('repetition: [0, 1, 2]', workflow)
        self.assertIn("'[\"matrix\",\"fairness\"]'", workflow)
        self.assertNotIn('--deadline', workflow)

    def test_invalid_case_routes_fail_before_network_construction(self):
        for suite, case in [('matrix', 'delay-10ms-loss-1pct:16:65559'),
                            ('fairness', 'delay-10ms-loss-1pct:16:65560')]:
            arguments = ['quic_recovery.py', '--binary', 'missing', '--tc', 'tc',
                         '--output', 'unused.json', '--suite', suite, '--repetition', '0',
                         '--case', case]
            with self.subTest(suite=suite), mock.patch.object(sys, 'argv', arguments), \
                    mock.patch('quic_recovery.Network') as network, \
                    self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            network.assert_not_called()

    def test_probe_identity_is_derived_from_supported_executables(self):
        self.assertEqual(probe_identity(Path('quic_load')),
                         {'runtime': 'moirai', 'executable': 'quic_load'})
        self.assertEqual(probe_identity(Path('quic_load_tokio.exe')),
                         {'runtime': 'tokio', 'executable': 'quic_load_tokio.exe'})
        with self.assertRaisesRegex(ValueError, "unsupported probe executable 'quic_load_copy'"):
            probe_identity(Path('quic_load_copy'))

    def test_failed_run_report_retains_selected_probe_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'quic_load_tokio'
            binary.write_bytes(b'tokio probe')
            output = Path(directory) / 'report.json'
            arguments = ['quic_recovery.py', '--binary', str(binary), '--tc', 'tc',
                         '--output', str(output), '--suite', 'matrix', '--repetition', '0']

            def command_output(command):
                outputs = {('git', 'rev-parse', 'HEAD'): 'revision\n',
                           ('git', 'rev-parse', 'HEAD^{tree}'): 'tree\n',
                           ('git', 'status', '--porcelain'): '',
                           ('rustc', '-Vv'): 'rustc evidence',
                           ('tc', '-V'): 'tc evidence'}
                return outputs[tuple(command)]

            with mock.patch.object(sys, 'argv', arguments), \
                    mock.patch('quic_recovery.execute', side_effect=command_output), \
                    mock.patch('quic_recovery.Network') as network, \
                    self.assertRaisesRegex(RuntimeError, 'injected setup failure'):
                network.return_value.tc = 'tc'
                network.return_value.setup.side_effect = RuntimeError('injected setup failure')
                main()

            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['probe'],
                             {'runtime': 'tokio', 'executable': 'quic_load_tokio'})
            self.assertEqual(report['binary_sha256'],
                             hashlib.sha256(b'tokio probe').hexdigest())
            self.assertEqual(report['error'], 'RuntimeError: injected setup failure')

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
