"""Value checks and corruption rejection over real authenticated trace inputs."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_profiles import Network, verify
from quic_recovery_audit import audit_default
from quic_trace_audit import protocol_trace, trace_audit


class ProtocolTraceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (Path(__file__).resolve().parents[2] /
                'moirai-quic/measurements/recovery/2026-10-14/loopback-trace.json.gz')
        cls.evidence = json.loads(gzip.decompress(path.read_bytes()))

    def test_namespace_probe_forwards_only_the_explicit_diagnostic_switch(self):
        network = Network(True)
        for role in (0, 1):
            self.assertEqual(network.probe_prefix(role, False),
                             ['sudo', 'ip', 'netns', 'exec', network.names[role]])
            self.assertEqual(network.probe_prefix(role, True),
                             ['sudo', 'ip', 'netns', 'exec', network.names[role],
                              'env', 'MOIRAI_QUIC_RECOVERY_TRACE=1'])
        for trace in (False, True):
            self.assertEqual(Network(False).probe_prefix(0, trace), [])

    def test_real_native_traces_match_authenticated_complete_reports(self):
        runs = self.evidence['runs']
        self.assertEqual(len(runs), 81)
        established = wakes = 0
        for run in runs:
            for role, responder in (('client', False), ('server', True)):
                report = run['reports'][role]
                verify(report, responder, run['streams'], run['bytes'])
                result = protocol_trace(run['reports']['diagnostics'][role+'_stderr'], True)
                self.assertEqual(result, run['traces'][role])
                self.assertEqual([p['phase'] for p in result['before_drain']['phases']],
                                 ['authenticated', 'exchange', 'complete'])
                established += result['last']['counts']['established']
                wakes += result['last']['counts']['timer_driver_wakes']
        self.assertEqual(established, 162)
        self.assertGreater(wakes, 0)

    def test_complete_linux_study_retains_all_cases_preflights_and_failure_spaces(self):
        directory = (Path(__file__).resolve().parents[2] /
                     'moirai-quic/measurements/recovery/2026-10-14')
        counts = {
            'cubic': {'pairs': 108, 'passed': 105, 'failed': 3,
                      'complete_payload_bytes_per_direction': 1_844_797_440},
            'newreno': {'pairs': 108, 'passed': 105, 'failed': 3,
                        'complete_payload_bytes_per_direction': 1_844_797_440},
            'bbr-experimental': {'pairs': 108, 'passed': 108, 'failed': 0,
                                 'complete_payload_bytes_per_direction': 2_247_598_080}}
        for source, data_ptos, lateness in (('733afc9f', [5, 5, 6], 109_881),
                                            ('140d60cf', [5, 5, 6, 6, 6, 6], 85_146)):
            with self.subTest(source=source):
                paths = [*sorted(directory.glob(f'{source}-matrix-*.json.gz')),
                         *sorted(directory.glob(f'{source}-fairness-*.json.gz'))]
                shards = [json.loads(gzip.decompress(p.read_bytes())) for p in paths]
                result = trace_audit(shards)
                self.assertEqual(result, json.loads(gzip.decompress(
                    (directory/f'{source}-audit.json.gz').read_bytes())))
                self.assertEqual(result['counts'], counts)
                self.assertEqual(len(result['protocol_traces']), 384)
                early = [t for t in result['protocol_traces'] if t['suite'] == 'fairness'
                         and t['server']['before_drain']
                         and t['server']['before_drain']['observation'] == 'failure'
                         and t['server']['before_drain']['phase'] == 'handshake']
                self.assertEqual(len(early), len(data_ptos))
                self.assertEqual(sorted(t['client']['before_drain']['counts']['pto_data']
                                        for t in early), data_ptos)
                self.assertEqual([t['client']['before_drain']['counts']['pto_handshake']
                                  for t in early], [1] * len(data_ptos))
                self.assertEqual(max(t['server']['before_drain']['counts']['max_timer_lateness_ns']
                                     for t in early), lateness)
                errors = [(r['policy'], r['repetition'], role, q['udp_error_deltas'])
                          for r in result['matrix'] for role, q in enumerate(r['queues'])
                          if any(q['udp_error_deltas'].values())]
                self.assertEqual(errors, [] if source == '733afc9f' else [
                    ('newreno', 2, 0, {'InErrors': 3, 'RcvbufErrors': 3, 'SndbufErrors': 0,
                                       'InCsumErrors': 0, 'MemErrors': 0})])
                altered = copy.deepcopy(shards)
                altered[0].pop('trace_preflight')
                with self.assertRaisesRegex(ValueError, 'trace preflight is absent/failed'):
                    trace_audit(altered)

    def test_original_gate_remains_full_uninstrumented_and_independent(self):
        directory = (Path(__file__).resolve().parents[2] /
                     'moirai-quic/measurements/recovery/2026-10-14')
        for source in ('733afc9f', '140d60cf'):
            with self.subTest(source=source):
                shard = json.loads(gzip.decompress(
                    (directory/f'{source}-default-matrix.json.gz').read_bytes()))
                result = audit_default(shard)
                self.assertEqual(result, json.loads(gzip.decompress(
                    (directory/f'{source}-default-audit.json.gz').read_bytes())))
                self.assertEqual(result['counts']['cubic'], {
                    'pairs': 108, 'passed': 105, 'failed': 3,
                    'complete_payload_bytes_per_direction': 1_844_797_440})
                failed = [r for r in result['matrix'] if r['status'] == 'failed']
                self.assertEqual({(r['profile']['name'], r['streams'], r['bytes'], r['repetition'])
                                  for r in failed},
                                 {('delay-10ms-loss-1pct', 16, 65_560, n) for n in range(3)})
                altered = copy.deepcopy(shard)
                altered['runs'].pop()
                with self.assertRaisesRegex(ValueError, 'missing/duplicate original cases'):
                    audit_default(altered)
        # Retain the complete failed diagnostic experiment, while rejecting
        # its opt-in intent without actual child-process capture/preflight.
        paths = [*directory.glob('d946459c-matrix-*.json.gz'),
                 *directory.glob('d946459c-fairness-*.json.gz')]
        filtered = [json.loads(gzip.decompress(p.read_bytes())) for p in paths]
        with self.assertRaisesRegex(ValueError, 'trace preflight is absent/failed'):
            trace_audit(filtered)

    def test_retained_archive_inventory_and_both_hashes_match(self):
        directory = (Path(__file__).resolve().parents[2] /
                     'moirai-quic/measurements/recovery/2026-10-14')
        archives = json.loads((directory/'archives.json').read_text())['archives']
        self.assertEqual({p.name for p in directory.glob('*.json.gz')},
                         {a['name'] for a in archives})
        self.assertEqual(len({a['name'] for a in archives}), len(archives))
        for archive in archives:
            with self.subTest(archive=archive['name']):
                raw = (directory/archive['name']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), archive['compressed_sha256'])
                self.assertEqual(hashlib.sha256(gzip.decompress(raw)).hexdigest(),
                                 archive['uncompressed_sha256'])

    def test_required_gates_retain_the_actual_performance_failure(self):
        path = (Path(__file__).resolve().parents[2] /
                'moirai-quic/measurements/recovery/2026-10-14/140d60cf-required-checks.json.gz')
        run = json.loads(gzip.decompress(path.read_bytes()))
        self.assertEqual(run['databaseId'], 38594188048)
        self.assertEqual(run['headSha'], '140d60cf43e152b816ccd8d4ff5ac12a0e53dfdf')
        self.assertEqual((run['status'], run['conclusion']), ('completed', 'failure'))
        self.assertEqual({j['name'] for j in run['jobs'] if j['conclusion'] == 'failure'}, {
            'Native QUIC validation / QUIC controlled network profiles', 'Verification result'})
        conclusions = {j['name']: j['conclusion'] for j in run['jobs']}
        for job in ('Supply-chain', 'Lockfile integrity / Lockfile integrity', 'Workspace gate',
                    'Native QUIC validation / QUIC windows-latest',
                    'Native QUIC validation / QUIC ubuntu-24.04'):
            with self.subTest(job=job):
                self.assertEqual(conclusions[job], 'success')

    def test_missing_outcomes_corrupt_counts_and_event_inventories_fail(self):
        original = self.evidence['runs'][0]['reports']['diagnostics']['client_stderr']
        lines = original.splitlines()
        trace_lines = [line for line in lines if 'protocol recovery observation' in line]
        self.assertGreaterEqual(len(trace_lines), 3)
        prefix, raw = trace_lines[-1].split('protocol_recovery=', 1)
        value, end = json.JSONDecoder().raw_decode(raw)

        def corrupted(mutate):
            changed = copy.deepcopy(value)
            mutate(changed)
            replacement = prefix + 'protocol_recovery=' + json.dumps(changed) + raw[end:]
            return original.replace(trace_lines[-1], replacement)

        mutations = {
            'negative count': lambda s: s['counts'].__setitem__('pacing_blocked', -1),
            'missing counter': lambda s: s['counts'].pop('congestion_blocked'),
            'timer sum': lambda s: s['counts'].__setitem__('max_timer_lateness_ns',
                                                        s['counts']['timer_lateness_ns'] + 1),
            'lost PTO inventory': lambda s: s['counts'].__setitem__('pto_handshake',
                                                                 s['counts']['pto_handshake'] + 1),
            'phase mismatch': lambda s: s.__setitem__('phase', 'exchange'),
            'changed history': lambda s: s['phases'][0].__setitem__('elapsed_ns', s['elapsed_ns'] + 1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                protocol_trace(corrupted(mutate), True)
        for observation in ('start', 'success', 'drained'):
            altered = '\n'.join(line for line in lines if f'"observation":"{observation}"' not in line)
            with self.subTest(missing=observation), self.assertRaises(ValueError):
                protocol_trace(altered, True)
        with self.assertRaisesRegex(ValueError, 'failure contains trace success'):
            protocol_trace(original, False)
        # A pre-drain exchange can complete before a real drain failure. Its
        # diagnostic exchange outcome never turns that pair into a success.
        altered = original.replace('"observation":"drained"', '"observation":"drain_failure"')
        self.assertEqual(protocol_trace(altered, False)['last']['observation'], 'drain_failure')


if __name__ == '__main__':
    unittest.main()
