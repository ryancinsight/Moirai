"""Value checks and corruption rejection over real authenticated trace inputs."""

import copy
import gzip
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_profiles import verify
from quic_trace_audit import protocol_trace


class ProtocolTraceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (Path(__file__).resolve().parents[2] /
                'moirai-quic/measurements/recovery/2026-10-14/loopback-trace.json.gz')
        cls.evidence = json.loads(gzip.decompress(path.read_bytes()))

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


if __name__ == '__main__':
    unittest.main()
