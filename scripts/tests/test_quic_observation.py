"""Provenance checks use the real pinned registry source, never a stand-in."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_observation.prepare import patch_source, registry_source


class ObservationProvenance(unittest.TestCase):
    def test_actual_pinned_source_accepts_only_the_read_only_patch(self):
        root = Path(__file__).resolve().parents[2]
        source, checksum = registry_source(root)
        self.assertEqual(len(checksum), 64)
        original = (source/'src/connection/mod.rs').read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            report = patch_source(source, Path(temporary)/'engine', root)
            self.assertEqual(set(report['changed_files']), {
                'src/connection/mod.rs', 'src/connection/observation.rs',
                'src/connection/packet_builder.rs', 'src/connection/streams/state.rs'})
            self.assertEqual(report['pristine_mod_sha256'], hashlib.sha256(original).hexdigest())
            self.assertNotEqual(report['observed_mod_sha256'], report['pristine_mod_sha256'])
        self.assertEqual((source/'src/connection/mod.rs').read_bytes(), original)

class RealStateObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gzip
        import json
        path = (Path(__file__).resolve().parents[2] /
                'moirai-quic/measurements/recovery/2026-10-14/state-native.json.gz')
        cls.evidence = json.loads(gzip.decompress(path.read_bytes()))

    def test_all_original_local_shapes_and_policies_have_actual_peer_states(self):
        from quic_profiles import verify
        from quic_observation.audit import state_trace, state_pair
        runs = self.evidence['runs']
        self.assertEqual(len(runs), 81)
        self.assertEqual({(r['policy'], r['repetition'], r['streams'], r['bytes']) for r in runs},
                         {(p, r, s, b) for p in ('cubic', 'newreno', 'bbr-experimental')
                          for r in range(3) for s in (1, 4, 16) for b in (24, 4096, 65560)})
        observed = 0
        for run in runs:
            state_pair(run['reports']['diagnostics']['client_stderr'],
                       run['reports']['diagnostics']['server_stderr'], True)
            for role, responder in (('client', False), ('server', True)):
                verify(run['reports'][role], responder, run['streams'], run['bytes'])
                trace = state_trace(run['reports']['diagnostics'][role+'_stderr'])
                self.assertEqual(trace['last_protocol_observation'], 'drained')
                self.assertGreater(trace['observed_states'], 0)
                self.assertGreater(len(trace['events']), 0)
                self.assertEqual(sum(b['observations'] for b in trace['buckets']), trace['observed_states'])
                observed += trace['observed_states']
        self.assertGreater(observed, 162)
        from quic_observation.prepare import verify_provenance
        verify_provenance(self.evidence['engine_observation'], Path(__file__).resolve().parents[2])

    def test_missing_states_corrupt_flight_credit_and_frame_ranges_fail(self):
        import copy
        from quic_observation.audit import state_trace, point
        trace = state_trace(self.evidence['runs'][0]['reports']['diagnostics']['client_stderr'])
        original = trace['handshake'][0]
        for key, value in [('flight_bytes', original['fields']['flight_bytes']+1),
                           ('connection_data_sent', original['fields']['connection_max_data']+1),
                           ('spaces', []), ('pto', [3])]:
            corrupt = copy.deepcopy(original); corrupt['fields'][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                point(corrupt)
        ack = next(p for p in trace['events'] if p['kind'] == 'engine ack received')
        corrupt = copy.deepcopy(ack); corrupt['fields']['ranges'] = [[2, 1]]
        with self.assertRaisesRegex(ValueError, 'reversed ACK range'):
            point(corrupt)
        with self.assertRaisesRegex(ValueError, 'engine state capture is absent'):
            state_trace('')

    def test_rejected_idle_pto_selector_capture_remains_rejected(self):
        import gzip
        import json
        from quic_observation.audit import state_trace
        path = (Path(__file__).resolve().parents[2] /
                'moirai-quic/measurements/recovery/2026-10-14/state-native-rejected-pto.json.gz')
        evidence = json.loads(gzip.decompress(path.read_bytes()))
        with self.assertRaisesRegex(ValueError, 'PTO candidate without an armed loss timer'):
            state_trace(evidence['runs'][0]['reports']['diagnostics']['client_stderr'])


if __name__ == '__main__':
    unittest.main()
