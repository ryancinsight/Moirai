"""Computed-value audits of real maximum-workload handshake-loss experiments."""
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_profiles import verify
from quic_observation.audit import state_pair

ROOT = Path(__file__).resolve().parents[2]
HISTORY = Path(os.environ.get('QUIC_HISTORY_REPOSITORY', ROOT))
DIRECTORY = ROOT/'moirai-quic/measurements/recovery/2026-10-14/state'


class HandshakeRecoveryEvidence(unittest.TestCase):
    def test_normalized_diff_preserves_exact_measured_engine_source(self):
        import shutil
        import subprocess
        import tempfile
        from quic_observation.prepare import (patch_source, provenance_inputs,
                                               registry_source, verify_provenance)
        measured = json.loads(gzip.decompress((DIRECTORY/'handshake-measured-patch.json.gz').read_bytes()))
        native = json.loads(gzip.decompress((DIRECTORY/'candidate-native-contracts.json.gz').read_bytes()))
        self.assertEqual(hashlib.sha256(measured['patch_text'].encode()).hexdigest(), measured['patch_sha256'])
        self.assertEqual(hashlib.sha256((DIRECTORY/'handshake-candidate.patch').read_bytes()).hexdigest(),
                         measured['normalized_patch_sha256'])
        shard = json.loads(gzip.decompress(next(DIRECTORY.glob('03dcc4a3-matrix-*.json.gz')).read_bytes()))
        provenance = shard['engine_observation']
        verify_provenance(provenance, ROOT)
        observer_patch, _ = provenance_inputs(provenance, ROOT, HISTORY)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, _ = registry_source(ROOT, directory/'registry')
            baseline = directory/'baseline'
            patch_source(source, baseline, ROOT, observer_patch,
                         set(provenance['changed_files']))
            self.assertEqual(hashlib.sha256(
                (baseline/'src/connection/mod.rs').read_bytes()).hexdigest(),
                native['baseline_mod_sha256'])
            original = directory/'measured.patch'
            original.write_bytes(measured['patch_text'].encode())
            patches = (('measured', original),
                       ('normalized', DIRECTORY/'handshake-candidate.patch'))
            for label, patch in patches:
                engine = directory/label
                shutil.copytree(baseline, engine)
                # The observer patch is insertion-only; this separately authenticated
                # historical controller patch intentionally replaces behavior.
                subprocess.run(['git', '-c', 'core.autocrlf=false', 'apply', str(patch)],
                               cwd=engine, check=True)
                path = engine/'src/connection/mod.rs'
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), native['candidate_mod_sha256'])

    def test_complete_linux_observer_and_independent_pristine_gate_replay_exactly(self):
        from quic_trace_audit import trace_audit
        from quic_recovery_audit import audit_default
        paths = [*sorted(DIRECTORY.glob('03dcc4a3-matrix-*.json.gz')),
                 *sorted(DIRECTORY.glob('03dcc4a3-fairness-*.json.gz'))]
        shards = [json.loads(gzip.decompress(p.read_bytes())) for p in paths]
        result = trace_audit(shards)
        self.assertEqual(result, json.loads(gzip.decompress(
            (DIRECTORY/'03dcc4a3-audit.json.gz').read_bytes())))
        self.assertEqual({p: (c['pairs'], c['passed']) for p, c in result['counts'].items()}, {
            'cubic': (108, 105), 'newreno': (108, 105), 'bbr-experimental': (108, 108)})
        self.assertEqual(len(result['engine_states']), 384)
        stalled = [t for t in result['engine_states'] if t['suite'] == 'fairness'
                   and any(p['fields']['pto'] == [2] and p['fields']['queued']
                           for p in t['client']['ptos'])]
        self.assertEqual(len(stalled), 6)
        for trace in stalled:
            points = [p['fields'] for p in trace['client']['ptos']
                      if p['fields']['pto'] == [2] and p['fields']['queued']]
            self.assertEqual(points[-1]['queued'], [[1, 2246, 634]])
            self.assertEqual(points[-1]['spaces'][1][7], 0)
            self.assertEqual(points[-1]['spaces'][1][4], 2880)
            server = trace['server']['buckets'][-1]['last']['fields']
            self.assertEqual(server['spaces'][1][5], 2246)
            self.assertTrue(server['tls_handshake'])
        original = audit_default(json.loads(gzip.decompress(
            (DIRECTORY/'03dcc4a3-original-matrix.json.gz').read_bytes())))
        self.assertEqual(original, json.loads(gzip.decompress(
            (DIRECTORY/'03dcc4a3-original-audit.json.gz').read_bytes())))
        self.assertEqual(original['counts']['cubic']['passed'], 105)
        self.assertEqual(original['counts']['cubic']['complete_payload_bytes_per_direction'], 1_844_797_440)
        self.assertEqual({(r['profile']['name'], r['streams'], r['bytes'], r['repetition'])
                          for r in original['matrix'] if r['status'] == 'failed'},
                         {('delay-10ms-loss-1pct', 16, 65560, n) for n in range(3)})

    def test_actual_feature_security_gates_pass_while_original_performance_fails(self):
        run = json.loads(gzip.decompress((DIRECTORY/'03dcc4a3-required-checks.json.gz').read_bytes()))
        self.assertEqual(run['headSha'], '03dcc4a31d7f793e4cb1cb7d96ff74f47033946c')
        conclusions = {j['name']: j['conclusion'] for j in run['jobs']}
        self.assertEqual(conclusions['Native QUIC validation / QUIC controlled network profiles'], 'failure')
        for job in ('Workspace gate', 'Supply-chain', 'Lockfile integrity / Lockfile integrity',
                    'Native QUIC validation / QUIC ubuntu-24.04',
                    'Native QUIC validation / QUIC windows-latest'):
            self.assertEqual(conclusions[job], 'success')

    def test_repeated_real_peers_preserve_exact_domain_and_security_boundaries(self):
        expected = set(itertools.product(('cubic', 'newreno', 'bbr-experimental'), range(3), range(3)))
        for label, complete in (('baseline', 17), ('candidate', 27)):
            with self.subTest(engine=label):
                evidence = json.loads(gzip.decompress((DIRECTORY/f'handshake-{label}.json.gz').read_bytes()))
                self.assertEqual(evidence['experiment_sha256'], hashlib.sha256(
                    (ROOT/'scripts/quic_observation/handshake.py').read_bytes()).hexdigest())
                if label == 'candidate':
                    measured = json.loads(gzip.decompress((DIRECTORY/'handshake-measured-patch.json.gz').read_bytes()))
                    self.assertEqual(evidence['candidate_patch_sha256'], measured['patch_sha256'])
                else:
                    self.assertIsNone(evidence['candidate_patch_sha256'])
                runs = evidence['runs']
                self.assertEqual(len(runs), 27)
                self.assertEqual({(r['policy'], r['repetition'], r['drop_client_handshake_datagram'])
                                  for r in runs}, expected)
                self.assertEqual(sum(r['complete'] for r in runs), complete)
                for run in runs:
                    self.assertEqual((run['streams'], run['messages'], run['bytes']), (16, 128, 65_560))
                    self.assertEqual((run['pair_deadline_seconds'], run['operation_deadline_seconds']), (60, 5))
                    self.assertEqual(run['errors'], [])
                    dropped = [p for p in run['handshake_datagrams'] if p['dropped']]
                    self.assertEqual(len(dropped), 1)
                    self.assertEqual(dropped[0]['ordinal'], run['drop_client_handshake_datagram'])
                    self.assertIn('Handshake', dropped[0]['spaces'])
                    states = state_pair(run['client_stderr'], run['server_stderr'], run['complete'])
                    self.assertEqual(states, run['states'])
                    if run['complete']:
                        self.assertEqual((run['client_returncode'], run['server_returncode']), (0, 0))
                        for role, responder in (('client', False), ('server', True)):
                            report = json.loads(run[role+'_stdout'])
                            verify(report, responder, 16, 65_560)
                            self.assertEqual(report['congestion_policy'], run['policy'])
                    else:
                        self.assertNotEqual((run['client_returncode'], run['server_returncode']), (0, 0))
                        blocked = [p['fields'] for p in states['client']['ptos']
                                   if p['fields']['pto'] == [2] and
                                   any(row[0] == 1 for row in p['fields']['queued'])]
                        self.assertGreater(len(blocked), 0)
                        self.assertEqual(blocked[0]['spaces'][1][7], 0)
                        self.assertGreater(blocked[0]['flight_bytes'], blocked[0]['window_bytes'])
                        server = states['server']['buckets'][-1]['last']['fields']
                        self.assertTrue(server['tls_handshake'])
                        self.assertLess(server['spaces'][1][5], blocked[-1]['spaces'][1][4])

    def test_state_archive_inventory_and_hashes_cover_complete_and_failed_evidence(self):
        archives = json.loads((DIRECTORY/'archives.json').read_text())['archives']
        self.assertEqual({p.name for p in DIRECTORY.glob('*.json.gz')}, {a['name'] for a in archives})
        self.assertEqual(len(archives), len({a['name'] for a in archives}))
        for archive in archives:
            raw = (DIRECTORY/archive['name']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), archive['compressed_sha256'])
            self.assertEqual(hashlib.sha256(gzip.decompress(raw)).hexdigest(), archive['uncompressed_sha256'])
        rejected = json.loads(gzip.decompress((DIRECTORY/'handshake-recording-failure.json.gz').read_bytes()))
        self.assertEqual(rejected['status'], 'incomplete evidence')

    def test_native_candidate_contracts_and_source_difference_are_recorded(self):
        from quic_observation.prepare import verify_provenance
        value = json.loads(gzip.decompress((DIRECTORY/'candidate-native-contracts.json.gz').read_bytes()))
        self.assertEqual(value['changed_engine_files'], ['src/connection/mod.rs'])
        self.assertNotEqual(value['candidate_mod_sha256'], value['baseline_mod_sha256'])
        measured = json.loads(gzip.decompress((DIRECTORY/'handshake-measured-patch.json.gz').read_bytes()))
        self.assertEqual(value['candidate_patch_sha256'], measured['patch_sha256'])
        shard = json.loads(gzip.decompress(next(DIRECTORY.glob('03dcc4a3-matrix-*.json.gz')).read_bytes()))
        verify_provenance(shard['engine_observation'], ROOT)
        self.assertEqual(value['canonical_lock_sha256'],
                         shard['engine_observation']['canonical_lock_sha256'])
        self.assertEqual(value['result']['returncode'], 0)
        self.assertIn('23 tests run: 23 passed, 0 skipped', value['log'])
        self.assertNotEqual(value['rejected_invocation']['result']['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
