"""Value-semantic QUIC observation decoder cases."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_observation_support import native_evidence


class RealStateObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = native_evidence()

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

    def test_state_bucket_schema_and_aggregates_match_native_widths(self):
        import copy
        import json
        from quic_observation.audit import state_bucket
        trace = self.evidence['runs'][0]['reports']['diagnostics']['client_stderr']
        line = next(line for line in reversed(trace.splitlines())
                    if 'protocol recovery observation' in line)
        sample = json.JSONDecoder().raw_decode(line.split('protocol_recovery=', 1)[1])[0]
        bucket = sample['engine']['buckets'][0]
        state_bucket(bucket, sample, None)
        missing = copy.deepcopy(bucket)
        missing.pop('max_window_bytes')
        with self.assertRaisesRegex(ValueError, 'engine bucket differs'):
            state_bucket(missing, sample, None)
        widths = {
            'second': 2**128, 'observations': 2**64,
            'min_flight_bytes': 2**64, 'max_flight_bytes': 2**64,
            'min_window_bytes': 2**64, 'max_window_bytes': 2**64,
            'min_connection_credit': 2**64, 'max_connection_credit': 2**64,
            'max_pacing_remaining_ns': 2**64, 'pacing_armed_observations': 2**64,
        }
        for field, overflow in widths.items():
            for invalid in (True, overflow):
                malformed = copy.deepcopy(bucket)
                malformed[field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    state_bucket(malformed, sample, None)
        malformed = copy.deepcopy(bucket)
        malformed['min_flight_bytes'] = malformed['max_flight_bytes'] + 1
        with self.assertRaisesRegex(ValueError, 'extrema exclude'):
            state_bucket(malformed, sample, None)
        malformed = copy.deepcopy(bucket)
        malformed['pacing_armed_observations'] = 0
        malformed['max_pacing_remaining_ns'] = 1
        with self.assertRaisesRegex(ValueError, 'pacing observation aggregate'):
            state_bucket(malformed, sample, None)

    def test_pacing_aggregates_cover_retained_state_endpoints(self):
        import copy
        import json
        from quic_observation.audit import state_bucket, state_trace
        trace = self.evidence['runs'][0]['reports']['diagnostics']['client_stderr']
        line = next(line for line in reversed(trace.splitlines())
                    if 'protocol recovery observation' in line)
        sample = json.JSONDecoder().raw_decode(line.split('protocol_recovery=', 1)[1])[0]
        bucket = copy.deepcopy(sample['engine']['buckets'][0])
        point = bucket['first']
        point['fields']['timers'] = [row for row in point['fields']['timers'] if row[0] != 6]
        point['fields']['timers'].append([6, 37])
        bucket.update({'observations': 1, 'last': copy.deepcopy(point),
                       'max_pacing_remaining_ns': 37, 'pacing_armed_observations': 1})
        fields = point['fields']
        credit = fields['connection_max_data']-fields['connection_data_sent']
        for name, value in (('flight_bytes', fields['flight_bytes']),
                            ('window_bytes', fields['window_bytes']),
                            ('connection_credit', credit)):
            bucket['min_'+name] = value
            bucket['max_'+name] = value
        singleton = copy.deepcopy(sample)
        singleton['engine']['observed_states'] = 1
        singleton['engine']['buckets'] = [bucket]
        encoded = ('protocol recovery observation protocol_recovery=' +
                   json.dumps(singleton))
        decoded = state_trace(encoded)
        self.assertEqual(decoded['buckets'][0]['pacing_armed_observations'], 1)
        self.assertEqual(decoded['buckets'][0]['max_pacing_remaining_ns'], 37)

        malformed = copy.deepcopy(bucket)
        malformed['observations'] = 3
        malformed['pacing_armed_observations'] = 0
        malformed['max_pacing_remaining_ns'] = 0
        with self.assertRaisesRegex(ValueError, 'aggregate excludes a retained point'):
            state_bucket(malformed, sample, None)

    def test_sampled_state_history_matches_native_final_order_and_counts(self):
        import copy
        import json
        from quic_observation.audit import state_trace
        trace = self.evidence['runs'][80]['reports']['diagnostics']['client_stderr']
        lines = [line for line in trace.splitlines()
                 if 'protocol recovery observation' in line]
        samples = [json.JSONDecoder().raw_decode(
            line.split('protocol_recovery=', 1)[1])[0] for line in lines]
        state_samples = [sample for sample in samples if sample.get('engine')]
        actual = state_trace(trace)
        self.assertEqual([bucket['second'] for bucket in state_samples[0]['engine']['buckets']],
                         [1])
        self.assertEqual([bucket['second'] for bucket in state_samples[-1]['engine']['buckets']],
                         [0, 1])
        self.assertEqual(actual['buckets'], state_samples[-1]['engine']['buckets'])

        def encoded(values):
            return '\n'.join('protocol recovery observation protocol_recovery=' +
                             json.dumps(value) for value in values)

        invalid = copy.deepcopy(samples)
        selected = next(value for value in invalid
                        if value.get('observation') == 'sample' and
                        value.get('engine', {}).get('buckets'))
        selected['engine']['buckets'][0]['observations'] = selected['engine']['observed_states'] + 1
        with self.assertRaisesRegex(ValueError, 'invalid bucket count'):
            state_trace(encoded(invalid))

        invalid = copy.deepcopy(samples)
        selected = next(value for value in invalid
                        if value.get('observation') == 'sample' and
                        value.get('engine', {}).get('buckets'))
        selected['engine']['buckets'].append(
            copy.deepcopy(state_samples[-1]['engine']['buckets'][0]))
        with self.assertRaisesRegex(ValueError, 'sample retains more than'):
            state_trace(encoded(invalid))

        invalid = copy.deepcopy(samples)
        for sample in invalid:
            if sample.get('engine') and sample.get('observation') != 'sample':
                removed = sum(bucket['observations'] for bucket in sample['engine']['buckets']
                              if bucket['second'] == 1)
                sample['engine']['buckets'] = [bucket for bucket in sample['engine']['buckets']
                                               if bucket['second'] != 1]
                sample['engine']['buckets'][0]['observations'] += removed
        with self.assertRaisesRegex(ValueError, 'omits sampled history'):
            state_trace(encoded(invalid))

        invalid = [copy.deepcopy(value) for value in samples
                   if value.get('observation') == 'sample' and value.get('engine')]
        invalid.append(copy.deepcopy(state_samples[-1]))
        invalid[-1]['engine']['buckets'].reverse()
        with self.assertRaisesRegex(ValueError, 'insertion order changes'):
            state_trace(encoded(invalid))

        invalid = copy.deepcopy(samples)
        invalid[-1]['engine']['observed_states'] += 1
        with self.assertRaisesRegex(ValueError, 'count differs from full bucket history'):
            state_trace(encoded(invalid))

        for invalid_count in (True, 2**64):
            invalid = copy.deepcopy(samples)
            invalid[-1]['engine']['observed_states'] = invalid_count
            with self.subTest(count=invalid_count), self.assertRaisesRegex(
                    ValueError, 'no actual engine states'):
                state_trace(encoded(invalid))

        invalid = copy.deepcopy(samples)
        invalid[-1]['observation'] = 'sample'
        final_bucket = invalid[-1]['engine']['buckets'][-1]
        final_bucket['observations'] = invalid[-1]['engine']['observed_states']
        invalid[-1]['engine']['buckets'] = [final_bucket]
        with self.assertRaisesRegex(ValueError, 'final engine capture is not a full snapshot'):
            state_trace(encoded(invalid))

    def test_pacing_aggregates_equal_possible_hidden_observations(self):
        import copy
        import itertools
        import json
        from quic_observation.audit import state_bucket
        trace = self.evidence['runs'][0]['reports']['diagnostics']['client_stderr']
        line = next(line for line in reversed(trace.splitlines())
                    if 'protocol recovery observation' in line)
        sample = json.JSONDecoder().raw_decode(line.split('protocol_recovery=', 1)[1])[0]
        original = sample['engine']['buckets'][0]['first']
        delays = (None, 0, 37, 38)
        for count in range(1, 5):
            retained_count = min(count, 2)
            for retained in itertools.product(delays, repeat=retained_count):
                bucket = copy.deepcopy(sample['engine']['buckets'][0])
                bucket.update(second=0, observations=count)
                for index, key in enumerate(('first', 'last')):
                    point = copy.deepcopy(original)
                    point['elapsed_ns'] = 100 if count == 1 else 100 + index
                    timer = retained[min(index, retained_count - 1)]
                    point['fields']['timers'] = [row for row in point['fields']['timers']
                                                if row[0] != 6]
                    if timer is not None:
                        point['fields']['timers'].append([6, timer])
                    bucket[key] = point
                fields = original['fields']
                for name, value in (
                        ('flight_bytes', fields['flight_bytes']),
                        ('window_bytes', fields['window_bytes']),
                        ('connection_credit', fields['connection_max_data'] -
                         fields['connection_data_sent'])):
                    bucket['min_' + name] = bucket['max_' + name] = value
                possible = set()
                for hidden in itertools.product(delays, repeat=count - retained_count):
                    actual = retained + hidden
                    possible.add((sum(value is not None for value in actual),
                                  max((value or 0 for value in actual), default=0)))
                for armed, maximum in itertools.product(range(count + 2), delays[1:]):
                    bucket['pacing_armed_observations'] = armed
                    bucket['max_pacing_remaining_ns'] = maximum
                    with self.subTest(count=count, retained=retained, armed=armed, maximum=maximum):
                        capture = {'elapsed_ns': 300, 'engine': {'observed_states': count}}
                        if (armed, maximum) in possible:
                            state_bucket(bucket, capture, None)
                        else:
                            with self.assertRaisesRegex(ValueError, 'pacing'):
                                state_bucket(bucket, capture, None)

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
