"""Value-semantic QUIC observation decoder cases."""
import copy
import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_observation.audit import StateOutcome, state_trace
from quic_observation_support import native_evidence


def deadline_sample(evidence):
    stderr = evidence['runs'][0]['reports']['diagnostics']['client_stderr']
    line = next(line for line in reversed(stderr.splitlines()) if 'protocol recovery observation' in line)
    sample = json.JSONDecoder().raw_decode(line.split('protocol_recovery=', 1)[1])[0]
    return {**sample, 'observation': 'sample'}, sample['elapsed_ns']


def deadline_summary(at, kind='engine pacing timeout', count=1, maximum=3):
    return {'kind': kind, 'observations': count, 'max_lateness_ns': maximum,
            'witness': {'elapsed_ns': at, 'kind': kind, 'fields': {'lateness_ns': maximum}}}


def encode_deadlines(values):
    return '\n'.join(f'protocol recovery observation protocol_recovery={json.dumps(value)}' for value in values)


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
        from quic_observation.audit import StateOutcome, state_trace
        trace = self.evidence['runs'][80]['reports']['diagnostics']['client_stderr']
        lines = [line for line in trace.splitlines()
                 if 'protocol recovery observation' in line]
        samples = [json.JSONDecoder().raw_decode(
            line.split('protocol_recovery=', 1)[1])[0] for line in lines]
        state_samples = [sample for sample in samples if sample.get('engine')]
        actual = state_trace(trace)
        with self.assertRaisesRegex(TypeError, 'StateOutcome'):
            state_trace(trace, False)
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
        for outcome in StateOutcome:
            with self.subTest(outcome=outcome), self.assertRaisesRegex(
                    ValueError, 'invalid bucket count'):
                state_trace(encoded(invalid), outcome)

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
        for outcome in StateOutcome:
            with self.subTest(outcome=outcome), self.assertRaisesRegex(
                    ValueError, 'omits sampled history'):
                state_trace(encoded(invalid), outcome)

        invalid = [copy.deepcopy(value) for value in samples
                   if value.get('observation') == 'sample' and value.get('engine')]
        invalid.append(copy.deepcopy(state_samples[-1]))
        invalid[-1]['engine']['buckets'].reverse()
        for outcome in StateOutcome:
            with self.subTest(outcome=outcome), self.assertRaisesRegex(
                    ValueError, 'insertion order changes'):
                state_trace(encoded(invalid), outcome)

        invalid = copy.deepcopy(samples)
        invalid[-1]['engine']['observed_states'] += 1
        for outcome in StateOutcome:
            with self.subTest(outcome=outcome), self.assertRaisesRegex(
                    ValueError, 'count differs from full bucket history'):
                state_trace(encoded(invalid), outcome)

        earlier, later = next((first, second) for first, second in
                               zip(state_samples, state_samples[1:])
                               if first['engine']['observed_states'] > 1)
        invalid = copy.deepcopy([earlier, later])
        invalid[-1]['engine']['observed_states'] = invalid[0]['engine']['observed_states'] - 1
        with self.assertRaisesRegex(ValueError, 'engine count decreases'):
            state_trace(encoded(invalid), StateOutcome.FAILURE)

        earlier, later = next((first, second) for first, second in
                               zip(state_samples, state_samples[1:])
                               if ({bucket['second'] for bucket in first['engine']['buckets']} &
                                   {bucket['second'] for bucket in second['engine']['buckets']}))
        invalid = copy.deepcopy([earlier, later])
        common = ({bucket['second'] for bucket in invalid[0]['engine']['buckets']} &
                  {bucket['second'] for bucket in invalid[1]['engine']['buckets']}).pop()
        changed = next(bucket for bucket in invalid[1]['engine']['buckets']
                       if bucket['second'] == common)
        changed['first']['fields']['tls_handshake'] ^= True
        if changed['observations'] == 1:
            changed['last'] = copy.deepcopy(changed['first'])
        with self.assertRaisesRegex(ValueError, 'bucket history changes'):
            state_trace(encoded(invalid), StateOutcome.FAILURE)

        for invalid_count in (True, 2**64):
            invalid = copy.deepcopy(samples)
            invalid[-1]['engine']['observed_states'] = invalid_count
            with self.subTest(count=invalid_count), self.assertRaisesRegex(
                    ValueError, 'no actual engine states'):
                state_trace(encoded(invalid))

        invalid = copy.deepcopy(samples)
        invalid[-1]['observation'] = 'sample'
        final_bucket = invalid[-1]['engine']['buckets'][-1]
        invalid[-1]['engine']['buckets'] = [final_bucket]
        with self.assertRaisesRegex(ValueError, 'final engine capture is not a full snapshot'):
            state_trace(encoded(invalid))

    def test_deadline_summary_history_is_bounded_monotonic_and_source_witnessed(self):
        sample, now = deadline_sample(self.evidence)
        later = copy.deepcopy(sample)
        later['elapsed_ns'] = now + 10
        later['engine'].update({'deadline_observations': 2,
                                'deadlines': [deadline_summary(now + 10, count=2, maximum=7)]})
        decoded = state_trace(encode_deadlines([sample, later]), StateOutcome.FAILURE)
        self.assertEqual((decoded['deadlines'][0]['max_lateness_ns'], decoded['deadline_observations']), (7, 2))

        first = copy.deepcopy(sample)
        first['engine'].update({'deadline_observations': 1, 'deadlines': [deadline_summary(now)]})
        second = copy.deepcopy(later)
        second['engine'].update({'deadline_observations': 3, 'deadlines': [
            deadline_summary(now + 10, count=2, maximum=7),
            deadline_summary(now + 10, 'engine loss timeout', 1, 0)]})
        mixed = copy.deepcopy(first)
        mixed['engine'] = None
        self.assertEqual(state_trace(encode_deadlines([mixed, first, second]),
                                     StateOutcome.FAILURE)['deadline_observations'], 3)

    def test_deadline_summary_rejects_malformed_and_contradictory_history(self):
        base, now = deadline_sample(self.evidence)
        base['engine'].update({'deadline_observations': 1, 'deadlines': [deadline_summary(now)]})
        for path in ('deadline_observations', 'observations', 'max_lateness_ns', 'lateness_ns'):
            invalid = copy.deepcopy(base)
            if path == 'deadline_observations':
                invalid['engine'][path] = True
            elif path == 'lateness_ns':
                invalid['engine']['deadlines'][0]['witness']['fields'][path] = True
            else:
                invalid['engine']['deadlines'][0][path] = True
            with self.subTest(boolean=path), self.assertRaises(ValueError):
                state_trace(encode_deadlines([invalid]), StateOutcome.FAILURE)
        for missing in ('deadline_observations', 'deadlines'):
            invalid = copy.deepcopy(base)
            invalid['engine'].pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(
                    ValueError, 'extension is incomplete'):
                state_trace(encode_deadlines([invalid]), StateOutcome.FAILURE)
        for field, value in (('kind', 'engine loss timeout'),
                             ('maximum', 4), ('elapsed_ns', now + 1)):
            invalid = copy.deepcopy(base)
            target = invalid['engine']['deadlines'][0]
            if field != 'maximum':
                target = target['witness']
            target['max_lateness_ns' if field == 'maximum' else field] = value
            with self.subTest(witness=field), self.assertRaisesRegex(ValueError, 'witness'):
                state_trace(encode_deadlines([invalid]), StateOutcome.FAILURE)
        later = copy.deepcopy(base)
        later['elapsed_ns'] = now + 1
        later['engine']['deadline_observations'] = 2
        later['engine']['deadlines'][0].update({'observations': 2, 'max_lateness_ns': 2})
        later['engine']['deadlines'][0]['witness'].update({'elapsed_ns': now + 1})
        later['engine']['deadlines'][0]['witness']['fields']['lateness_ns'] = 2
        with self.assertRaisesRegex(ValueError, 'count or maximum decreases'):
            state_trace(encode_deadlines([base, later]), StateOutcome.FAILURE)
        earlier = copy.deepcopy(base)
        earlier['engine']['deadline_observations'] = 2
        earlier['engine']['deadlines'][0]['observations'] = 2
        reduced = copy.deepcopy(base)
        reduced['elapsed_ns'] = now + 1
        with self.assertRaisesRegex(ValueError, 'count or maximum decreases'):
            state_trace(encode_deadlines([earlier, reduced]), StateOutcome.FAILURE)
        changed = copy.deepcopy(later)
        changed['engine']['deadline_observations'] = 1
        changed['engine']['deadlines'][0].update({'observations': 1, 'max_lateness_ns': 4})
        changed['engine']['deadlines'][0]['witness']['fields']['lateness_ns'] = 4
        with self.assertRaisesRegex(ValueError, 'changes without observations'):
            state_trace(encode_deadlines([base, changed]), StateOutcome.FAILURE)

    def test_failed_archived_samples_match_existing_known_history(self):
        import copy
        import gzip
        import json
        from quic_observation.audit import StateOutcome, state_pair, state_trace
        from quic_trace_audit import diagnostics
        directory = (Path(__file__).resolve().parents[2] /
                     'moirai-quic/measurements/recovery/2026-10-14/state')
        paths = [*sorted(directory.glob('03dcc4a3-matrix-*.json.gz')),
                 *sorted(directory.glob('03dcc4a3-fairness-*.json.gz'))]
        shards = [json.loads(gzip.decompress(path.read_bytes())) for path in paths]
        flows = [flow for shard in shards for record in shard['runs']
                 for flow in record.get('flows', [record])]
        golden = json.loads(gzip.decompress(
            (directory/'03dcc4a3-audit.json.gz').read_bytes()))['engine_states']
        self.assertEqual(len(flows), len(golden))
        failed = 0
        sampled = 0
        for flow, expected in zip(flows, golden, strict=True):
            if 'error' not in flow:
                continue
            failed += 1
            actual = state_pair(diagnostics(flow, 'client'), diagnostics(flow, 'server'), False)
            self.assertEqual(actual, {role: expected[role] for role in ('client', 'server')})
            if all(actual[role]['last_protocol_observation'] == 'sample'
                   for role in ('client', 'server')):
                sampled += 1
                for role in ('client', 'server'):
                    state = actual[role]
                    self.assertLess(sum(bucket['observations'] for bucket in state['buckets']),
                                    state['observed_states'])
                    with self.assertRaisesRegex(ValueError, 'not a full snapshot'):
                        state_trace(diagnostics(flow, role))
        self.assertEqual((failed, sampled), (27, 21))

        trace = diagnostics(next(flow for flow in flows if 'error' in flow), 'client')
        decoder = json.JSONDecoder()
        invalid = [decoder.raw_decode(line.split('protocol_recovery=', 1)[1])[0]
                   for line in trace.splitlines()
                   if 'protocol recovery observation' in line]
        invalid = [copy.deepcopy(sample) for sample in invalid
                   if sample.get('observation') == 'sample' and sample.get('engine')]
        largest_bucket = 0
        latest_buckets = {}
        for sample in invalid:
            for bucket in sample['engine']['buckets']:
                largest_bucket = max(largest_bucket, bucket['observations'])
                latest_buckets[bucket['second']] = bucket
            sample['engine']['observed_states'] = largest_bucket
        self.assertEqual((len(invalid), largest_bucket,
                          sum(bucket['observations'] for bucket in latest_buckets.values())),
                         (59, 640, 3_727))
        encoded = '\n'.join('protocol recovery observation protocol_recovery=' +
                            json.dumps(sample) for sample in invalid)
        with self.assertRaisesRegex(ValueError, 'known engine count exceeds'):
            state_trace(encoded, StateOutcome.FAILURE)

    def test_failed_outcome_preserves_pacing_and_callback_checks(self):
        import copy
        import json
        from quic_observation.audit import StateOutcome, state_trace
        stderr = self.evidence['runs'][0]['reports']['diagnostics']['client_stderr']
        line = next(line for line in reversed(stderr.splitlines())
                    if 'protocol recovery observation' in line)
        sample = json.JSONDecoder().raw_decode(line.split('protocol_recovery=', 1)[1])[0]

        pacing = copy.deepcopy(sample)
        pacing['engine']['buckets'][0]['pacing_armed_observations'] = 0
        pacing['engine']['buckets'][0]['max_pacing_remaining_ns'] = 1
        encoded = 'protocol recovery observation protocol_recovery=' + json.dumps(pacing)
        with self.assertRaisesRegex(ValueError, 'pacing observation aggregate'):
            state_trace(encoded, StateOutcome.FAILURE)

        corrupted = copy.deepcopy(sample)
        corrupted['engine'].update({
            'callback_observations': 1, 'callback_buckets_dropped': 0,
            'callbacks': [],
        })
        encoded = 'protocol recovery observation protocol_recovery=' + json.dumps(corrupted)
        with self.assertRaisesRegex(ValueError, 'callback count'):
            state_trace(encoded, StateOutcome.FAILURE)

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
