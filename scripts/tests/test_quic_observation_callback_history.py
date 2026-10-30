"""Value-semantic QUIC observation decoder cases."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_observation_support import cubic_fields, native_evidence


class CallbackObservationHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = native_evidence()

    def test_callback_and_blocker_totals_equal_retained_history(self):
        import copy
        import json
        from quic_observation.audit import state_trace
        callback = {'elapsed_ns': 17, 'kind': 'engine cubic callback',
                    'fields': cubic_fields()}
        blocker = {'elapsed_ns': 23, 'kind': 'engine pacing blocked',
                   'fields': {'flight_bytes': 1_200, 'bytes_to_send': 1_200,
                              'window_bytes': 12_000, 'smoothed_rtt_ns': 10_000_000,
                              'delay_ns': 37}}
        stderr = self.evidence['runs'][0]['reports']['diagnostics']['client_stderr']
        line = next(line for line in reversed(stderr.splitlines())
                    if 'protocol recovery observation' in line)
        sample = json.JSONDecoder().raw_decode(line.split('protocol_recovery=', 1)[1])[0]
        sample['engine'].update({
            'callback_observations': 277, 'callback_buckets_dropped': 0,
            'callbacks': [{'second': 0, 'callback': 'ack', 'outcome': 'app_limited',
                           'observations': 277, 'first': callback, 'last': callback}],
            'blocker_observations': 19, 'blocker_buckets_dropped': 0,
            'blockers': [{'second': 0, 'kind': 'engine pacing blocked',
                          'observations': 19, 'first': blocker, 'last': blocker}],
        })
        def trace(value):
            return 'protocol recovery observation protocol_recovery=' + json.dumps(value)
        decoded = state_trace(trace(sample))
        self.assertEqual(decoded['callback_observations'], 277)
        self.assertEqual(decoded['callbacks'][0]['first'], callback)
        self.assertEqual(decoded['blocker_observations'], 19)
        self.assertEqual(decoded['blockers'][0]['last'], blocker)
        for field, value, error in (
                ('callback_observations', 0, 'callback count'),
                ('blocker_observations', 0, 'blocker count')):
            malformed = copy.deepcopy(sample)
            malformed['engine'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                state_trace(trace(malformed))
        for field, error in (('callbacks', 'callback count'), ('blockers', 'blocker count')):
            malformed = copy.deepcopy(sample)
            malformed['engine'][field] = []
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                state_trace(trace(malformed))

    def test_callback_extensions_cannot_be_partial_or_disappear(self):
        import json
        from quic_observation.audit import state_trace
        stderr = self.evidence['runs'][0]['reports']['diagnostics']['client_stderr']
        line = next(line for line in reversed(stderr.splitlines())
                    if 'protocol recovery observation' in line)
        sample = json.JSONDecoder().raw_decode(line.split('protocol_recovery=', 1)[1])[0]
        def text(value):
            return 'protocol recovery observation protocol_recovery=' + json.dumps(value)
        for prefix, group in (('callback', 'callbacks'), ('blocker', 'blockers')):
            partial = {**sample, 'engine': {**sample['engine'],
                                           prefix+'_observations': 5_000,
                                           prefix+'_buckets_dropped': 5_000}}
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, 'incomplete'):
                state_trace(text(partial))
            complete = {**sample, 'engine': {**sample['engine'], group: [],
                                            prefix+'_observations': 0,
                                            prefix+'_buckets_dropped': 0}}
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, 'incomplete'):
                state_trace(text(complete)+'\n'+text(sample))

    def test_blocker_bucket_rejects_actual_ack_event(self):
        from quic_observation.audit import retained_buckets, state_trace
        stderr = self.evidence['runs'][0]['reports']['diagnostics']['client_stderr']
        ack = next(event for event in state_trace(stderr)['events']
                   if event['kind'] == 'engine ack received')
        bucket = {'second': ack['elapsed_ns']//10**9, 'kind': ack['kind'],
                  'observations': 1, 'first': ack, 'last': ack}
        state = {'blocker_observations': 1, 'blocker_buckets_dropped': 0, 'blockers': [bucket]}
        with self.assertRaisesRegex(ValueError, 'blocker kind differs'):
            retained_buckets(state, {'elapsed_ns': ack['elapsed_ns']}, None, None, {},
                             'blockers', 'blocker', 'blocker', ('kind',))

    def test_callback_history_rejects_changed_endpoints_and_old_insertions(self):
        import copy
        from quic_observation.audit import retained_buckets
        event = {'elapsed_ns': 17, 'kind': 'engine cubic callback', 'fields': cubic_fields()}
        bucket = {'second': 0, 'callback': 'ack', 'outcome': 'app_limited',
                  'observations': 1, 'first': event, 'last': copy.deepcopy(event)}
        previous = {'callback_observations': 1, 'callback_buckets_dropped': 0,
                    'callbacks': [bucket]}
        key = (0, 'ack', 'app_limited')
        for change, error in (('first', 'history changes'), ('last', 'before the previous')):
            state = copy.deepcopy(previous)
            state['callback_observations'] = 2
            current = state['callbacks'][0]
            current['observations'] = 2
            current['last']['elapsed_ns'] = 30
            if change == 'first':
                current['first']['fields']['bytes'] = 1300
            else:
                current['last']['elapsed_ns'] = 18
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, error):
                retained_buckets(state, {'elapsed_ns': 40}, previous, 20,
                                 {key: copy.deepcopy(bucket)}, 'callbacks', 'callback',
                                 'CUBIC callback', ('callback', 'outcome'))
        appended = copy.deepcopy(bucket)
        appended.update({'callback': 'ack', 'outcome': 'slow_start'})
        appended['first']['fields']['outcome'] = 'slow_start'
        appended['last']['fields']['outcome'] = 'slow_start'
        appended['first']['elapsed_ns'] = 18
        appended['last']['elapsed_ns'] = 18
        state = {**previous, 'callback_observations': 2,
                 'callbacks': [bucket, appended]}
        with self.assertRaisesRegex(ValueError, 'appears before the previous'):
            retained_buckets(state, {'elapsed_ns': 40}, previous, 20,
                             {key: copy.deepcopy(bucket)}, 'callbacks', 'callback',
                             'CUBIC callback', ('callback', 'outcome'))

    def test_callback_and_blocker_native_budgets_are_enforced(self):
        from quic_observation.audit import retained_buckets
        for prefix, group, limit, identity in (
                ('callback', 'callbacks', 1024, {'callback': 'ack', 'outcome': 'app_limited'}),
                ('blocker', 'blockers', 256, {'kind': 'engine pacing blocked'})):
            buckets = []
            for second in range(limit+1):
                point = {'elapsed_ns': second*10**9+17,
                         'kind': 'engine cubic callback' if prefix == 'callback' else identity['kind'],
                         'fields': cubic_fields() if prefix == 'callback' else
                         {'flight_bytes': 1200, 'bytes_to_send': 1200, 'window_bytes': 12000,
                          'smoothed_rtt_ns': 10000000, 'delay_ns': 37}}
                buckets.append({'second': second, **identity, 'observations': 1,
                                'first': point, 'last': point})
            state = {prefix+'_observations': limit, prefix+'_buckets_dropped': 0,
                     group: buckets[:limit]}
            history = {}
            retained_buckets(state, {'elapsed_ns': (limit+1)*10**9}, None, None, history,
                             group, prefix, prefix, tuple(identity))
            self.assertEqual(len(history), limit)
            self.assertEqual(list(history.values())[-1], buckets[limit-1])
            state = {**state, prefix+'_observations': limit+1, group: buckets}
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, 'capture exceeds'):
                retained_buckets(state, {'elapsed_ns': (limit+1)*10**9}, None, None, {}, group,
                                 prefix, prefix, tuple(identity))

    def test_callback_counters_and_bucket_seconds_match_native_widths(self):
        import copy
        from quic_observation.audit import retained_buckets
        empty = {'callback_observations': 0, 'callback_buckets_dropped': 0, 'callbacks': []}
        for field, value in (('callback_observations', 2**64),
                             ('callback_observations', True), ('callback_buckets_dropped', False)):
            state = {**empty, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                retained_buckets(state, {'elapsed_ns': 0}, None, None, {}, 'callbacks',
                                 'callback', 'CUBIC callback', ('callback', 'outcome'))
        event = {'elapsed_ns': 17, 'kind': 'engine cubic callback', 'fields': cubic_fields()}
        bucket = {'second': 0, 'callback': 'ack', 'outcome': 'app_limited',
                  'observations': 1, 'first': event, 'last': copy.deepcopy(event)}
        different = copy.deepcopy(bucket)
        different['last']['fields']['bytes'] += 1
        state = {**empty, 'callback_observations': 1, 'callbacks': [different]}
        with self.assertRaisesRegex(ValueError, 'singleton endpoints differ'):
            retained_buckets(state, {'elapsed_ns': 20}, None, None, {}, 'callbacks',
                             'callback', 'CUBIC callback', ('callback', 'outcome'))
        for field, value in (('second', 2**128), ('second', True), ('observations', 2**64)):
            malformed = copy.deepcopy(bucket)
            malformed[field] = value
            state = {**empty, 'callback_observations': 1, 'callbacks': [malformed]}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                retained_buckets(state, {'elapsed_ns': 20}, None, None, {}, 'callbacks',
                                 'callback', 'CUBIC callback', ('callback', 'outcome'))

    def test_callback_history_rejects_reordered_native_buckets(self):
        import copy
        from quic_observation.audit import retained_buckets
        buckets = []
        for second in (0, 1):
            point = {'elapsed_ns': second*10**9+17, 'kind': 'engine cubic callback',
                     'fields': cubic_fields()}
            buckets.append({'second': second, 'callback': 'ack', 'outcome': 'app_limited',
                            'observations': 1, 'first': point, 'last': point})
        previous = {'callback_observations': 2, 'callback_buckets_dropped': 0,
                    'callbacks': buckets}
        history = {(value['second'], 'ack', 'app_limited'): copy.deepcopy(value)
                   for value in buckets}
        state = {**previous, 'callbacks': list(reversed(buckets))}
        for prior in (None, previous):
            with self.subTest(initial=prior is None), self.assertRaisesRegex(ValueError, 'insertion order changes'):
                retained_buckets(state, {'elapsed_ns': 2*10**9}, prior, 10**9+20 if prior else None,
                                 history if prior else {}, 'callbacks', 'callback',
                                 'CUBIC callback', ('callback', 'outcome'))

    def test_capture_clock_cannot_regress_before_callback_append(self):
        import copy
        import json
        from quic_observation.audit import state_trace
        stderr = self.evidence['runs'][0]['reports']['diagnostics']['client_stderr']
        line = next(line for line in reversed(stderr.splitlines())
                    if 'protocol recovery observation' in line)
        sample = json.JSONDecoder().raw_decode(line.split('protocol_recovery=', 1)[1])[0]
        origin = (sample['elapsed_ns']//10**9+1)*10**9
        first = {'elapsed_ns': origin+25, 'kind': 'engine cubic callback',
                 'fields': cubic_fields()}
        bucket = {'second': origin//10**9, 'callback': first['fields']['callback'],
                  'outcome': first['fields']['outcome'], 'observations': 1,
                  'first': first, 'last': copy.deepcopy(first)}
        captures = []
        for elapsed in (origin+100, origin+50, origin+110):
            capture = copy.deepcopy(sample)
            capture['elapsed_ns'] = elapsed
            capture['engine'].update({'callback_observations': 1,
                                      'callback_buckets_dropped': 0,
                                      'callbacks': [copy.deepcopy(bucket)]})
            captures.append(capture)
        captures[-1]['engine']['callback_observations'] = 2
        captures[-1]['engine']['callbacks'][0]['observations'] = 2
        captures[-1]['engine']['callbacks'][0]['last']['elapsed_ns'] = origin+75
        encoded = '\n'.join('protocol recovery observation protocol_recovery=' +
                            json.dumps(capture) for capture in captures)
        with self.assertRaisesRegex(ValueError, 'protocol capture time decreases'):
            state_trace(encoded)



if __name__ == '__main__':
    unittest.main()
