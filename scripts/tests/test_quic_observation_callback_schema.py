"""Value-semantic QUIC observation point schema cases."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_observation_support import cubic_fields, native_evidence


def representative_points(evidence):
    from quic_observation.audit import state_trace
    points = {}
    for run in evidence['runs']:
        trace = state_trace(run['reports']['diagnostics']['client_stderr'])
        retained = [*trace['handshake'], *trace['events'], *trace['ptos']]
        retained.extend(value for bucket in trace['buckets']
                        for value in (bucket['first'], bucket['last']))
        points.update((value['kind'], value) for value in retained)
    points.update({
        'engine cubic callback': {
            'elapsed_ns': 17, 'kind': 'engine cubic callback',
            'fields': cubic_fields()},
        'engine congestion blocked': {
            'elapsed_ns': 19, 'kind': 'engine congestion blocked',
            'fields': {'flight_bytes': 11_000, 'bytes_to_send': 1_200,
                       'window_bytes': 12_000}},
        'engine pacing blocked': {
            'elapsed_ns': 23, 'kind': 'engine pacing blocked',
            'fields': {'flight_bytes': 1_200, 'bytes_to_send': 1_200,
                       'window_bytes': 12_000, 'smoothed_rtt_ns': 10_000_000,
                       'delay_ns': 37}},
    })
    return points


class CallbackObservationSchemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = native_evidence()
        cls.points = representative_points(cls.evidence)

    def test_callback_schema_accepts_every_native_outcome(self):
        from quic_observation.audit import point
        outcomes = {'ack': ('app_limited', 'recovery', 'slow_start', 'avoidance'),
                    'congestion': ('recovery', 'loss', 'persistent')}
        for callback, names in outcomes.items():
            for outcome in names:
                with self.subTest(callback=callback, outcome=outcome):
                    point({'elapsed_ns': 17, 'kind': 'engine cubic callback',
                           'fields': cubic_fields(callback=callback, outcome=outcome)})

    def test_callback_schema_rejects_malformed_metadata(self):
        from quic_observation.audit import point
        event = {'elapsed_ns': 17, 'kind': 'engine cubic callback',
                 'fields': cubic_fields()}
        for field in ('bytes', 'rtt_ns', 'sent_age_ns', 'lost_bytes',
                      'recovery_age_ns_before', 'recovery_age_ns_after'):
            for value in (True, -1, 2**64):
                malformed = copy.deepcopy(event)
                malformed['fields'][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    point(malformed)
        for field in ('app_limited', 'persistent', 'recovery_active_before',
                      'sent_after_recovery_before', 'recovery_active_after',
                      'sent_after_recovery_after'):
            malformed = copy.deepcopy(event)
            malformed['fields'][field] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                point(malformed)
        for field, value in (('callback', 'unknown'), ('callback', []),
                             ('outcome', 'loss'), ('outcome', []),
                             ('state_before', [0]),
                             ('state_after', [0, 0, 0, 0, 0, True]),
                             ('state_before', [0, 0, 0, 0, 0, 2**64]),
                             ('state_after', [0, 0, 2**16, 0, 0, 0])):
            malformed = copy.deepcopy(event)
            malformed['fields'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                point(malformed)
        missing = copy.deepcopy(event)
        missing['fields'].pop('rtt_ns')
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            point(missing)

    def test_point_schema_is_closed_for_every_native_kind(self):
        from quic_observation.audit import POINT_KINDS, point
        self.assertEqual(set(self.points), POINT_KINDS)
        for kind, value in self.points.items():
            point(value)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'point differs'):
                point({**value, 'unexpected': 1})
            for invalid in (True, -1, 2**128):
                malformed = {**value, 'elapsed_ns': invalid}
                with self.subTest(kind=kind, elapsed=invalid), self.assertRaisesRegex(
                        ValueError, 'invalid engine time'):
                    point(malformed)
        for malformed, error in (([], 'point differs'),
                                 ({'elapsed_ns': 1, 'kind': 'unknown', 'fields': {}},
                                  'unknown engine point kind'),
                                 ({'elapsed_ns': 1, 'kind': 'engine ack received',
                                   'fields': []}, 'invalid engine point fields')):
            with self.subTest(value=malformed), self.assertRaisesRegex(ValueError, error):
                point(malformed)

    def test_native_point_integer_fields_match_source_widths(self):
        from quic_observation.audit import point
        state = self.points['engine recovery state']
        scalar_fields = ('flight_bytes', 'flight_packets', 'window_bytes',
                         'connection_max_data', 'connection_data_sent',
                         'buffered_data', 'send_window')
        row_widths = {'spaces': 12, 'queued': 3, 'sent': 4, 'acked': 2,
                      'streams': 5, 'timers': 2}
        for field in scalar_fields:
            for invalid in (True, -1, 2**64):
                malformed = copy.deepcopy(state)
                malformed['fields'][field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    point(malformed)
        for field, width in row_widths.items():
            for invalid in (True, -1, 2**64):
                malformed = copy.deepcopy(state)
                malformed['fields'][field] = [[0] * (width - 1) + [invalid]]
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    point(malformed)
        for invalid in (True, -1, 2**64):
            malformed = copy.deepcopy(state)
            malformed['fields']['pto'] = [invalid]
            with self.subTest(field='pto', value=invalid), self.assertRaises(ValueError):
                point(malformed)
        for kind, fields in (('engine ack received', ('ranges',)),
                             ('engine crypto received', ('offset', 'len')),
                             ('engine packet sent', ('pn', 'size', 'ranges'))):
            for field in fields:
                malformed = copy.deepcopy(self.points[kind])
                malformed['fields'][field] = ([[True, 0]] if field == 'ranges' else True)
                with self.subTest(kind=kind, field=field), self.assertRaises(ValueError):
                    point(malformed)

    def test_blocker_schema_rejects_wrong_fields_and_predicates(self):
        from quic_observation.audit import point
        for kind in ('engine congestion blocked', 'engine pacing blocked'):
            event = self.points[kind]
            point(event)
            for field in event['fields']:
                for value in (True, -1, 2**64):
                    malformed = copy.deepcopy(event)
                    malformed['fields'][field] = value
                    with self.subTest(kind=kind, field=field, value=value), self.assertRaises(
                            ValueError):
                        point(malformed)
            malformed = copy.deepcopy(event)
            malformed['fields']['window_bytes'] = (20_000 if kind ==
                                                    'engine congestion blocked' else 0)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'predicate'):
                point(malformed)


if __name__ == '__main__':
    unittest.main()
