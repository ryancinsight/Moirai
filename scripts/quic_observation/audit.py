"""Validate actual numeric engine states without inferring continuous durations."""
from enum import Enum, auto
import json

from quic_recovery_audit import require

STATE_FIELDS = {'reason', 'flight_bytes', 'flight_packets', 'window_bytes',
                'connection_max_data', 'connection_data_sent', 'buffered_data', 'send_window',
                'engine_handshake', 'tls_handshake', 'spaces', 'queued', 'sent', 'acked',
                'streams', 'timers', 'pto'}
CUBIC_CALLBACK_FIELDS = {
    'callback', 'outcome', 'bytes', 'app_limited', 'rtt_ns', 'sent_age_ns',
    'persistent', 'lost_bytes', 'recovery_active_before', 'recovery_age_ns_before',
    'sent_after_recovery_before', 'recovery_active_after', 'recovery_age_ns_after',
    'sent_after_recovery_after', 'state_before', 'state_after'}
POINT_KINDS = {'engine recovery state', 'engine ack received',
               'engine crypto received', 'engine packet sent',
               'engine cubic callback', 'engine congestion blocked',
               'engine pacing blocked'}
# JSON limits match Capture in the pinned native collector.
BUCKET_LIMITS = {'callback': 1_024, 'blocker': 256}


class StateOutcome(Enum):
    """Select the completeness required from a process state history."""

    SUCCESS = auto()
    FAILURE = auto()


def within_bits(value, width):
    return type(value) is int and 0 <= value < 1 << width


def rows(value, width):
    require(isinstance(value, list), 'engine rows are absent')
    require(all(isinstance(row, list) and len(row) == width and
                all(within_bits(n, 64) for n in row) for row in value),
            'invalid numeric engine row')


def cubic_callback(fields):
    """Check serialized callback metadata; native tests own CUBIC arithmetic."""
    require(set(fields) == CUBIC_CALLBACK_FIELDS, 'CUBIC callback metadata differs')
    outcomes = {'ack': {'app_limited', 'recovery', 'slow_start', 'avoidance'},
                'congestion': {'recovery', 'loss', 'persistent'}}
    callback, outcome = fields['callback'], fields['outcome']
    require(type(callback) is str and callback in outcomes and
            type(outcome) is str and outcome in outcomes[callback],
            'unknown CUBIC callback or outcome')
    for key in ('app_limited', 'persistent', 'recovery_active_before',
                'sent_after_recovery_before', 'recovery_active_after',
                'sent_after_recovery_after'):
        require(type(fields[key]) is bool, 'invalid CUBIC callback flag')
    for key in ('bytes', 'rtt_ns', 'sent_age_ns', 'lost_bytes',
                'recovery_age_ns_before', 'recovery_age_ns_after'):
        require(within_bits(fields[key], 64), 'invalid CUBIC callback integer')
    for key in ('state_before', 'state_after'):
        rows([fields[key]], 6)
        require(fields[key][2] < 1 << 16, 'CUBIC MTU exceeds source representation')


def point(value):
    require(type(value) is dict and set(value) == {'elapsed_ns', 'kind', 'fields'},
            'engine point differs')
    require(within_bits(value['elapsed_ns'], 128), 'invalid engine time')
    require(type(value['kind']) is str and value['kind'] in POINT_KINDS,
            'unknown engine point kind')
    require(type(value['fields']) is dict, 'invalid engine point fields')
    f = value['fields']
    if value['kind'] != 'engine recovery state':
        if value['kind'] == 'engine cubic callback':
            cubic_callback(f)
        elif value['kind'] == 'engine congestion blocked':
            require(set(f) == {'flight_bytes', 'bytes_to_send', 'window_bytes'},
                    'congestion blocker metadata differs')
            require(all(within_bits(f[key], 64) for key in f) and
                    f['flight_bytes'] + f['bytes_to_send'] >= f['window_bytes'],
                    'congestion blocker predicate is false')
        elif value['kind'] == 'engine pacing blocked':
            require(set(f) == {'flight_bytes', 'bytes_to_send', 'window_bytes',
                               'smoothed_rtt_ns', 'delay_ns'},
                    'pacing blocker metadata differs')
            require(all(within_bits(f[key], 64) for key in f) and
                    f['bytes_to_send'] > 0 and f['window_bytes'] > 0 and
                    f['smoothed_rtt_ns'] > 0 and f['window_bytes'] < 1 << 32 and
                    f['flight_bytes'] + f['bytes_to_send'] < f['window_bytes'],
                    'pacing blocker predicate is false')
        elif value['kind'] == 'engine ack received':
            require(f.get('space') in ('Initial', 'Handshake', 'Data'), 'invalid frame space')
            require(set(f) == {'space', 'ranges'}, 'ACK metadata differs')
            rows(f['ranges'], 2)
            require(all(start <= end for start, end in f['ranges']), 'reversed ACK range')
        elif value['kind'] == 'engine crypto received':
            require(f.get('space') in ('Initial', 'Handshake', 'Data'), 'invalid frame space')
            require(set(f) == {'space', 'offset', 'len'}, 'CRYPTO metadata differs')
            require(all(within_bits(f[key], 64) for key in ('offset', 'len')),
                    'invalid CRYPTO range')
        elif value['kind'] == 'engine packet sent':
            require(f.get('space') in ('Initial', 'Handshake', 'Data'), 'invalid frame space')
            require(set(f) == {'space', 'pn', 'size', 'ack_eliciting', 'ranges'}, 'packet metadata differs')
            rows(f['ranges'], 2)
            require(within_bits(f['pn'], 64) and within_bits(f['size'], 64) and
                    type(f['ack_eliciting']) is bool, 'invalid sent packet')
        return
    require(set(f) == STATE_FIELDS, 'engine state inventory differs')
    for key in ('engine_handshake', 'tls_handshake'):
        require(type(f[key]) is bool, 'invalid handshake flag')
    for key in ('flight_bytes', 'flight_packets', 'window_bytes', 'connection_max_data',
                'connection_data_sent', 'buffered_data', 'send_window'):
        require(within_bits(f[key], 64), 'invalid engine integer')
    require(f['reason'] in ('driver', 'pto'), 'unknown engine snapshot boundary')
    for key, width in (('spaces', 12), ('queued', 3), ('sent', 4), ('acked', 2),
                       ('streams', 5), ('timers', 2)):
        rows(f[key], width)
    require([r[0] for r in f['spaces']] == [0, 1, 2], 'missing/duplicate packet space')
    require(sum(r[7] for r in f['spaces']) == f['flight_bytes'], 'cross-space flight-byte mismatch')
    require(sum(r[8] for r in f['spaces']) == f['flight_packets'], 'cross-space flight-packet mismatch')
    require(f['connection_data_sent'] <= f['connection_max_data'], 'connection exceeds peer credit')
    require(f['buffered_data'] <= f['send_window'], 'send-buffer budget exceeded')
    require(len({r[0] for r in f['streams']}) == len(f['streams']), 'duplicate stream state')
    require(all(r[1] <= r[2] and r[3] in (0, 1) and r[4] in (0, 1) for r in f['streams']),
            'invalid stream credit/flag')
    require(all(r[0] < 3 for k in ('queued', 'sent', 'acked') for r in f[k]), 'invalid space reference')
    require(all(offset + size <= f['spaces'][space][4] for space, offset, size in f['queued']) and
            all(offset + size <= f['spaces'][space][4] for space, _, offset, size in f['sent']),
            'CRYPTO range exceeds generated bytes')
    require(all(r[0] < 9 for r in f['timers']), 'invalid timer kind')
    require(isinstance(f['pto'], list) and len(f['pto']) <= 1 and
            all(within_bits(n, 64) and n < 3 for n in f['pto']),
            'invalid selected PTO space')
    require(not f['pto'] or f['reason'] == 'pto' or any(r[0] == 0 for r in f['timers']),
            'PTO candidate without an armed loss timer')


def retained_buckets(state, sample, previous, previous_elapsed, history, group, prefix,
                     label, identity_fields):
    observations = f'{prefix}_observations'
    require(within_bits(state.get(observations), 64) and
            type(state.get(f'{prefix}_buckets_dropped')) is int and
            state[f'{prefix}_buckets_dropped'] == 0 and
            isinstance(state.get(group), list) and len(state[group]) <= BUCKET_LIMITS[prefix],
            f'{label} capture exceeds its bound')
    current = []
    total = 0
    last_first = 0
    for bucket in state[group]:
        require(set(bucket) == {'second', 'observations', 'first', 'last', *identity_fields},
                f'{label} bucket differs')
        require(within_bits(bucket['second'], 128), f'invalid {label} bucket second')
        first, last = bucket['first'], bucket['last']
        point(first); point(last)
        require(first['elapsed_ns'] <= last['elapsed_ns'] <= sample['elapsed_ns'] and
                first['elapsed_ns']//10**9 == last['elapsed_ns']//10**9 == bucket['second'],
                f'invalid {label} bucket time')
        require(first['elapsed_ns'] >= last_first, f'{label} insertion order changes')
        last_first = first['elapsed_ns']
        identity = tuple(bucket[field] for field in identity_fields)
        require(prefix != 'blocker' or identity[0] in
                ('engine congestion blocked', 'engine pacing blocked'),
                'blocker kind differs')
        for retained in (first, last):
            require(retained['kind'] == ('engine cubic callback' if prefix == 'callback'
                                         else identity[0]),
                    f'{label} endpoint kind differs')
            observed = tuple(retained['fields'].get(field, retained.get(field))
                             for field in identity_fields)
            require(identity == observed, f'{label} bucket identity differs')
        require(within_bits(bucket['observations'], 64) and bucket['observations'] > 0,
                f'{label} bucket identity differs')
        require(bucket['observations'] != 1 or first == last,
                f'{label} singleton endpoints differ')
        key = (bucket['second'], *identity)
        require(key not in current, f'duplicate {label} bucket')
        current.append(key)
        total += bucket['observations']
        old = history.get(key)
        if old:
            require(old['first'] == first and old['observations'] <= bucket['observations'] and
                    old['last']['elapsed_ns'] <= last['elapsed_ns'],
                    f'{label} bucket history changes')
            if old['observations'] == bucket['observations']:
                require(old == bucket, f'{label} bucket changes without an observation')
            else:
                require(last['elapsed_ns'] >= previous_elapsed,
                        f'{label} adds an observation before the previous capture')
        elif previous_elapsed is not None:
            require(first['elapsed_ns'] >= previous_elapsed,
                    f'{label} appears before the previous capture')
        history[key] = bucket
    require(total == state[observations], f'{label} count differs from retained history')
    if previous:
        prior = [(bucket['second'], *(bucket[field] for field in identity_fields))
                 for bucket in previous[group]]
        require(current[:len(prior)] == prior, f'{label} insertion order changes')
        require(state[observations] >= previous[observations], f'{label} count decreases')


def state_bucket_values(point_value):
    fields = point_value['fields']
    timers = {row[0]: row[1] for row in fields['timers']}
    return (fields['flight_bytes'], fields['window_bytes'],
            fields['connection_max_data'] - fields['connection_data_sent'],
            timers.get(6))


def state_bucket(bucket, sample, old):
    expected = {'second', 'observations', 'first', 'last', 'min_flight_bytes',
                'max_flight_bytes', 'min_window_bytes', 'max_window_bytes',
                'min_connection_credit', 'max_connection_credit',
                'max_pacing_remaining_ns', 'pacing_armed_observations'}
    require(set(bucket) == expected, 'engine bucket differs')
    require(within_bits(bucket['second'], 128), 'invalid engine bucket second')
    first, last = bucket['first'], bucket['last']
    point(first); point(last)
    require(first['elapsed_ns'] <= last['elapsed_ns'] <= sample['elapsed_ns'],
            'invalid bucket time')
    require(first['elapsed_ns']//10**9 == last['elapsed_ns']//10**9 == bucket['second'],
            'engine bucket crosses observation interval')
    count = bucket['observations']
    require(within_bits(count, 64) and 0 < count <= sample['engine']['observed_states'],
            'invalid bucket count')
    require(count != 1 or first == last, 'engine singleton endpoints differ')
    retained = (first,) if count == 1 else (first, last)
    endpoints = [state_bucket_values(value) for value in retained]
    names = ('flight_bytes', 'window_bytes', 'connection_credit')
    for index, name in enumerate(names):
        values = [value[index] for value in endpoints]
        lower, upper = bucket['min_'+name], bucket['max_'+name]
        require(within_bits(lower, 64) and within_bits(upper, 64) and
                lower <= min(values) <= max(values) <= upper,
                'engine bucket extrema exclude a retained point')
        if count <= 2:
            require((lower, upper) == (min(values), max(values)),
                    'engine endpoint extrema differ')
    armed = bucket['pacing_armed_observations']
    maximum = bucket['max_pacing_remaining_ns']
    require(within_bits(armed, 64) and within_bits(maximum, 64) and
            armed <= count and (armed != 0 or maximum == 0),
            'invalid pacing observation aggregate')
    endpoint_delays = [value[3] for value in endpoints]
    retained_armed = sum(value is not None for value in endpoint_delays)
    retained_maximum = max((value or 0 for value in endpoint_delays), default=0)
    require(retained_armed <= armed and retained_maximum <= maximum,
            'engine pacing aggregate excludes a retained point')
    if count <= 2:
        require(armed == retained_armed and maximum == retained_maximum,
                'engine endpoint pacing aggregate differs')
    require(armed <= retained_armed + count - len(endpoints) and
            (armed > retained_armed or maximum == retained_maximum),
            'engine pacing aggregate contradicts retained endpoints')
    if old is None:
        return
    require(old['first'] == first and old['observations'] <= count and
            old['last']['elapsed_ns'] <= last['elapsed_ns'],
            'engine bucket history changes')
    delta = count - old['observations']
    if delta == 0:
        require(old == bucket, 'engine bucket changes without an observation')
        return
    for name in names:
        require(bucket['min_'+name] <= old['min_'+name] and
                bucket['max_'+name] >= old['max_'+name],
                'engine bucket extrema contract')
    require(old['max_pacing_remaining_ns'] <= maximum and
            0 <= armed - old['pacing_armed_observations'] <= delta,
            'engine pacing aggregate contract')
    if delta == 1:
        latest = state_bucket_values(last)
        for index, name in enumerate(names):
            require(bucket['min_'+name] == min(old['min_'+name], latest[index]) and
                    bucket['max_'+name] == max(old['max_'+name], latest[index]),
                    'engine single-observation extrema differ')
        delay = latest[3]
        require(armed == old['pacing_armed_observations'] + (delay is not None) and
                maximum == max(old['max_pacing_remaining_ns'], delay or 0),
                'engine single-observation pacing aggregate differs')


def state_trace(text, outcome: StateOutcome = StateOutcome.SUCCESS):
    if type(outcome) is not StateOutcome:
        raise TypeError('outcome must be StateOutcome')
    samples = []
    buckets = {}
    previous = None
    previous_elapsed = None
    callback_buckets = {}
    blocker_buckets = {}
    full_bucket_keys = []
    final_full_snapshot = False
    for line in (text or '').splitlines():
        if 'protocol recovery observation' not in line:
            continue
        sample, _ = json.JSONDecoder().raw_decode(line.split('protocol_recovery=', 1)[1])
        state = sample.get('engine')
        if state is None:
            continue
        require(within_bits(sample.get('elapsed_ns'), 128),
                'invalid protocol capture time')
        if previous_elapsed is not None:
            require(sample['elapsed_ns'] >= previous_elapsed,
                    'protocol capture time decreases')
        require(within_bits(state['observed_states'], 64) and state['observed_states'] > 0,
                'no actual engine states')
        for key in ('handshake', 'events', 'ptos'):
            entries = state[key]
            require(all(a['elapsed_ns'] <= b['elapsed_ns'] for a, b in zip(entries, entries[1:])),
                    'engine event clocks decrease')
            if previous:
                require(entries[:len(previous[key])] == previous[key], 'engine event history changes')
            for entry in entries:
                point(entry)
                require(entry['elapsed_ns'] <= sample['elapsed_ns'], 'engine event exceeds observation')
        if previous:
            require(state['observed_states'] >= previous['observed_states'], 'engine count decreases')
        current_bucket_keys = []
        for bucket in state['buckets']:
            key = bucket.get('second')
            require(key not in current_bucket_keys, 'duplicate engine bucket')
            current_bucket_keys.append(key)
            state_bucket(bucket, sample, buckets.get(key))
            buckets[key] = bucket
        final_full_snapshot = sample.get('observation') != 'sample'
        if final_full_snapshot:
            require(current_bucket_keys == sorted(current_bucket_keys),
                    'engine bucket insertion order changes')
            require(current_bucket_keys[:len(full_bucket_keys)] == full_bucket_keys,
                    'engine bucket insertion order changes')
            require(sum(bucket['observations'] for bucket in state['buckets']) ==
                    state['observed_states'], 'engine count differs from full bucket history')
            require(set(current_bucket_keys) == set(buckets),
                    'full engine capture omits sampled history')
            full_bucket_keys = current_bucket_keys
        else:
            require(len(state['buckets']) <= 1,
                    'sample retains more than its latest bucket')
        require(sum(bucket['observations'] for bucket in buckets.values()) <=
                state['observed_states'], 'known engine count exceeds cumulative history')
        for group, prefix, label, identity in (
                ('callbacks', 'callback', 'CUBIC callback', ('callback', 'outcome')),
                ('blockers', 'blocker', 'blocker', ('kind',))):
            keys = {group, prefix+'_observations', prefix+'_buckets_dropped'}
            prior = previous if previous and keys.intersection(previous) else None
            if keys.intersection(state) or prior:
                require(keys.issubset(state), f'{label} extension is incomplete')
                retained_buckets(state, sample, prior, previous_elapsed,
                                 callback_buckets if prefix == 'callback' else blocker_buckets,
                                 group, prefix, label, identity)
        previous = state
        previous_elapsed = sample['elapsed_ns']
        samples.append(sample)
    require(bool(samples), 'engine state capture is absent')
    require(bool(previous['handshake']), 'engine handshake capture is absent')
    require(outcome is StateOutcome.FAILURE or final_full_snapshot,
            'final engine capture is not a full snapshot')
    ordered = sorted([*previous['handshake'], *previous['ptos'],
                      *[p for b in buckets.values() for p in (b['first'], b['last'])]],
                     key=lambda p: p['elapsed_ns'])
    for before, after in zip(ordered, ordered[1:]):
        require(all(a[k] >= b[k] for a, b in zip(after['fields']['spaces'], before['fields']['spaces'])
                    for k in (2, 4, 5)), 'packet or CRYPTO offset decreases')
    sent = {space: {p['fields']['pn'] for p in previous['events']
                    if p['kind'] == 'engine packet sent' and p['fields']['space'] == space}
            for space in ('Initial', 'Handshake')}
    for event in previous['events']:
        if event['kind'] == 'engine ack received':
            numbers = sent[event['fields']['space']]
            require(all(sum(lo <= pn <= hi for pn in numbers) == hi - lo + 1
                        for lo, hi in event['fields']['ranges']), 'ACK refers to unobserved packet')
    if final_full_snapshot:
        require({bucket['second'] for bucket in previous['buckets']} == set(buckets),
                'final engine capture omits sampled history')
    known_buckets = (previous['buckets'] if final_full_snapshot else
                     [buckets[key] for key in sorted(buckets)])
    result = {**previous, 'buckets': known_buckets,
              'last_protocol_observation': samples[-1]['observation']}
    if 'callbacks' in previous:
        result['callbacks'] = list(callback_buckets.values())
    if 'blockers' in previous:
        result['blockers'] = list(blocker_buckets.values())
    return result


def state_pair(client, server, success):
    outcome = StateOutcome.SUCCESS if success else StateOutcome.FAILURE
    states = {role: state_trace(text, outcome) for role, text in (('client', client), ('server', server))}
    if success:
        for role, peer in (('client', 'server'), ('server', 'client')):
            local = states[role]['buckets'][-1]['last']['fields']['spaces']
            remote = states[peer]['buckets'][-1]['last']['fields']['spaces']
            require(all(local[n][4] == remote[n][5] for n in (0, 1)),
                    'complete handshake CRYPTO generation/consumption differs between peers')
    return states
