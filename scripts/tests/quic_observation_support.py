"""Shared native QUIC observation fixtures."""
import functools
import gzip
import json
from pathlib import Path


@functools.cache
def native_evidence():
    path = (Path(__file__).resolve().parents[2] /
            'moirai-quic/measurements/recovery/2026-10-14/state-native.json.gz')
    return json.loads(gzip.decompress(path.read_bytes()))


def cubic_fields(**changes):
    """Return one complete callback schema with app-limited ACK defaults."""
    fields = {
        'callback': 'ack', 'outcome': 'app_limited', 'bytes': 1_200,
        'app_limited': True, 'rtt_ns': 10, 'sent_age_ns': 20,
        'persistent': False, 'lost_bytes': 0,
        'recovery_active_before': False, 'recovery_age_ns_before': 0,
        'sent_after_recovery_before': False, 'recovery_active_after': False,
        'recovery_age_ns_after': 0, 'sent_after_recovery_after': False,
        'state_before': [12_000, 24_000, 1_200, 0, 0, 0],
        'state_after': [12_000, 24_000, 1_200, 0, 0, 0],
    }
    fields.update(changes)
    return fields
