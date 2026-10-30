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
