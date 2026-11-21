"""Audit historical provenance only; never qualifies the current source tree.

No implicit fallback is provided. Callers must supply the full immutable measured
commit and an existing report. No archive or current qualification is rewritten.
"""
import re
from pathlib import Path
import subprocess
import tempfile

from .prepare import digest, verify_provenance


def verify_historical_provenance(report, repository, revision):
    """Reconstruct historical inputs and verify the complete report, or fail.

The commit must already exist locally. This function never fetches, executes
historical code, or substitutes historical evidence into current verification.
"""
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('historical revision must be a full immutable commit SHA')

    def git(*arguments):
        return subprocess.check_output(
            ['git', '--no-replace-objects', '-C', str(repository), *arguments],
            stderr=subprocess.PIPE)

    if git('cat-file', '-t', revision).strip() != b'commit':
        raise ValueError('historical revision is not a commit')
    inputs = {
        'Cargo.lock': 'canonical_lock_sha256',
        'scripts/quic_observation/engine.patch': 'patch_sha256',
    }
    with tempfile.TemporaryDirectory(prefix='quic-historical-audit-') as temporary:
        root = Path(temporary)
        for path, field in inputs.items():
            raw = git('show', f'{revision}:{path}')
            if digest(raw) != report.get(field):
                raise ValueError(f'historical input hash mismatch: {path}')
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        verify_provenance(report, root)
    return {'scope': 'historical-only', 'revision': revision}
