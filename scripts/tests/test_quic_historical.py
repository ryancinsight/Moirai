"""Historical audit must never silently qualify current source."""
import copy
import gzip
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_observation.historical import verify_historical_provenance
from quic_observation.prepare import (verify_provenance, registry_source, patch_source,
                                      digest, diagnostic_lock)

ROOT = Path(__file__).resolve().parents[2]
REVISION = '03dcc4a31d7f793e4cb1cb7d96ff74f47033946c'


class HistoricalProvenance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='quic-historical-test-')
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name)/'repository'
        self.repository.mkdir()
        for path in ('Cargo.lock', 'scripts/quic_observation/engine.patch'):
            target = self.repository/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/path).read_bytes())
        hooks = Path(self.temporary.name)/'empty-hooks'
        hooks.mkdir()
        def git(*args):
            return subprocess.check_output(['git', '-c', 'core.autocrlf=false', '-c', f'core.hooksPath={hooks}',
                                            '-C', str(self.repository), *args],
                                           stderr=subprocess.PIPE)
        self.git = git
        git('init', '-q')
        git('add', '.')
        git('-c', 'user.name=Historical audit test', '-c', 'user.email=test@example.invalid',
            '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
        self.revision = git('rev-parse', 'HEAD').decode().strip()
        source, checksum = registry_source(ROOT)
        self.report = patch_source(source, Path(self.temporary.name)/'engine', ROOT)
        lock = (ROOT/'Cargo.lock').read_text()
        self.report.update(registry_checksum=checksum,
                           canonical_lock_sha256=digest(lock.encode()),
                           diagnostic_lock_sha256=digest(diagnostic_lock(lock).encode()))

    def test_full_historical_report_reproduces_without_mutation(self):
        original = copy.deepcopy(self.report)
        self.assertEqual(verify_historical_provenance(self.report, self.repository, self.revision),
                         {'scope': 'historical-only', 'revision': self.revision})
        self.assertEqual(self.report, original)

    def test_existing_archive_still_rejected_by_current_verifier(self):
        path = ROOT/'moirai-quic/measurements/recovery/2026-10-14/state-native.json.gz'
        report = json.loads(gzip.decompress(path.read_bytes()))['engine_observation']
        with self.assertRaisesRegex(ValueError, 'diagnostic build provenance differs'):
            verify_provenance(report, ROOT)

    def test_moving_refs_and_revision_expressions_are_rejected(self):
        for revision in ('main', 'HEAD', REVISION[:8], REVISION+'^', '--help', None):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                verify_historical_provenance(self.report, self.repository, revision)

    def test_replacement_missing_and_noncommit_objects(self):
        path = self.repository/'Cargo.lock'
        path.write_bytes(path.read_bytes()+b'\n# replacement-only\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Historical audit test', '-c', 'user.email=test@example.invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '-qm', 'replacement')
        replacement = self.git('rev-parse', 'HEAD').decode().strip()
        self.git('replace', self.revision, replacement)
        self.assertEqual(verify_historical_provenance(self.report, self.repository, self.revision)
                         ['revision'], self.revision)
        with self.assertRaises(subprocess.CalledProcessError):
            verify_historical_provenance(self.report, self.repository, '0'*40)
        blob = self.git('rev-parse', 'HEAD:Cargo.lock').decode().strip()
        with self.assertRaisesRegex(ValueError, 'not a commit'):
            verify_historical_provenance(self.report, self.repository, blob)

    def test_changed_observed_file_and_extra_field_fail(self):
        report = copy.deepcopy(self.report)
        report['observed_files_sha256']['src/connection/mod.rs'] = '0'*64
        with self.assertRaises(ValueError):
            verify_historical_provenance(report, self.repository, self.revision)
        report = copy.deepcopy(self.report)
        report['current_qualification'] = True
        with self.assertRaises(ValueError):
            verify_historical_provenance(report, self.repository, self.revision)

    def test_changed_lock_patch_inventory_and_observed_hashes_fail(self):
        for field in ('canonical_lock_sha256', 'patch_sha256', 'observed_mod_sha256',
                      'registry_checksum', 'diagnostic_lock_sha256'):
            report = copy.deepcopy(self.report)
            report[field] = '0'*64
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_historical_provenance(report, self.repository, self.revision)
        report = copy.deepcopy(self.report)
        report['changed_files'].pop()
        with self.assertRaises(ValueError):
            verify_historical_provenance(report, self.repository, self.revision)


if __name__ == '__main__':
    unittest.main()
