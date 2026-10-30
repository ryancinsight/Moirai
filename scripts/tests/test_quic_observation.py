"""Provenance checks use the real pinned registry source, never a stand-in."""
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_observation.prepare import patch_source, registry_source


def git_repository(path):
    path.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=path, check=True)
    subprocess.run(['git', 'config', 'user.email', 'observer@example.invalid'],
                   cwd=path, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Observer'], cwd=path, check=True)
    return path


def git_tree(repository, entries):
    records = []
    for mode, name, contents in entries:
        blob = subprocess.check_output(
            ['git', 'hash-object', '-w', '--stdin'], cwd=repository, input=contents,
        ).decode().strip()
        records.append(f'{mode} blob {blob}\t{name}\0')
    return subprocess.check_output(
        ['git', 'mktree', '-z'], cwd=repository,
        input=''.join(records).encode(),
    ).decode().strip()


def source_identity(root):
    return (Path(os.environ.get('QUIC_HISTORY_REPOSITORY', root)),
            os.environ.get('QUIC_SOURCE_TREE', 'HEAD'))


class ObservationProvenance(unittest.TestCase):
    def test_archived_source_preserves_confined_links_and_rejects_graph_escapes(self):
        from quic_observation.prepare import archived_source, validate_archive_links
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            repository = git_repository(temporary/'history')
            targets = {'direct': './target', 'normalized': 'dir/../target'}
            tree = git_tree(repository, [
                ('100644', 'target', b'target\n'),
                *((('120000', name, target.encode())
                   for name, target in targets.items())),
            ])
            archive = subprocess.check_output(
                ['git', '--no-replace-objects', 'archive', '--format=tar', tree],
                cwd=repository,
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
                members = {member.name: member for member in contents.getmembers()}
                for name, target in targets.items():
                    self.assertEqual(members[name].linkname, target)
                    self.assertEqual(
                        tarfile.data_filter(members[name], temporary/'filtered').linkname,
                        'target',
                    )
            validate_archive_links(targets)
            unsafe = ({'link': '../outside'}, {'link': '/outside'},
                      {'link': 'C:/outside'}, {'link': '..\\outside'},
                      {'alias': '.', 'escape': 'alias/../outside'},
                      {'a': 'b', 'b': 'a'})
            for links in unsafe:
                with self.subTest(links=links):
                    with self.assertRaisesRegex(ValueError, 'escapes source|cycle'):
                        validate_archive_links(links)
                    rejected = git_tree(repository, [
                        *((('120000', name, target.encode())
                           for name, target in links.items())),
                    ])
                    with self.assertRaisesRegex(ValueError, 'escapes source|cycle'):
                        archived_source(repository, rejected, temporary/f'rejected-{len(links)}')
            probe = temporary/'probe'
            try:
                os.symlink('target', probe)
            except OSError:
                with self.assertRaisesRegex(ValueError, 'cannot materialize'):
                    archived_source(repository, tree, temporary/'unsupported')
            else:
                probe.unlink()
                archived_source(repository, tree, temporary/'source')
                for name, target in targets.items():
                    self.assertEqual(os.readlink(temporary/'source'/name), target)

    def test_archived_source_ignores_replacements_and_rejects_case_collisions(self):
        from quic_observation.prepare import archived_source
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            repository = git_repository(temporary/'history')
            original = git_tree(repository, [('100644', 'identity.txt', b'original\n')])
            replacement = git_tree(repository, [('100644', 'identity.txt', b'replacement\n')])
            subprocess.run(['git', 'replace', original, replacement],
                           cwd=repository, check=True)
            self.assertEqual(subprocess.check_output(
                ['git', 'show', f'{original}:identity.txt'], cwd=repository), b'replacement\n')
            observed, gitlinks = archived_source(repository, original, temporary/'source')
            self.assertEqual((observed, gitlinks), (original, []))
            self.assertEqual((temporary/'source/identity.txt').read_bytes(), b'original\n')
            collision = git_tree(repository, [
                ('100644', 'File.txt', b'identical\n'),
                ('100644', 'file.txt', b'identical\n'),
            ])
            with self.assertRaisesRegex(ValueError, 'case-folding path collision'):
                archived_source(repository, collision, temporary/'collision')

    def test_prepare_uses_exact_tree_instead_of_mutable_caller_files(self):
        import json
        from quic_observation.prepare import prepare
        root = Path(__file__).resolve().parents[2]
        history, revision = source_identity(root)
        tracked = 'moirai-quic/examples/quic_load.rs'
        expected = subprocess.check_output(
            ['git', 'show', f'{revision}:{tracked}'], cwd=history)
        tree = subprocess.check_output(
            ['git', 'rev-parse', f'{revision}^{{tree}}'], cwd=history, text=True).strip()
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            mutable = temporary/'mutable-caller'
            changed = mutable/tracked
            changed.parent.mkdir(parents=True)
            changed.write_bytes(b'modified caller source\n')
            hostile = temporary/'hostile.attributes'
            hostile.write_text(f'{tracked} export-ignore\n')
            config = {'GIT_CONFIG_COUNT': '1',
                      'GIT_CONFIG_KEY_0': 'core.attributesFile',
                      'GIT_CONFIG_VALUE_0': str(hostile)}
            prior = {key: os.environ.get(key) for key in config}
            os.environ.update(config)
            output = temporary/'prepared'
            try:
                prepare(mutable, output, history, revision)
            finally:
                for key, value in prior.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            self.assertEqual((output/'workspace'/tracked).read_bytes(), expected)
            self.assertNotEqual((output/'workspace'/tracked).read_bytes(), changed.read_bytes())
            provenance = json.loads((output/'provenance.json').read_text())
            self.assertEqual((provenance['source_tree'], provenance['source_gitlinks']),
                             (tree, []))

    def test_exact_shallow_tree_and_moving_symbolic_claim(self):
        from quic_observation.prepare import (diagnostic_lock, digest, patch_source,
                                              provenance_inputs, verify_provenance)
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            repository = git_repository(temporary/'history')
            patch = repository/'scripts/quic_observation/engine.patch'
            patch.parent.mkdir(parents=True)
            shutil.copyfile(root/'scripts/quic_observation/engine.patch', patch)
            shutil.copyfile(root/'Cargo.lock', repository/'Cargo.lock')
            identity = repository/'identity.txt'
            identity.write_bytes(b'first\n')
            subprocess.run(['git', 'add', 'Cargo.lock', 'identity.txt',
                            'scripts/quic_observation/engine.patch'],
                           cwd=repository, check=True)
            subprocess.run(['git', 'commit', '-qm', 'first'], cwd=repository, check=True)
            first_tree = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD^{tree}'], cwd=repository, text=True).strip()
            exact_inputs = (patch.read_bytes(), (repository/'Cargo.lock').read_text())
            source, checksum = registry_source(root, temporary/'registry')
            report = patch_source(source, temporary/'engine', root)
            original = (root/'Cargo.lock').read_text()
            report.update({'registry_checksum': checksum,
                           'canonical_lock_sha256': digest(original.encode()),
                           'diagnostic_lock_sha256': digest(diagnostic_lock(original).encode()),
                           'source_tree': first_tree, 'source_gitlinks': []})
            verify_provenance(report, root, repository)
            shallow = temporary/'shallow'
            subprocess.run(['git', 'clone', '-q', '--depth=1', repository.as_uri(), shallow],
                           check=True)
            self.assertEqual(provenance_inputs(report, shallow, shallow), exact_inputs)
            symbolic = {**report, 'source_tree': 'HEAD'}
            for contents in (b'first\n', b'second\n'):
                self.assertEqual(subprocess.check_output(
                    ['git', 'show', 'HEAD:identity.txt'], cwd=repository), contents)
                with self.assertRaisesRegex(ValueError, 'provenance differs'):
                    verify_provenance(symbolic, root, repository)
                if contents == b'first\n':
                    identity.write_bytes(b'second\n')
                    subprocess.run(['git', 'add', 'identity.txt'], cwd=repository, check=True)
                    subprocess.run(['git', 'commit', '-qm', 'second'],
                                   cwd=repository, check=True)
            verify_provenance(report, root, repository)

    def test_provenance_finds_pair_created_by_lock_only_commit(self):
        from quic_observation.prepare import digest, provenance_inputs
        with tempfile.TemporaryDirectory() as temporary:
            repository = git_repository(Path(temporary)/'history')
            patch = repository/'scripts/quic_observation/engine.patch'
            patch.parent.mkdir(parents=True)
            patch.write_bytes(b'patch-one\n')
            lock = repository/'Cargo.lock'
            lock.write_text('lock-one\n')
            subprocess.run(['git', 'add', 'scripts/quic_observation/engine.patch',
                            'Cargo.lock'], cwd=repository, check=True)
            subprocess.run(['git', 'commit', '-qm', 'initial pair'],
                           cwd=repository, check=True)
            lock.write_text('lock-two\n')
            subprocess.run(['git', 'add', 'Cargo.lock'], cwd=repository, check=True)
            subprocess.run(['git', 'commit', '-qm', 'lock only'],
                           cwd=repository, check=True)
            patch.write_bytes(b'patch-three\n')
            subprocess.run(['git', 'add', 'scripts/quic_observation/engine.patch'],
                           cwd=repository, check=True)
            subprocess.run(['git', 'commit', '-qm', 'later patch'],
                           cwd=repository, check=True)
            report = {'patch_sha256': digest(b'patch-one\n'),
                      'canonical_lock_sha256': digest(b'lock-two\n')}
            self.assertEqual(provenance_inputs(report, repository, repository),
                             (b'patch-one\n', 'lock-two\n'))

    def test_actual_pinned_source_accepts_only_the_read_only_patch(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source, checksum = registry_source(root, temporary/'registry')
            self.assertEqual(len(checksum), 64)
            original = (source/'src/connection/mod.rs').read_bytes()
            report = patch_source(source, Path(temporary)/'engine', root)
            history, revision = source_identity(root)
            tree = subprocess.check_output(
                ['git', 'rev-parse', f'{revision}^{{tree}}'], cwd=history, text=True).strip()
            from quic_observation.prepare import diagnostic_lock, digest, verify_provenance
            original_lock = (root/'Cargo.lock').read_text()
            report.update({'registry_checksum': checksum,
                           'canonical_lock_sha256': digest(original_lock.encode()),
                           'diagnostic_lock_sha256': digest(diagnostic_lock(original_lock).encode()),
                           'source_tree': tree, 'source_gitlinks': []})
            self.assertEqual(set(report['changed_files']), {
                'src/connection/mod.rs', 'src/connection/observation.rs',
                'src/connection/packet_builder.rs', 'src/connection/streams/state.rs',
                'src/congestion/cubic.rs', 'src/tests/mod.rs'})
            self.assertEqual(report['pristine_mod_sha256'], hashlib.sha256(original).hexdigest())
            self.assertNotEqual(report['observed_mod_sha256'], report['pristine_mod_sha256'])
            malformed = {**report, 'changed_files': [*report['changed_files'], 'src/lib.rs']}
            with self.assertRaisesRegex(ValueError, 'unrecognized patch inventory'):
                verify_provenance(malformed, root)
            omitted = {**report, 'changed_files': report['changed_files'][:-1]}
            with self.assertRaisesRegex(ValueError, 'authenticated patch inventory'):
                verify_provenance(omitted, root, history)
            self.assertEqual((source/'src/connection/mod.rs').read_bytes(), original)

    def test_prepared_lock_bytes_match_provenance(self):
        import json
        from quic_observation.prepare import digest, prepare
        root = Path(__file__).resolve().parents[2]
        history, revision = source_identity(root)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'prepared'
            prepare(root, output, history, revision)
            provenance = json.loads((output/'provenance.json').read_text())
            self.assertEqual(digest((output/'workspace/Cargo.lock').read_bytes()),
                             provenance['diagnostic_lock_sha256'])
            cargo = Path(os.environ.get('CARGO_HOME', Path.home()/'.cargo'))
            archive = next((cargo/'registry/cache').glob('*/quinn-proto-0.11.19.crate'))
            with tarfile.open(archive) as contents:
                expected = contents.extractfile('quinn-proto-0.11.19/Cargo.lock').read()
            self.assertEqual((output/'engine/Cargo.lock').read_bytes(), expected)

    def test_registry_extra_file_is_not_a_diagnostic_build_input(self):
        root = Path(__file__).resolve().parents[2]
        cargo = Path(os.environ.get('CARGO_HOME', Path.home()/'.cargo'))
        source = next((cargo/'registry/src').glob('*/quinn-proto-0.11.19'))
        archive = cargo/'registry/cache'/source.parent.name/'quinn-proto-0.11.19.crate'
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            cached = temporary/'cargo/registry/src'/source.parent.name/source.name
            shutil.copytree(source, cached)
            cached_archive = temporary/'cargo/registry/cache'/source.parent.name/archive.name
            cached_archive.parent.mkdir(parents=True)
            shutil.copyfile(archive, cached_archive)
            (cached/'unauthenticated.rs').write_text('compile_error!("not authenticated");\n')
            prior = os.environ.get('CARGO_HOME')
            os.environ['CARGO_HOME'] = str(temporary/'cargo')
            try:
                observed, _ = registry_source(root, temporary/'registry')
            finally:
                if prior is None:
                    os.environ.pop('CARGO_HOME')
                else:
                    os.environ['CARGO_HOME'] = prior
            self.assertFalse((observed/'unauthenticated.rs').exists())

    def test_registry_archive_rejects_unsafe_member_paths(self):
        from quic_observation.prepare import digest
        root = Path(__file__).resolve().parents[2]
        packages = tomllib.loads((root/'Cargo.lock').read_text())['package']
        checksum = next(p['checksum'] for p in packages
                        if p['name'] == 'quinn-proto' and p['version'] == '0.11.19')
        names = ('quinn-proto-0.11.19/../escape.rs',
                 'quinn-proto-0.11.19\\escape.rs')
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                temporary = Path(temporary)
                cargo = temporary/'cargo'
                source = cargo/'registry/src/index/quinn-proto-0.11.19'
                source.mkdir(parents=True)
                archive = cargo/'registry/cache/index/quinn-proto-0.11.19.crate'
                archive.parent.mkdir(parents=True)
                with tarfile.open(archive, 'w:gz') as contents:
                    member = tarfile.TarInfo(name)
                    member.size = 1
                    contents.addfile(member, io.BytesIO(b'x'))
                source_root = temporary/'source'
                source_root.mkdir()
                lock = (root/'Cargo.lock').read_text().replace(
                    checksum, digest(archive.read_bytes()), 1)
                (source_root/'Cargo.lock').write_text(lock)
                prior = os.environ.get('CARGO_HOME')
                os.environ['CARGO_HOME'] = str(cargo)
                try:
                    with self.assertRaisesRegex(ValueError, 'unsafe path'):
                        registry_source(source_root, temporary/'registry')
                finally:
                    if prior is None:
                        os.environ.pop('CARGO_HOME')
                    else:
                        os.environ['CARGO_HOME'] = prior

class RealStateObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gzip
        import json
        path = (Path(__file__).resolve().parents[2] /
                'moirai-quic/measurements/recovery/2026-10-14/state-native.json.gz')
        cls.evidence = json.loads(gzip.decompress(path.read_bytes()))

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
