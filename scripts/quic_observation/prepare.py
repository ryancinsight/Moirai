#!/usr/bin/env python3
"""Build-only read-only instrumentation of the exact pinned registry engine."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

ENGINE = 'quinn-proto'
VERSION = '0.11.19'
ALLOWED = {'src/connection/mod.rs', 'src/connection/observation.rs',
           'src/connection/packet_builder.rs', 'src/connection/streams/state.rs'}
GIT = ['git', '--no-replace-objects']


def git_environment():
    environment = os.environ.copy()
    environment['GIT_NO_REPLACE_OBJECTS'] = '1'
    return environment


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def registry_source(root, destination):
    packages = tomllib.loads((root/'Cargo.lock').read_text())['package']
    package = next(p for p in packages if p['name'] == ENGINE)
    if package['version'] != VERSION:
        raise ValueError('observation patch requires the exact pinned engine')
    cargo = Path(os.environ.get('CARGO_HOME', Path.home()/'.cargo'))
    sources = list((cargo/'registry/src').glob(f'*/{ENGINE}-{VERSION}'))
    valid = []
    for source in sources:
        archive = cargo/'registry/cache'/source.parent.name/f'{ENGINE}-{VERSION}.crate'
        archive_raw = archive.read_bytes() if archive.exists() else b''
        if digest(archive_raw) == package['checksum']:
            with tarfile.open(fileobj=io.BytesIO(archive_raw)) as contents:
                entries = contents.getmembers()
                if not entries or not all(entry.isfile() for entry in entries):
                    raise ValueError('cached engine archive contains a non-file entry')
                names = []
                for entry in entries:
                    if '\\' in entry.name:
                        raise ValueError('cached engine archive contains an unsafe path')
                    name = PurePosixPath(entry.name).relative_to(f'{ENGINE}-{VERSION}')
                    if not name.parts or '..' in name.parts:
                        raise ValueError('cached engine archive contains an unsafe path')
                    names.append(name.as_posix())
                    if (source/name).read_bytes() != contents.extractfile(entry).read():
                        raise ValueError(f'modified registry source: {name}')
                if (len(set(names)) != len(names) or
                        len({name.casefold() for name in names}) != len(names)):
                    raise ValueError('cached engine archive contains colliding paths')
                valid.append((archive_raw, names))
    if len(valid) != 1:
        raise ValueError('exactly one verified cached engine is required; cargo fetch first')
    archive_raw, names = valid[0]
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive_raw)) as contents:
        for name in names:
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents.extractfile(f'{ENGINE}-{VERSION}/{name}').read())
    return destination, package['checksum']


def patch_source(source, destination, root, patch_raw=None, allowed=None):
    patch = root/'scripts/quic_observation/engine.patch'
    raw = patch.read_bytes() if patch_raw is None else patch_raw
    text = raw.decode()
    if any(line.startswith('-') and not line.startswith('--- ') for line in text.splitlines()):
        raise ValueError('instrumentation cannot remove or replace an engine statement')
    changed = {line.removeprefix('+++ b/') for line in text.splitlines()
               if line.startswith('+++ b/')}
    if changed != (ALLOWED if allowed is None else allowed):
        raise ValueError('unexpected instrumentation file inventory')
    shutil.copytree(source, destination)
    command = ['git', '-c', 'core.autocrlf=false', 'apply', '--unidiff-zero']
    subprocess.run([*command, '--check', '-'], cwd=destination, input=raw, check=True)
    subprocess.run([*command, '-'], cwd=destination, input=raw, check=True)
    for name in changed - {'src/connection/observation.rs'}:
        original_lines = iter((source/name).read_text().splitlines())
        observed_lines = iter((destination/name).read_text().splitlines())
        if not all(any(candidate == line for candidate in observed_lines)
                   for line in original_lines):
            raise ValueError(f'instrumentation changes original statement sequence: {name}')
    for original in source.rglob('*'):
        if original.is_file() and original.relative_to(source).as_posix() not in changed:
            if original.read_bytes() != (destination/original.relative_to(source)).read_bytes():
                raise ValueError('instrumentation changes an unrelated source file')
    return {'engine': ENGINE, 'version': VERSION, 'patch_sha256': digest(raw),
            'changed_files': sorted(changed),
            'pristine_mod_sha256': digest((source/'src/connection/mod.rs').read_bytes()),
            'observed_mod_sha256': digest((destination/'src/connection/mod.rs').read_bytes()),
            'observed_files_sha256': {p: digest((destination/p).read_bytes()) for p in sorted(changed)}}


def diagnostic_lock(original):
    blocks = original.split('[[package]]')
    for index, block in enumerate(blocks):
        if block.startswith(f'\nname = "{ENGINE}"\n'):
            lines = block.splitlines(keepends=True)
            blocks[index] = ''.join(line for line in lines if not line.startswith(('source = ', 'checksum = ')))
    observed = '[[package]]'.join(blocks)
    packages = lambda raw: {p['name']+'@'+p['version']: p for p in tomllib.loads(raw)['package']}
    before, after = packages(original), packages(observed)
    expected = {k: dict(p) for k, p in before.items()}
    entry = expected[f'{ENGINE}@{VERSION}']; entry.pop('source'); entry.pop('checksum')
    if after != expected:
        raise ValueError('diagnostic lock changes more than engine source/checksum')
    return observed


def provenance_inputs(report, root, history_repository):
    patch_name = 'scripts/quic_observation/engine.patch'
    expected_patch = report.get('patch_sha256')
    expected_lock = report.get('canonical_lock_sha256')
    raw = (root/patch_name).read_bytes()
    original = (root/'Cargo.lock').read_text()
    if digest(raw) == expected_patch and digest(original.encode()) == expected_lock:
        return raw, original
    revisions = subprocess.check_output(
        [*GIT, 'log', '--all', '--format=%H', '--', patch_name, 'Cargo.lock'],
        cwd=history_repository, text=True, env=git_environment()).splitlines()
    for revision in revisions:
        try:
            candidate = subprocess.check_output(
                [*GIT, 'show', f'{revision}:{patch_name}'], cwd=history_repository,
                stderr=subprocess.DEVNULL, env=git_environment())
            candidate_lock = subprocess.check_output(
                [*GIT, 'show', f'{revision}:Cargo.lock'], cwd=history_repository,
                stderr=subprocess.DEVNULL, text=True, env=git_environment())
        except subprocess.CalledProcessError:
            continue
        if (digest(candidate) == expected_patch and
                digest(candidate_lock.encode()) == expected_lock):
            return candidate, candidate_lock
    raise ValueError('diagnostic patch provenance is not reachable')


def verify_provenance(report, root, history_repository=None):
    history_repository = root if history_repository is None else history_repository
    reported_files = set(report.get('changed_files', ()))
    if reported_files != ALLOWED:
        raise ValueError('diagnostic report has an unrecognized patch inventory')
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        source, checksum = registry_source(root, temporary/'registry')
        raw, original = provenance_inputs(report, root, history_repository)
        expected = patch_source(source, Path(temporary)/'engine', root, raw,
                                reported_files)
    expected.update({'registry_checksum': checksum,
                     'canonical_lock_sha256': digest(original.encode()),
                     'diagnostic_lock_sha256': digest(diagnostic_lock(original).encode())})
    if report != expected:
        raise ValueError('diagnostic build provenance differs from verified source/patch/locks')


def prepare(root, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        source, checksum = registry_source(root, Path(temporary)/'registry')
        report = patch_source(source, destination/'engine', root)
    report['registry_checksum'] = checksum
    workspace = destination/'workspace'
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    for name in filter(None, files):
        original_file = root/name
        if not original_file.is_file():
            raise ValueError(f'tracked source is absent: {name}')
        target = workspace/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_file, target)
    original = (root/'Cargo.lock').read_text()
    observed = diagnostic_lock(original)
    (workspace/'Cargo.lock').write_text(observed, newline='\n')
    config = '[patch.crates-io]\nquinn-proto = { path = '+json.dumps(str(destination/'engine'))+' }\n'
    (destination/'config.toml').write_text(config)
    report.update({'canonical_lock_sha256': digest(original.encode()),
                   'diagnostic_lock_sha256': digest(observed.encode())})
    (destination/'provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'workspace': str(workspace), 'config': str(destination/'config.toml')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    prepare(Path(__file__).resolve().parents[2], arguments.output.resolve())


if __name__ == '__main__':
    main()
