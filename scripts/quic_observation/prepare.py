#!/usr/bin/env python3
"""Build-only read-only instrumentation of the exact pinned registry engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

ENGINE = 'quinn-proto'
VERSION = '0.11.19'
ALLOWED = {'src/connection/mod.rs', 'src/connection/observation.rs',
           'src/connection/packet_builder.rs', 'src/connection/streams/state.rs'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def registry_source(root):
    packages = tomllib.loads((root/'Cargo.lock').read_text())['package']
    package = next(p for p in packages if p['name'] == ENGINE)
    if package['version'] != VERSION:
        raise ValueError('observation patch requires the exact pinned engine')
    cargo = Path(os.environ.get('CARGO_HOME', Path.home()/'.cargo'))
    sources = list((cargo/'registry/src').glob(f'*/{ENGINE}-{VERSION}'))
    valid = []
    for source in sources:
        archive = cargo/'registry/cache'/source.parent.name/f'{ENGINE}-{VERSION}.crate'
        if archive.exists() and digest(archive.read_bytes()) == package['checksum']:
            with tarfile.open(archive) as contents:
                for entry in contents.getmembers():
                    if not entry.isfile():
                        continue
                    name = Path(entry.name).relative_to(f'{ENGINE}-{VERSION}')
                    if (source/name).read_bytes() != contents.extractfile(entry).read():
                        raise ValueError(f'modified registry source: {name}')
            valid.append(source)
    if len(valid) != 1:
        raise ValueError('exactly one verified cached engine is required; cargo fetch first')
    return valid[0], package['checksum']


def patch_source(source, destination, root):
    patch = root/'scripts/quic_observation/engine.patch'
    raw = patch.read_text()
    if any(line.startswith('-') and not line.startswith('--- ') for line in raw.splitlines()):
        raise ValueError('instrumentation cannot remove or replace an engine statement')
    changed = {line.removeprefix('+++ b/') for line in raw.splitlines() if line.startswith('+++ b/')}
    if changed != ALLOWED:
        raise ValueError('unexpected instrumentation file inventory')
    shutil.copytree(source, destination)
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=destination, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=destination, check=True)
    for original in source.rglob('*'):
        if original.is_file() and str(original.relative_to(source)) not in changed:
            if original.read_bytes() != (destination/original.relative_to(source)).read_bytes():
                raise ValueError('instrumentation changes an unrelated source file')
    return {'engine': ENGINE, 'version': VERSION, 'patch_sha256': digest(patch.read_bytes()),
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


def verify_provenance(report, root):
    source, checksum = registry_source(root)
    with tempfile.TemporaryDirectory() as temporary:
        expected = patch_source(source, Path(temporary)/'engine', root)
    original = (root/'Cargo.lock').read_text()
    expected.update({'registry_checksum': checksum,
                     'canonical_lock_sha256': digest(original.encode()),
                     'diagnostic_lock_sha256': digest(diagnostic_lock(original).encode())})
    if report != expected:
        raise ValueError('diagnostic build provenance differs from verified source/patch/locks')


def prepare(root, destination):
    source, checksum = registry_source(root)
    destination.mkdir(parents=True, exist_ok=False)
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
    (workspace/'Cargo.lock').write_text(observed)
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
