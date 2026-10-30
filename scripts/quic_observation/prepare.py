#!/usr/bin/env python3
"""Build-only read-only instrumentation of the exact pinned registry engine."""
import argparse
import hashlib
import io
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import posixpath
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
    source_tree = report.get('source_tree')
    if source_tree is not None:
        try:
            raw = subprocess.check_output(
                [*GIT, 'show', f'{source_tree}:{patch_name}'], cwd=history_repository,
                env=git_environment())
            original = subprocess.check_output(
                [*GIT, 'show', f'{source_tree}:Cargo.lock'], cwd=history_repository,
                text=True, env=git_environment())
        except subprocess.CalledProcessError as error:
            raise ValueError('diagnostic source tree is not reachable') from error
        if digest(raw) != expected_patch or digest(original.encode()) != expected_lock:
            raise ValueError('diagnostic source tree differs from reported patch/lock')
        return raw, original
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


def link_parts(paths, value, name):
    if paths is ntpath:
        value = value.replace('/', '\\')
    drive, tail = paths.splitdrive(value)
    if drive or paths.isabs(value):
        raise ValueError(f'diagnostic symlink escapes source: {name}')
    return [part for part in tail.split(paths.sep) if part not in ('', paths.curdir)]


def validate_archive_links(links):
    for paths in (posixpath, ntpath):
        table = {}
        for name, target in links.items():
            if not target:
                raise ValueError(f'diagnostic symlink target is empty: {name}')
            key = paths.normcase(paths.sep.join(link_parts(paths, name, name)))
            table[key] = target
        for name, target in links.items():
            parent = link_parts(paths, paths.dirname(name), name)
            pending = link_parts(paths, target, name)
            resolved = parent
            expansions = 0
            while pending:
                part = pending.pop(0)
                if part == paths.pardir:
                    if not resolved:
                        raise ValueError(f'diagnostic symlink escapes source: {name}')
                    resolved.pop()
                    continue
                resolved.append(part)
                key = paths.normcase(paths.sep.join(resolved))
                if key in table:
                    expansions += 1
                    if expansions > len(table):
                        raise ValueError(f'diagnostic symlink cycle: {name}')
                    resolved.pop()
                    pending[:0] = link_parts(paths, table[key], name)


def archived_source(history_repository, revision, destination):
    source_tree = subprocess.check_output(
        [*GIT, 'rev-parse', f'{revision}^{{tree}}'],
        cwd=history_repository, text=True, env=git_environment()).strip()
    entries = subprocess.check_output([
        *GIT, 'ls-tree', '-r', '-z', '--full-tree',
        '--format=%(objectmode) %(objecttype) %(objectname) %(path)', source_tree],
        cwd=history_repository, env=git_environment()).decode().split('\0')
    blobs = {}
    gitlinks = []
    for entry in filter(None, entries):
        mode, object_type, object_name, name = entry.split(' ', 3)
        if object_type == 'blob':
            blobs[name] = (mode, object_name)
        elif mode == '160000' and object_type == 'commit':
            gitlinks.append({'path': name, 'commit': object_name})
        else:
            raise ValueError(f'unsupported diagnostic source entry: {name}')
    object_format = subprocess.check_output(
        [*GIT, 'rev-parse', '--show-object-format'],
        cwd=history_repository, text=True, env=git_environment()).strip()
    archive = destination.parent/'source.tar'
    attributes = destination.parent/'archive.attributes'
    attributes.write_bytes(b'')
    with archive.open('wb') as output:
        subprocess.run([*GIT, '-c', 'core.autocrlf=false',
                        '-c', f'core.attributesFile={attributes}',
                        'archive', '--format=tar', source_tree],
                       cwd=history_repository, stdout=output, check=True,
                       env=git_environment())
    with tarfile.open(archive) as contents:
        material = [member for member in contents.getmembers()
                    if member.isfile() or member.issym()]
        members = {member.name: member for member in material}
        if len(members) != len(material) or set(members) != set(blobs):
            raise ValueError('diagnostic archive inventory differs from its Git tree')
        if len({name.casefold() for name in blobs}) != len(blobs):
            raise ValueError('diagnostic source contains a case-folding path collision')
        links = {}
        for name, (mode, _) in blobs.items():
            member = members[name]
            if mode == '120000':
                if not member.issym():
                    raise ValueError(f'diagnostic symlink mode differs: {name}')
                links[name] = member.linkname
            elif not member.isfile() or bool(member.mode & 0o111) != (mode == '100755'):
                raise ValueError(f'diagnostic file mode differs: {name}')
        validate_archive_links(links)
        destination.mkdir()
        contents.extractall(
            destination,
            members=(member for member in contents.getmembers() if not member.issym()),
            filter='data',
        )
        for name, target in links.items():
            try:
                os.symlink(target, destination/name)
            except OSError as error:
                raise ValueError('host cannot materialize diagnostic symlinks') from error
        extracted = {path.relative_to(destination).as_posix()
                     for path in destination.rglob('*')
                     if path.is_file() or path.is_symlink()}
        if extracted != set(blobs):
            raise ValueError('diagnostic extracted inventory differs from its Git tree')
        for name, (mode, object_name) in blobs.items():
            if mode == '120000':
                if not (destination/name).is_symlink():
                    raise ValueError(f'diagnostic symlink mode differs: {name}')
                raw = os.fsencode(os.readlink(destination/name))
                if raw != os.fsencode(links[name]):
                    raise ValueError(f'diagnostic symlink target differs: {name}')
            else:
                raw = (destination/name).read_bytes()
            hashed = hashlib.new(object_format)
            hashed.update(f'blob {len(raw)}\0'.encode())
            hashed.update(raw)
            if hashed.hexdigest() != object_name:
                raise ValueError(f'diagnostic blob differs: {name}')
    return source_tree, gitlinks


def verify_provenance(report, root, history_repository=None):
    history_repository = root if history_repository is None else history_repository
    reported_files = set(report.get('changed_files', ()))
    if reported_files != ALLOWED:
        raise ValueError('diagnostic report has an unrecognized patch inventory')
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        source_root = root
        if 'source_tree' in report:
            source_root = temporary/'source'
            source_tree, source_gitlinks = archived_source(
                history_repository, report['source_tree'], source_root)
        source, checksum = registry_source(source_root, temporary/'registry')
        raw, original = provenance_inputs(report, source_root, history_repository)
        expected = patch_source(source, Path(temporary)/'engine', source_root, raw,
                                reported_files)
    expected.update({'registry_checksum': checksum,
                     'canonical_lock_sha256': digest(original.encode()),
                     'diagnostic_lock_sha256': digest(diagnostic_lock(original).encode())})
    if 'source_tree' in report:
        expected['source_tree'] = source_tree
        expected['source_gitlinks'] = source_gitlinks
    if report != expected:
        raise ValueError('diagnostic build provenance differs from verified source/patch/locks')


def prepare(root, destination, history_repository=None, revision='HEAD'):
    history_repository = root if history_repository is None else history_repository
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        source_root = Path(temporary)/'source'
        source_tree, source_gitlinks = archived_source(
            history_repository, revision, source_root)
        source, checksum = registry_source(source_root, Path(temporary)/'registry')
        report = patch_source(source, destination/'engine', source_root)
        report['registry_checksum'] = checksum
        workspace = destination/'workspace'
        shutil.copytree(source_root, workspace, symlinks=True)
        original = (source_root/'Cargo.lock').read_text()
        observed = diagnostic_lock(original)
        config = ('[build]\n'
                  'rustflags = ["--cfg", "quic_observation", '
                  '"--check-cfg=cfg(quic_observation)"]\n\n'
                  '[patch.crates-io]\nquinn-proto = { path = ' +
                  json.dumps(str(destination/'engine')) + ' }\n')
        (destination/'config.toml').write_text(config, newline='\n')
        # The lock is written last so Cargo's freshness check never treats the
        # newly created path patch as newer lock input requiring a no-op rewrite.
        (workspace/'Cargo.lock').write_text(observed, newline='\n')
        subprocess.run([
            'cargo', '--config', str(destination/'config.toml'), 'metadata', '--offline',
            '--manifest-path', str(workspace/'Cargo.toml'), '--format-version', '1'],
            cwd=destination, stdout=subprocess.DEVNULL, check=True)
        if (workspace/'Cargo.lock').read_bytes() != observed.encode():
            raise ValueError('Cargo changes the diagnostic lock during patch resolution')
        report.update({'canonical_lock_sha256': digest(original.encode()),
                       'source_tree': source_tree,
                       'source_gitlinks': source_gitlinks,
                       'diagnostic_lock_sha256': digest(observed.encode())})
        provenance_inputs(report, source_root, history_repository)
        (destination/'provenance.json').write_text(
            json.dumps(report, indent=2)+'\n', newline='\n')
    print(json.dumps({'workspace': str(workspace), 'config': str(destination/'config.toml')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-tree')
    parser.add_argument('--history-repository', type=Path)
    arguments = parser.parse_args()
    if arguments.source_tree is None:
        if arguments.history_repository is not None:
            parser.error('--history-repository requires --source-tree')
        prepare(Path(__file__).resolve().parents[2], arguments.output.resolve())
        return
    if arguments.history_repository is None:
        parser.error('--source-tree requires --history-repository')
    history = arguments.history_repository.resolve()
    prepare(history, arguments.output.resolve(), history, arguments.source_tree)


if __name__ == '__main__':
    main()
