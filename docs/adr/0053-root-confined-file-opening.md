# ADR 0053: Root-confined file opening

- Status: Accepted
- Date: 2026-09-09
- Item: [MOI-FS-CONFINED-2026-09-09](../backlog.md#MOI-FS-CONFINED-2026-09-09)

## Context

RITK consumes file sets selected by a user and follows relative references in
DICOMDIR records. Canonicalizing a candidate and opening it by path leaves a
race: a directory component can be replaced after the check and before the
read. RITK's accepted viewer migration decision assigns root-confined opens to
Moirai's filesystem layer, while DICOM parsing and medical policy remain in
RITK.

The contract must hold on native hosts without creating filesystem authority
 for browser paths. It must reject absolute paths, parent traversal, and link
 components, and it must return the already-confined handle that the caller
 reads. Read-size limits remain a consumer concern.

## Decision

`moirai-pal::fs::open_file_within_root` owns the native primitive. The API
accepts a candidate path and a directory root and returns a read-only
`std::fs::File`.

- Unix opens the root directory with `O_DIRECTORY | O_NOFOLLOW` and walks each
  normal relative component with `openat` and `O_NOFOLLOW`. Intermediate
  components must be directories and the final component must be a regular
  file.
- Windows opens the root as a reparse-point-aware directory handle and opens
  each child relative to the current directory handle with `NtCreateFile`.
  Directory and non-directory create options enforce the component kind, and
  `FILE_OPEN_REPARSE_POINT` prevents transparent reparse traversal.
- WebAssembly returns `io::ErrorKind::Unsupported`; browser file entries are
  owned by the existing DOM file provider and never become native paths.

The lexical validation and handle walk are one operation. Callers read the
returned handle directly; a canonicalize-then-open sequence is not part of the
contract. RITK composes this primitive with its existing bounded parser read.

## Alternatives rejected

1. A second implementation in `ritk-dicom` would put platform filesystem
   authority in the domain repository and contradict the viewer migration
   boundary.
2. Canonicalize-then-open would preserve the time-of-check/time-of-use race.
3. A path-only `std::fs::File::open` fallback would silently weaken the
   contract on one platform, so unsupported targets fail explicitly instead.

## Threat model and limits

The input path may be controlled by a DICOMDIR or a dropped file-set member.
Traversal and link substitution are the relevant spoofing and escape threats;
the handle walk addresses them before bytes are read. The caller still owns
the parse byte budget, file-set selection policy, and any authorization that
precedes choosing the root. A root path that the caller itself does not trust
is outside this API's authority model.

## Verification

The PAL tests cover successful nested reads, absolute and parent paths,
intermediate and final symlinks, and final non-file components. RITK's DICOM
reader tests exercise DICOMDIR references through the PAL contract. Native
warning-denied tests, documentation, and the `wasm32-unknown-unknown` library
check are required before the item closes.
