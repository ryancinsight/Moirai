# ADR 0050: Bounded browser file access

Status: Accepted

Date: 2026-09-08

Revision: 2026-09-12 — the same bounded file reader accepts a user-activated
`<input type="file">` selection through `WebEvent::selected_files`; no path or
browser handle leaves Moirai. The drag/drop surface remains unchanged, and
the consumer receives one source-neutral file batch.

Revision: 2026-09-11 — the shared browser drop bound is 512 entries so a
committed 409-slice DICOM study fits in one bounded batch; the consumer-owned
256 MiB byte limit and per-read chunk bound remain unchanged.

Revision: 2026-09-14 — the saved-study chooser matrix showed Safari accepting
file metadata but rejecting `FileReader.readAsArrayBuffer` for a bounded
`Blob` slice. Reads now await the slice's `Blob.arrayBuffer()` promise; the
slice and the 1 MiB caller buffer keep the allocation bound while the source
remains the browser-owned `File`. Chromium and Firefox retain the same byte
contract, and the WebKit run is the regression oracle for this provider path.

Follow-on revision 2026-09-14 — hosted Chromium and Firefox accepted the
default reader, but WebKit still rejected the first `Blob.stream()` read after
accepting the saved study. Each bounded `Blob` slice is now exposed through a
local object URL and fetched as a response stream, which avoids the
disk-backed-file stream path while keeping byte chunks browser-owned. The
reader lock is released on every completion or error, and the object URL is
revoked on success, failure or cancellation. The provider continues to advance
its cursor only by copied bytes and surfaces response/stream failures as typed
I/O errors; it does not allocate beyond the provider bound or expose the
browser handle. The hosted chooser matrix remains the acceptance oracle for
this provider path.

Revision: 2026-09-15 — the hosted run still rejected Safari's first sliced
object-URL read. A first read that covers a file no larger than the provider's
1 MiB bound now uses `File.arrayBuffer()` directly; larger, later and positioned
reads retain the bounded object-URL response stream. Metis requests the full
bounded file on its first read when that is safe, so the saved 529,864-byte
DICOM instances use the Safari-compatible path without allowing an unbounded
allocation. The cross-engine chooser run remains the acceptance oracle.

Follow-on revision 2026-09-15 — `WebFile::from_js_file` now validates the
browser `File.size()` value before constructing a reader and caches the checked
`u64`. `size()`, `seek()` and `read()` therefore share one boundary proof, and
malformed JavaScript metadata returns `InvalidInput` instead of reaching a
release build's narrowing cast.

Driver: [MOI-WASM-DOM-FILE-2026-09-08](../backlog.md#MOI-WASM-DOM-FILE-2026-09-08),
[Metis input controls](../../metis/backlog.md#METIS-INPUT-001).

## Context

Metis needs to pass selected DICOM bytes to the owning RITK decoder from its
browser host. Moirai already owns the WASM browser-file lifecycle, but
the public drop seam currently stops at metadata. Requiring every consumer to
import `web-sys` would duplicate browser bindings and callback teardown. A file
name is display metadata and cannot become a filesystem path or authority.

## Decision

Extend Moirai's owned WASM DOM seam with `WebEvent::drop_files` and
`WebEvent::selected_files`. The former returns a
bounded `DropFiles` snapshot whose entries pair the existing validated metadata
with an owned browser file reader. The entry exposes metadata accessors and an
asynchronous read/seek surface; the JavaScript `File` object remains private to
Moirai. Reads use caller-provided buffers, reject a buffer above the provider's
fixed chunk bound, validate the browser-reported size at the fallible reader
constructor and cursor arithmetic, and
advance the cursor only by bytes copied. A first whole-file read within the
bound uses the browser `File.arrayBuffer()` promise; each other bounded `Blob`
slice is read through a local object URL's response stream. Dropping the Rust
future revokes the URL, while the browser owns completion of that already-started
read. The response reader is released before the future returns or reports an
error.

The existing `drop_metadata` API and value semantics remain unchanged. The new
surface does not parse DICOM, infer a path, grant native permissions, or retain
an unbounded file in memory. Consumers must pass bounded chunks to their own
format decoder and apply their own content and authority policy. The chooser
method returns the same reader entries in a source-neutral `BrowserFiles`
batch, so a user file selection does not need to synthesize a drag event.

## Alternatives

Exposing `web_sys::File` would make each Atlas consumer own browser bindings and
callback lifetimes. Copying the complete file would make an untrusted size drive
an unbounded allocation. Returning only metadata cannot support a browser DICOM
workflow. A third-party browser file wrapper would duplicate the existing
Moirai PAL and add another runtime dependency.

## Threat model and limits

The drop event, metadata and file contents are untrusted. Provider bounds cap
entry count at 512, metadata strings, one read chunk and cursor arithmetic. The
reader constructor rejects non-finite, negative, fractional and overflowing
JavaScript sizes before any cursor operation. Content validation and DICOM
parsing belong to the consumer decoder. Browser-selected
files remain subject to browser origin and user-grant rules; this seam does not
create native filesystem authority or prove a trusted operating-system drop.

## Verification

Native value tests cover metadata preservation, malformed JavaScript size
rejection and read cursor policy where the pure validation helpers are
available. The provider change passes 77 `moirai-pal` nextest cases,
warning-denied native Clippy and warning-denied native-library and
`wasm32-unknown-unknown` checks. Metis adds the consumer compile/use surface.
The cross-engine RITK chooser run is the browser regression check; a browser
trace must identify its engine and whether the file came from a trusted user
operation before claiming byte-read evidence.
