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

Follow-on revision 2026-09-14 — the WebKit regression drives a stream-based
read for each bounded `Blob` slice. `Blob.stream()` is consumed with a bounded
reader and a scratch view no larger than the caller request; the reader lock is
released on every completion or error. The provider continues to advance its
cursor only by copied bytes and surfaces stream failures as typed I/O errors;
it does not fall back to a whole-file allocation or expose the browser handle.
The hosted chooser matrix remains the acceptance oracle for this change.

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
fixed chunk bound, validate the browser-reported size and cursor arithmetic, and
advance the cursor only by bytes copied. Each bounded `Blob` slice is read
through its browser stream; dropping the Rust future stops consuming the
result, while the browser owns completion of that already-started read. The
stream reader is released before the future returns or reports an error.

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
entry count at 512, metadata strings, one read chunk and cursor arithmetic. Content
validation and DICOM parsing belong to the consumer decoder. Browser-selected
files remain subject to browser origin and user-grant rules; this seam does not
create native filesystem authority or prove a trusted operating-system drop.

## Verification

Native value tests cover metadata preservation and read cursor policy where the
pure validation helpers are available. Commit
`f51b5c2670c840e8a8302d7f5a5ebeecc57ea076` passes 63 `moirai-pal` tests,
warning-denied native Clippy and a warning-denied
`wasm32-unknown-unknown` provider check. Metis adds the consumer compile/use
surface. The cross-engine RITK chooser run is the browser regression check; a
browser trace must identify its engine and whether the file came from a trusted
user operation before claiming byte-read evidence.
