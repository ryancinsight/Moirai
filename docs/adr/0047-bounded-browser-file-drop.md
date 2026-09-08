# ADR 0047: Bounded browser file-drop metadata

Status: Accepted

Date: 2026-09-07

Driver: [MOI-WASM-DOM-DROP-2026-09-07](../backlog.md#MOI-WASM-DOM-DROP-2026-09-07),
[Metis input controls](../../metis/backlog.md#METIS-INPUT-001).

## Context

Metis needs a browser drop boundary for the DICOM viewer migration and other
HTML5 applications. The current Moirai DOM seam owns event listeners and
pointer metadata, but consumers cannot inspect files in a drop without
importing `web-sys` and retaining a second browser binding surface. Browser
drop metadata is untrusted: a page can receive an unexpectedly large file list,
overlong names or non-finite JavaScript numbers. The provider must bound its
own metadata allocation before an application policy sees it.

## Decision

Moirai's WASM DOM seam owns `WebEvent::drop_metadata`. A non-drag event returns
`Ok(None)`. A drag event returns a `DropMetadata` value containing CSS-pixel
coordinates and an owned, bounded slice of `DroppedFile` records. Each record
contains a validated name, media type and byte size. The provider accepts at
most 64 files, names up to 4,096 UTF-8 bytes and media types up to 256 bytes;
names reject NUL and empty values, media types may be empty, and file sizes
must be finite, non-negative integers representable by `u64`. These bounds cap
provider-owned metadata before allocation and match the browser `FileList`
contract; file bytes are never copied by this seam.

The public types expose borrowed accessors only. `WebEvent::drop_metadata`
returns a typed `io::Error` for a missing `DataTransfer`, missing `FileList`,
missing indexed file or invalid metadata, so consumers cannot silently accept a
partial drop. Applications decide whether a file is a DICOM object, how bytes
are read and which path or origin policy applies. `WebEventListener` continues
to own callback teardown, and consumers remain free of `web-sys` imports.

## Alternatives

Exposing `DataTransfer` or `File` directly would leak browser bindings and make
each consumer repeat bounds and lifetime rules. Reading file bytes in Moirai
would couple a platform abstraction to application format policy and create a
new asynchronous resource owner. Returning only a file count would hide the
metadata needed to present an input-sensitive drop result. A third-party DOM
wrapper would duplicate the existing Moirai callback and cancellation model.

## Threat model and limits

Drop events and all file metadata are attacker-controlled. Count and string
bounds prevent metadata-driven allocation growth; finite integer validation
prevents invalid JavaScript numbers from entering Rust; names are display
metadata and are not trusted paths. The seam does not read file bytes, validate
DICOM content, enforce filesystem permissions or produce native-host events.
Those operations remain application and host contracts with their own trust
boundaries.

## Verification

The provider's `wasm32-unknown-unknown` build verifies the `web-sys` feature
surface and generated bindings. Metis supplies native value tests for its
application drop policy, and its browser build consumes only Moirai's exported
types. The browser manual records the rendered drop-zone state and the limits
of synthetic versus trusted file-drop events; no screenshot or compile result
is treated as proof of file-byte or native-host support.
