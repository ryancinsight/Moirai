# ADR 0058: Bounded native file selection

- Status: Accepted
- Date: 2026-09-14
- Item: [MOI-WINDOW-DIALOG-2026-09-14](../backlog.md#MOI-WINDOW-DIALOG-2026-09-14)

## Context

Metis native applications need a user-activated way to choose a saved study
before an owning application such as RITK decodes it. The browser already has a
bounded DOM file provider, while RITK's existing native shell still uses an
application-side dialog dependency. The platform boundary must not interpret
DICOM or turn a browser file name into native filesystem authority.

## Decision

`moirai-pal::windows::dialog::pick` owns the Windows common-dialog primitive.
It initializes a single-threaded COM apartment, creates `IFileOpenDialog`,
selects either one regular file or one directory, forces filesystem results,
maps user cancellation to `Ok(None)`, converts the task-memory UTF-16 result
under a 32,767-unit bound and releases COM/task-memory resources through RAII.
The public result is only a `PathBuf`; consumers validate authority, file type,
root confinement and parse budgets at their own boundary. WASM consumers use
`WebEvent::selected_files` from ADR 0050, and non-Windows native providers are
separate future implementations.

## Alternatives

1. Keeping `rfd` in each application duplicates native authority and prevents
   the Metis/Moirai host from owning one bounded platform contract.
2. Returning `std::fs::File` from the picker would couple selection to one
   consumer's read policy; RITK's existing root-confined opener remains the
   authority for subsequent reads.
3. Returning a browser path or exposing `web-sys::File` crosses the platform
   boundary and is rejected by ADR 0050.

## Threat model and limits

The native dialog is user-mediated but its path is still untrusted input. The
provider bounds the returned UTF-16 length and rejects invalid conversion;
consumers must apply traversal, symlink, authorization and parser-size policy.
The API does not grant access, read the selected file or prove a DICOM format.

## Verification

The PAL unit tests prove explicit selection modes and the Windows cancellation
HRESULT mapping. Native Windows tests and warning-denied Clippy must cover the
COM/task-memory path before the item closes; a manual RITK run must select one
of the committed real studies and produce the existing byte-identified image
oracle.
