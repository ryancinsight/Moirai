# ADR 0058: Bounded native file selection

- Status: Accepted
- Date: 2026-09-14
- Item: MOI-WINDOW-DIALOG-2026-09-14 (delivered; see repository history)
- Revision 2026-11-09: save mode added to the same API (MOI-PAL-SAVE-DIALOG-001) so
  RITK exports through a native Save As dialog; `DialogSelection` gains a
  data-carrying variant and loses `Copy`.

## Context

Metis native applications need a user-activated way to choose a saved study
before an owning application such as RITK decodes it, and to choose where an
exported image, volume, mesh or session file is written. The browser already has a
bounded DOM file provider, while RITK's existing native shell still uses an
application-side dialog dependency. The platform boundary must not interpret
DICOM or turn a browser file name into native filesystem authority.

## Decision

`moirai-pal::windows::dialog::pick` owns the Windows common-dialog primitive.
It initializes a single-threaded COM apartment, holds an `IFileDialog`
created from the `FileOpenDialog` CLSID,
selects either one regular file or one directory, forces filesystem results,
maps user cancellation to `Ok(None)`, converts the task-memory UTF-16 result
under a 32,767-unit bound and releases COM/task-memory resources through RAII.
`DialogSelection::Save(SaveRequest)` creates `IFileSaveDialog` instead, through
the shared `IFileDialog` configuration, and applies a suggested file name, an
ordered list of `FileFilter` (label, wildcard pattern) entries with the first
selected, `FOS_OVERWRITEPROMPT` and `FOS_PATHMUSTEXIST`. `SaveRequest` and
`FileFilter` validate at construction: no NUL, no path separator or colon in a
name or pattern, the same 32,767-unit bound on every string and at most 1,024
filters. The chosen destination uses the same result type as an open: a path,
cancellation as `Ok(None)`, or an `io::Error`. The caller creates or writes the
file. The public result is only a `PathBuf`; consumers validate authority, file
type, root confinement and parse budgets at their own boundary. WASM consumers
use `WebEvent::selected_files` from ADR 0050, and non-Windows native providers
are separate future implementations.

## Alternatives

1. Keeping `rfd` in each application duplicates native authority and prevents
   the Metis/Moirai host from owning one bounded platform contract.
2. Returning `std::fs::File` from the picker would couple selection to one
   consumer's read policy; RITK's existing root-confined opener remains the
   authority for subsequent reads.
3. A parallel `pick_save` function with its own result type would fork the COM
   lifecycle, cancellation mapping and path bound; one entry point keeps them
   single. The cost is three breaks for existing callers, each reported by
   `cargo semver-checks`: `DialogSelection` is no longer `Copy` (clone it to
   reuse a selection); it is `#[non_exhaustive]` so later modes do not break
   matches again (add a wildcard arm to any `match` on it); and it gains a
   data-carrying variant, so a cast such as `DialogSelection::File as u8`,
   valid on the unit-only enum, no longer compiles (write an explicit
   `match` to obtain a number).
4. Returning a browser path or exposing `web-sys::File` crosses the platform
   boundary and is rejected by ADR 0050.

## Threat model and limits

The native dialog is user-mediated but its path is still untrusted input. The
provider bounds the returned UTF-16 length and rejects invalid conversion;
consumers must apply traversal, symlink, authorization and parser-size policy.
The API does not grant access, read the selected file or prove a DICOM format.

## Verification

The PAL unit tests cover explicit selection modes, the option flags of each
mode, request and filter validation including the UTF-16 and count bounds, the
UTF-16 filter encoding, and real dialogs configured and read back without being
shown: the interface each selection creates, the save dialog's options,
suggested name and selected file type, and an empty filter list. `Show` is
modal and cannot run headless, so choosing, the overwrite prompt and
cancellation need a manual run. A further test pins the numeric value of the
Windows cancellation HRESULT; the `Show` path that maps it is untested.
Native Windows tests and warning-denied Clippy must cover the
COM/task-memory path before the item closes; a manual RITK run must select one
of the committed real studies and produce the existing byte-identified image
oracle.
