# moirai-pal

[![crates.io](https://img.shields.io/crates/v/moirai-pal.svg)](https://crates.io/crates/moirai-pal)
[![docs.rs](https://docs.rs/moirai-pal/badge.svg)](https://docs.rs/moirai-pal)

Platform Abstraction Layer for the [Moirai](https://github.com/ryancinsight/Moirai)
runtime's async I/O. One `Reactor` trait over the platform's native readiness
mechanism, so the async stack needs no external runtime:

| Target | Mechanism |
|--------|-----------|
| Linux | `epoll` |
| macOS / BSD | `kqueue` |
| Windows | `WSAPoll` socket readiness polling and a thread-owned Win32 window |
| WebAssembly | Web APIs via JavaScript interop |

Modules: `reactor`, `net`, `fs`, `timer`, plus the per-platform `unix`,
`windows`, and `wasm` implementations. On Windows, `windows::window::NativeWindow`
provides a bounded message queue and ARGB software presentation surface for a
consumer-owned event loop. `NativeWindow::wait_events` adds a finite message
queue wait for event-driven hosts; waits beyond its 30-second bound are
rejected. Each native window enters per-monitor-v2 DPI awareness for its
thread-owned lifetime and restores the prior thread context on drop, so
`WM_DPICHANGED` reports real monitor transitions when the host supplies them.
Native IME start, preedit, commit and cancellation phases are returned as
bounded UTF-8 snapshots.

`windows::webview::WebViewHost` embeds the installed Windows WebView2 runtime
inside a `NativeWindow` for packaged HTML5, CSS and WebAssembly applications.
The module is opt-in: enable the `webview2` feature on `moirai-pal`; default
Windows consumers compile the native window and file providers without the
WebView2 COM binding.
It admits only a validated `file:///` package prefix, bounds UTF-16 JSON
messages, denies external and new-window navigation, and keeps creation,
navigation and teardown waits finite. The WebView2 runtime is a Windows system
prerequisite; this provider does not require a registry token, signing key or
other application credential.
All WebView2 permission requests are denied synchronously before a profile or
OS prompt can grant access, and the host emits a bounded
`WebViewEvent::PermissionDenied` snapshot for the consumer's audit surface.
`WebViewHost::capture_preview_png` obtains a bounded PNG from WebView2's own
preview stream, so an occluded or hardware-composed surface can be inspected
without a GDI screenshot. The controller fills its window's client area and is
visible from creation, so a hidden window still renders and captures; a
controller hidden through `set_visible(false)` refuses capture, because
WebView2 withholds a hidden controller's capture until it is shown.

`fs::open_file_within_root` opens a native regular file through directory
handles, using `openat` on Unix and relative `NtCreateFile` calls on Windows.
It rejects traversal and link components before returning the handle that the
caller reads. WebAssembly reports an explicit unsupported error; browser file
bytes use the DOM file provider instead.

The WASM module owns the browser boundary used by Atlas applications. `WebDocument`
and `WebElement` provide bounded DOM updates, input/select values, checked
checkbox/radio state, disabled button/input/select state, modal dialog lifecycle
and focus control. Pointer events expose their browser identifier, and elements
can capture, query, and release that identifier through the same owned seam.
Pointer events also expose a value snapshot with normalized device type,
viewport and target-relative CSS-pixel coordinates, button state, modifier
keys and primary-pointer state. Wheel events expose bounded deltas with their
browser unit, viewport and target-relative coordinates and the same
modifier-key snapshot. Target-relative coordinates let a consumer route one
event stream to a specific canvas without re-reading layout through a second
browser binding.
Drag events expose CSS-pixel coordinates and an owned `DropMetadata` snapshot
with at most 512 validated files; names, media types and byte sizes are bounded
before they reach an application. The provider bounds file metadata separately
from the consumer's 256 MiB byte batch limit. The seam does not read file bytes
or treat a name as a filesystem path.
Text controls expose bounded values, UTF-16 selection ranges with direction,
and bounded `InputEvent`/`CompositionEvent` metadata through owned snapshots.
`WebDocument::clipboard` resolves the secure-context browser text clipboard;
`WebClipboard::read_text` and `write_text` are asynchronous and enforce the
same 1 MiB UTF-8 bound as text controls. Browser permission and transient
user-activation failures remain explicit errors, and no native clipboard
handle crosses the provider boundary.
`WebDocument::canvas_by_id` resolves an HTML5 canvas and `WebCanvas::present`
uploads a borrowed, validated RGBA8 frame under the shared platform size and
byte bounds; the provider retains no frame bytes after the call.
`WebGpuCanvas::from_current_document` is the explicit GPU counterpart: it
asynchronously acquires the browser's WebGPU adapter and device, configures a
`webgpu` canvas, and presents the same borrowed frame through
`copyExternalImageToTexture`. WebGPU absence, device loss, and upload errors
are returned to the consumer; the PAL never changes a requested GPU surface to
the 2-D path implicitly. `WebGpuCanvas::recreate` explicitly replaces a lost
device and clears the configured extent. Each device has a cancellable
`GPUDevice.lost` observer, and presentation returns a typed I/O error after the
browser event loop reports loss. Failed recovery leaves the prior state
unchanged. Browser WebGPU remains an optional secure-context capability, and
consumers own their presentation-policy choice.
Unsupported targets and browser metadata failures return explicit errors or
`None`; grapheme segmentation and editing policy stay with the application or
host layer. Native IME event production stays in the Windows provider.
`WebEventListener` removes its callback registration when dropped. `WebAnimationFrame`
binds one `requestAnimationFrame` callback to a cancellation-safe future and
cancels it when the future is dropped. `spawn_local` uses the browser event loop
for futures; applications do not create a second executor or retain detached
JavaScript closures.
`spawn_local_with_handle` adds
a single-owner `LocalTaskHandle`; cancelling or dropping it wakes the task and
drops its child future, releasing a pending PAL receive or timer.

```toml
[dependencies]
moirai-pal = "0.6"
```

```rust
use moirai_pal::{create_reactor, Interest};

fn setup() -> std::io::Result<()> {
    let reactor = create_reactor()?;         // PlatformReactor for this target
    let interest = Interest::READABLE;       // read readiness (plus errors)
    let _ = (reactor, interest);
    Ok(())
}
```

This is a runtime-internal layer; most users reach it through
[`moirai-async`](https://crates.io/crates/moirai-async) rather than directly.

Full documentation: <https://docs.rs/moirai-pal>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.

## Rust page provider contracts

Workspace 0.9.0 adds `WebViewBridge` and `WebViewMessageListener` for the actual
WebView2 message object. Posting parses JSON into the native object value;
receiving admits object/string data under the shared 64 KiB byte and UTF16
bounds. `WebViewBridge::operation_wait_limit` exposes the shared 30-second
operation bound. A listener's explicit `close` detaches its callback before
releasing Rust ownership. A failed Drop retains one counted callback guard and
refuses replacement until realm destruction. `WebEventListener::close`,
`active_count` and `retained_cleanup_count` expose the same DOM teardown law;
failed owners cannot accumulate across remount. `WebNumber::decimal` calls the
browser's actual base10 Number formatter for ECMAScript numeric spelling.

`WebPermissions::current` binds requests to the current document.
`request(kind, duration)` returns an owned future for geolocation, camera,
microphone or notifications. Unavailable, rejected, granted, denied and default
results remain distinct, with native rejection code/name/message preserved.
The deadline is positive and capped at the shared 30-second bound. Geolocation
clears its watch on first callback, deadline and drop; two realm-owned callback
dispatchers remain valid for already-queued tasks. Cancelling an unsettled
request leaves its kind busy until actual settlement or document destruction.
Clearing a watch can suppress later settlement, so its retained guard may last
for the document lifetime. Media grant leases stop every track; late acquisition
after cancellation drops the lease before publication. Resource snapshots expose
active requests/watches, media leases/tracks, up to four unsettled guards and
the actual zero-or-two geolocation dispatchers. The provider publishes no location.

`WebViewConfig::with_user_data_folder` admits a caller-owned empty local profile
separate from the mapped package. Runtime profile overrides fail admission.
Native permission callbacks request nonpersistent DENY decisions. With an
explicit fresh profile, host `close` observes the actual normal browser-process
exit before releasing its environment; the caller retains directory cleanup.
Existing default-profile close behavior remains its controller lifecycle.

[ADR 0077](../docs/adr/0077-rust-page-provider-ownership.md) and the
[source-adjacent design](src/wasm/webview/design.md) record the API and ownership
contracts. Portable tests and strict native/WASM/Windows diagnostics establish
source behavior; real Windows repeated-denial/profile cleanup, native page
handoff and consented positive/late media cleanup remain acceptance gates.

## Private native startup verification

The source-adjacent [startup observation contract](src/windows/webview/host/diagnostics/design.md)
defines an unstable fixture-only diagnostic, disabled during ordinary usage.
It observes actual native resources, exceptions and explicitly selected DOM
nodes under bounded retained bytes and the existing navigation deadline.
No public API, injected JavaScript, Ready substitution or permission override is
introduced. Actual Windows observation and consumer startup attribution remain
pending; successful source compilation alone does not establish either.
