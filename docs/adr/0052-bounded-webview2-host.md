# ADR 0052: Bounded Windows WebView2 host

Status: Accepted

Date: 2026-09-09

Driver: [MOI-WINDOW-WEBVIEW2-2026-09-09](../backlog.md#MOI-WINDOW-WEBVIEW2-2026-09-09),
[Metis desktop](../../metis/backlog.md#METIS-DESKTOP-001)

## Context

Metis needs a desktop shell that can render its existing HTML5, CSS and
WebAssembly bundle while keeping the native window, callback ownership and
message pump in Atlas infrastructure. Moirai owns a bounded Win32 `HWND`, but
it does not yet own a browser controller. No Atlas repository contains a
WebView2 provider. Retaining a toolkit runtime would duplicate the shell and
would keep the Tauri, Wry, egui, GPUI or Iced runtime choice in every consumer.

## Decision

Add a Windows-only WebView2 provider below `moirai-pal::windows`. The provider
consumes and owns one `NativeWindow`, initializes COM on its creating thread,
creates one WebView2 environment and controller, and exposes only Rust-owned
configuration and value events. The owner thread pumps its message queue with a
finite deadline; callbacks never wait on an unbounded channel or execute while
the provider state lock is held.

The initial navigation is a validated `file:///` URL. The policy derives one
directory prefix from that URL, rejects traversal and encoded separator
segments, and cancels every navigation outside that prefix. New-window
requests are always handled and denied. Web messages are capped before they
enter the retained queue, retain their source URL, and are delivered in
arrival order. The queue has a fixed capacity and reports overflow on the next
poll. Outbound JSON is bounded and NUL-free before WebView2 receives it.

The provider stores every callback token and removes it before closing the
controller. `Drop` performs only synchronous COM release and controller close;
fallible shutdown is available through `close` and never blocks or awaits.
`WebViewHost` owns its `NativeWindow`, so the parent HWND cannot outlive the
controller and no raw window handle crosses the public PAL boundary.

The implementation uses `webview2-com` 0.38.2 and its generated
`webview2-com-sys` bindings only at this Windows ABI boundary. This is a
justified external dependency: the Atlas stack has no WebView2 implementation,
and the generated bindings preserve the operating-system COM contract without
adding a GUI runtime. The existing `windows` dependency is advanced to the
same 0.61 line required by those bindings. The provider does not import Wry,
Tauri, egui, GPUI or Iced.

## Alternatives

Wry or Tauri runtime embedding was rejected because it would reintroduce a
second event, process and resource policy layer. A handwritten WebView2 COM
ABI was rejected because generated bindings already ship the loader ABI and
reduce pointer-layout risk. A custom HTTP server or unrestricted `http` URL was
rejected because it expands the renderer trust boundary and makes package
contents depend on a network endpoint. `NavigateToString` as the production
path was rejected because it cannot prove that the packaged HTML/CSS/WASM
inventory is the page being rendered.

## Safety and failure behavior

All WebView2 and COM calls stay inside the Windows provider. Each unsafe block
states the COM apartment, thread, pointer and callback lifetime assumptions.
Callback errors are converted to HRESULTs and never unwind through the COM
ABI. UTF-16 extraction is bounded and rejects malformed input; queue and
message limits fail with typed I/O errors. A failed callback removal or close is
reported by `close`; the Drop fallback still releases the controller and owned
callbacks. No navigation, message or runtime failure silently falls back to a
network page or software-only substitute.

## Verification

Policy tests cover allowed descendants, network and traversal rejection,
encoded-separator rejection, URL and message bounds, and new-window denial.
The current increment compiles the real Windows provider and exercises the
native window lifecycle plus policy boundary; a runtime capture against an
installed WebView2 runtime and the Metis packaged HTML/CSS/WASM bundle is the
consumer integration increment. An ignored integration smoke is checked in for
that host and verifies the packaged page, bridge message and denied navigation
when the runtime is present. The provider's finite pump bound and queue
overflow are value-tested without sleeps or polling loops.

The Windows loader and browser runtime are system prerequisites; non-Windows
and WASM builds omit this module. Miri cannot execute COM or Win32, so the
unsafe boundary is covered by the native Windows lifecycle test and source-level
safety review. Runtime registration and the Metis packaged-bundle capture
remain the consumer integration increment.
