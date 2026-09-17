# ADR 0062: Deny WebView2 permission requests at the host boundary

Status: Accepted

Date: 2026-09-17

Driver: [MOI-WINDOW-WEBVIEW2-PERMISSIONS-2026-09-17](../backlog.md#MOI-WINDOW-WEBVIEW2-PERMISSIONS-2026-09-17),
[Metis desktop](../../metis/backlog.md#METIS-DESKTOP-001)

## Context

The WebView2 provider already restricts packaged navigation, new-window
requests and page-message sources. WebView2 permission requests are a separate
host callback. Leaving that callback unhandled delegates camera, microphone,
geolocation, clipboard, file, notification and sensor decisions to the
runtime's default profile state, which can include an OS prompt or a persisted
grant.

## Decision

`WebViewHost` installs one `PermissionRequested` callback before loading the
configured page. The callback reads the bounded request URI, maps the runtime
permission kind to the public `WebViewPermission` enum, records the
`IsUserInitiated` snapshot and calls `SetState(DENY)` synchronously. Only after
the deny call succeeds does it enqueue `WebViewEvent::PermissionDenied`.
Unknown numeric kinds remain `WebViewPermission::Unknown` and follow the same
deny path. Teardown removes the callback token before releasing the controller.

The provider exposes observation, not an application allowlist. A consumer can
report the typed denial or choose a separate privileged host; the packaged
WebView2 role never grants page code an OS capability implicitly.

## Alternatives rejected

1. Relying on the page's content-security policy does not cover every
   WebView2 capability and leaves the host default state authoritative.
2. Setting profile-wide defaults without handling the callback leaves a race
   between a request and profile state and hides the request from consumers.
3. Adding a consumer callback that can allow arbitrary kinds would move OS
   authority above the provider boundary and make the secure default depend on
   every application implementation.

## Threat model and limits

The untrusted packaged page can request any WebView2 permission kind. The
synchronous state transition prevents the request from reaching an OS prompt or
profile grant through this host. The event URI and kind are bounded snapshots;
they do not prove page identity beyond the existing packaged-origin policy.
Windows policy, WebView2 correctness and a separate privileged host remain
outside this provider's proof. The callback does not sandbox the browser
process or revoke capabilities already granted by another profile.

## Verification

The native PAL suite checks known and unknown kind mapping and retains the
ignored installed-runtime bridge smoke. An ignored installed-runtime scenario
requests geolocation and asserts a typed denial with `user_initiated = false`.
Strict native Clippy and the locked Windows build compile the callback against
the pinned `webview2-com` bindings; non-Windows and WASM targets do not include
the Windows module. Metis consumes the event and records the visible denial in
its native-host manual.
