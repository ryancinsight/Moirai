# ADR 0077: Rust page provider ownership

Status: Accepted

Date: 2026-10-13

Change class: [minor]

Target: workspace 0.9.0. Accepted records the selected provider design; runtime
acceptance and release approval remain separate gates.

## Intent and ownership

Metis moves its embedded Windows form/probe behavior into Rust. Moirai owns the
existing WASM WebIDL boundary, Windows WebView2 host and supervised-process
mechanism. It supplies platform operations and resource ownership; Metis owns
page schema, application commands, ready/stop policy, prepared-module identity
and backend authority. The independently versioned Apple host retains its
AppKit/UIKit/WebKit ownership.

The [source-adjacent design](../../moirai-pal/src/wasm/webview/design.md)
defines contracts for message admission, callback lifetime, permission
settlement, process working directories, fresh native profiles and actual exit.
Public additions require the 0.9.0 minor increment. Existing native public limit
names keep their established export location; the shared private definition is
the single owner. This change adds no compatibility aliases or parallel policies.

## Decision

- Resolve the actual `window.chrome.webview` EventTarget and callable method.
  Parse posted JSON into its native object, serialize incoming values and apply
  the existing 64 KiB byte/UTF16 limits. One owned listener and a counted failed
  teardown guard enforce the callback boundary.
- Request actual geolocation, constrained media and three-valued notification
  permission through separate WebIDL domains. Own every track and watch;
  retain bounded settlement guards when cancellation cannot stop browser work.
  Realm-owned geolocation dispatchers prevent queued callback lifetime defects.
- Expose fallible DOM callback close with actual active/retained counts and the
  native ECMAScript Number formatter. Keep application policy in the consumer.
- Set an owned absolute supervised-child directory before resource creation.
  Admit a separate empty caller-owned native profile, request nonsaved DENY
  decisions and observe the actual normal browser-process exit before cleanup.

## Alternatives and failures

Authored application JavaScript duplicates the Rust page policy. Direct WebIDL
bindings in Metis duplicate PAL. Extending the Apple host with Windows browser
mechanisms reverses provider ownership. A process-global directory mutation
changes unrelated subprocess behavior. These alternatives are rejected.

Dropping a Rust closure after `clearWatch` does not retract independently queued
geolocation tasks. Forgetting per-request closures or hiding unresolved Promise
workers loses bounded ownership. Returning synthetic denial, clearing a busy
guard before settlement, accepting a screenshot as readiness or assuming that
controller close released a profile does not establish the requested behavior.

Missing interfaces, malformed/oversized JSON, native rejection, deadline,
teardown exception, invalid child directory, profile override and browser-exit
failure remain observable. Metis may not replace them with successful outcomes.

## Verification and residual risk

Portable production-state tests verify slot cardinality, cancellation guards,
media count arithmetic, finite deadline admission and monotone callback keys.
Native/WASM/Windows warning-denying diagnostics verify their respective ABIs.
Real browser tests exercise message/DOM/Number semantics through generated Rust
interop; they do not establish actual WebView2 native handoff.

The installed native fixture requests geolocation twice and requires two actual
native DENY events, two real browser errors and profile cleanup after actual
exit. Execution on this macOS development host is pending. The consumer's real
Windows readiness, four-kind denial, stop/remount, captures and packaged flow
remain required. Consented ordinary-browser positive grant and late media
cleanup are separately pending; the native host continues to deny permission.

`backlog.md` carries only active todo/blocked acceptance work. No release tag or
complete runtime claim follows from compilation, portable tests or this ADR.
