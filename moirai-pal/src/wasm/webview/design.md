# Rust page provider contracts

Owner: [ADR 0077](../../../../docs/adr/0077-rust-page-provider-ownership.md).

State: profile/observer and 0.9.2 installed Rust bridge fixtures verified; consumer Rust handoff and positive-grant acceptance pending.

## Dependency and API ownership

```text
consumer page commands/schema/lifecycle
  -> moirai_pal::wasm::{WebViewBridge, WebEventListener, WebPermissions, WebNumber}
  -> actual browser WebIDL and event loop

consumer native host/profile owner
  -> moirai_pal::windows::webview::{WebViewConfig, WebViewHost}
  -> actual WebView2 controller/environment and owner-thread message pump

consumer build orchestration
  -> moirai_transport::process::{ProcessSpec, ProcessSupervisor}
  -> child-only OS working directory and contained process lifecycle
```

No provider imports Metis messages, commands, backend calculations or page
policy. Public types export through their existing crate/module facade; private
leaves own their respective operation domains. `webview_limits.rs` owns the
message byte/unit/event and operation wait constants shared by native/WASM.

## Message and callback law

`WebViewBridge::current` checks the actual native object and callable methods. Absence is
Unsupported; invalid shape and thrown native operations retain context. Posting
validates supplied JSON before parsing and native serialization, then calls
postMessage with its object value. Receiving accepts actual object/string data
and validates its complete serialization. Admission requires UTF8 bytes and
UTF16 units each at most the shared 64 KiB limit. These bounds do not claim to
cover JSON parser or serializer private allocations.

One `WebViewMessageListener` owns the document's native-message callback.
Successful explicit close detaches before releasing the Rust closure. A failed
close leaves the handle active; a failed Drop moves its existing ownership to
one counted realm guard. New attachment returns WouldBlock while an owner or
guard exists. Browser destruction releases the remaining realm.

### Checked native method contract — 0.9.2 [patch]

The installed consumer in [Metis run 38531378051](https://github.com/ryancinsight/metis/actions/runs/38531378051)
loads all five resources with correct MIME types, then Rust reports
`WebView2 bridge is not an EventTarget`. Native DOM observations identify this
nominal cast as the rejection. Microsoft's [WebView interface](https://learn.microsoft.com/en-us/microsoft-edge/webview2/reference/javascript/webview)
specifies `postMessage`, `addEventListener` and `removeEventListener`; the
observed object fails this realm's DOM prototype identity test.

The private handle checks a non-null object and caches all three callable
properties once. Each invocation uses the same actual receiver. Missing or
noncallable members and getter/call exceptions retain their operation context.
No prototype mutation, unchecked native-object cast, injected page code or
replacement host object supplies compatibility. Public signatures stay unchanged.

The listener retains the exact callback, receiver and cached methods. Failed
registration requires matching removal before callback release. If removal
fails, the same single realm guard retains ownership and the returned error
preserves both failures. Successful registration keeps the existing close/drop
law. At most one active or retained listener exists; failed registration never
reports successful attachment or cleanup.

The installed consumer supplies the pre-correction failing case. Positive
regression compiles Rust/WASM and exercises current/listen, a real native JSON
roundtrip, close and reattachment inside installed WebView2. Ordinary browser
tests verify checked extraction and negative values, not native delivery. All
message bounds and native fixture deadlines remain unchanged.

Local verification runs 14 actual Chrome contracts with no skipped cases,
including seven structural checks on real browser objects. Strict WASM and
Windows PAL diagnostics pass; the actual Cargo/bindgen producer retains its
original generated payload and selected-source/tool/lock hashes. The full
workspace passes 1,310 native tests with nine existing ignored probes. These
checks do not establish installed Windows bridge delivery. The dedicated
[compiled Rust fixture](../../../examples/webview_bridge_fixture/design.md)
requires native roundtrips, close/remount and actual profile destruction in CI.
The [installed Windows run](https://github.com/ryancinsight/Moirai/actions/runs/38539086879/job/115769121797)
at source `e15604d39dd4a9fe7912bc0c39f01cbacae2cfde` passes the Rust fixture
in 8.155 seconds. Both directions admit 65,536 bytes and reject 65,537 bytes;
Unicode/string identities and computed sums 65,793/197,379 survive both mounts.
The first retired callback remains at three calls, the second computed reply
reports two calls, and the final guard count is zero. Actual Destroyed and
package/profile absence assertions pass. All five resources return 200. Final
observer closure has no pending request or observed failure. Observed exception
absence covers records received before teardown; it does not prove every
possible future browser dispatch is error-free. Native consumer acceptance
remains separate.

`WebEventListener` applies the same detach-before-release rule to DOM events.
On the first failed Drop, let A be the finite number of already-admitted live
owners. Further admission stops; retained owners R obey 0 <= R <= A. Ownership
transfer uses each existing node, and realm destruction drains the list without
recursive stack growth. Counts report actual active and retained ownership;
failed removal never claims a callback was freed.

`WebNumber::decimal` calls actual Number.toString with radix ten. WebIDL Number
is binary64; no precision conversion or application formatting algorithm is
introduced. Browser decimal/exponent thresholds, signed-zero and nonfinite
spelling remain the browser's computed result.

## Permission law

`WebPermissions::current` acquires one current Document's shared registry.
`request(WebPermissionKind, Duration)` returns an owned future; zero duration
rejects, while a positive deadline becomes min(caller duration, 30 seconds).
The timer starts before the actual API request. A provider bound from a prior
document rejects new requests. There are exactly four admission slots:
Geolocation, Camera, Microphone and Notifications.

For each slot, the transition law is:

```text
Idle -> ActivePending -> ActiveSettled -> Idle (consumer observes result)
                   \-> DetachedPending -> Idle (actual late settlement)
ActiveSettled -> Idle (consumer drops an unobserved result)
```

ActivePending cancellation first releases the actual watch, then retains its
settlement guard. ActiveSettled cancellation drops any acquired lease.
Per-document counts obey active_requests <= 4, geolocation_watches <= 1 and
unsettled_guards <= 4. Monotone checked generations never wrap; stale or duplicate
callbacks cannot settle a later request. The returned snapshot also reports
media leases/tracks and actual realm geolocation dispatchers.

The actual watchPosition first result/error clears its native watch before
settlement. [Geolocation sections 6.4–6.7](https://www.w3.org/TR/geolocation/)
remove watch IDs independently from already queued callbacks and permission
waiting. Therefore two realm-owned Rust dispatchers remain after first use,
until document destruction. Native bind attaches a never-reused nonzero u64
key encoded as a canonical decimal string, preserving every bit through JS.
The dispatcher removes its pending record before clearing/settling. Late or
duplicate bound callbacks find no newer owner under that key.

A cleared watch may not callback again. Its one DetachedPending guard then
keeps Geolocation busy until document destruction. This is reported ownership;
it is not a completed cancellation claim. Pending records retain only native
registration/registry state and a weak request consumer, with no page elements
or location data. The provider publishes no coordinates.

Camera/microphone use one getUserMedia algorithm whose kind selects video-only
or audio-only constraints. An actual stream becomes `WebMediaLease`; its drop
stops every native track. A late successful acquisition after cancellation
creates then drops that lease without publication. Unsettled getUserMedia and
requestPermission Promises have no abort parameter; each kind's guard remains
busy until genuine settlement or document destruction.
[Media Capture](https://www.w3.org/TR/mediacapture-streams/) owns track lifecycle.
[Notifications](https://notifications.spec.whatwg.org/) returns exactly granted,
denied or default. Unavailable APIs, actual rejection with native code/name/
message, grant, denied and default remain distinct outcomes. No grant or denial
is inferred from a timeout, a completed count or a missing device.

## Native directory, profile and exit law

`ProcessSpec::working_directory` stores an owned absolute child directory.
Spawn validates native representation and an existing directory before opening
process resources. Unix Command.current_dir and Windows CreateProcessW apply
that directory to the child; the parent cwd does not change. OS failure retains
the operation and native code. Relative executable lookup is platform-specific,
so deterministic consumers select an absolute executable.

`WebViewConfig::with_user_data_folder` requires an existing empty local drive
directory, bounded native Unicode representation, no relative/link/reparse
component and no overlap with mapped package resources. The caller creates and
exclusively owns both directories; the provider revalidates before COM creation
and checks the runtime's effective profile. Runtime overrides fail admission.

Permission callbacks set actual DENY, request nonsaved profile decisions and
mark the request handled. The fresh profile supplies independent per-request
native evidence; an old shared profile does not establish that evidence. Native
policy does not grant permission for positive browser fixtures.

The SDK [permission Uri](https://learn.microsoft.com/en-us/microsoft-edge/webview2/reference/win32/icorewebview2permissionrequestedeventargs#get_uri)
reports the requesting origin, while WebMessageReceived.Source reports the
document URL. Actual PR 619 CI observes `https://metis.invalid/` for denial and
`https://metis.invalid/index.html` for a page message. Two same-document API
calls produce one native denial and two browser errors in that run; call counts
do not imply native callback counts. The repaired fixture navigates to a second
packaged document after the first actual native/browser completion under the
same profile, origin and controller. It still requires two actual denials and
two actual errors.
The repaired Windows job [115537873626](https://github.com/ryancinsight/Moirai/actions/runs/38467480001/job/115537873626)
passes at [reviewed source](https://github.com/ryancinsight/Moirai/commit/0d42510b50cec901852124ddbdcd0727632ebcb9), observing the real two-document
denials/errors, browser exit and profile cleanup. This provider evidence does
not establish the Metis Rust module handoff or four-kind native page acceptance.
The run passes 25 native policy/destruction tests and four actual capture
fixtures, including two 100-host cases, with WebView2 `153.0.4234.48` and SDK
bindings `0.39.1`. The repeated two-document fixture completes in 7.410 seconds.

Fresh-profile close registers BrowserProcessExited for the actual nonzero
browser PID before closing callbacks/controller. The owner-thread pump observes
the matching normal exit within the existing operation deadline, then releases
the environment/window. Cleanup attempts continue after errors; the first
failure remains observable on repeated close. The provider does not remove the
caller-owned directory. Default profiles retain controller-only close behavior.

## Executable acceptance

- Portable state/key tests inspect exact slot counts, busy errors, media sums,
  cancellation/settlement, stale generations, decimal identities around 2^53
  and u64::MAX, overflow and deadline boundaries.
- Shared message/cwd tests inspect actual computed length/admission and native
  child directory/output without global-directory mutation. Rust browser tests
  inspect JSON values/bounds, real DOM registration/removal and Number results.
- Installed Windows geolocation requests across two packaged documents under
  the same profile/origin/controller must produce two actual
  native DENY events and two browser errors, then actual browser exit and caller
  profile cleanup. Source [reviewed source](https://github.com/ryancinsight/Moirai/commit/0d42510b50cec901852124ddbdcd0727632ebcb9) passes the actual
  Windows job above; this macOS host supplies no local Windows execution claim.
- Consumer acceptance must use the actual Rust-ready native page and four
  native permission kinds, then stop/remount with exact reported resources.
  Unexpected grants, unavailable required APIs and deadlines fail acceptance.
- With explicit human consent in an ordinary browser and real devices, observe
  grants and every native track stopping on lease drop and late acquisition;
  check real watch clearing and notification granted/denied/default. Ignored
  requests retain their reported guards and block replacement until settlement.

Native/WASM/Windows warning-denying diagnostics and minor compatibility checks
are separate gates. No source/property test substitutes for the actual native
or consented positive fixtures; no provider-private allocation total is claimed.
