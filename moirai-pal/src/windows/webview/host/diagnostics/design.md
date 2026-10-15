# Private startup observation contract

## SDK admission and ready events (0.9.4)

Given a positive selected-observation wait entered at `t` with duration `d`,
its deadline is the single checked `D = t + d`. Native dispatch, callback
preparation and JSON/native-string encoding spend that budget. Immediately
before each outgoing diagnostic SDK call, admission requires `now < D` and
no already-produced public window or WebView event. A ready public batch
returns before a fresh sample or pending-chain command can enter the SDK.
Explicit polling and zero-duration calls have no wait deadline; they still
give ready public events priority. Ordinary hosts retain their existing pump.
Constructor activation uses its original navigation deadline, before there
is an application event consumer.

The command state is `in flight -> reply -> prepared -> in flight`, with at
most one occupied request slot. If readiness/deadline rejects admission,
`prepared` retains the exact command, parameters, stage and request identity;
it is not a failed or attempted SDK call. A later admissible caller turn
continues that command before considering a new sample. Generation changes,
quota failure and close retire unissued work; an actual in-flight callback
keeps its existing weak completion owner. No reply disappears, request is
duplicated, deadline is renewed or private wake admits a second sample.

An earlier boolean sampled before native dispatch cannot prove admission at
the SDK boundary. Dropping a consumed reply on denial loses a stage, and
starting another request leaks or reuses its identity. Retaining one prepared
command avoids both failures without caching DOM results or changing the
sampling domain. No sleep, input retry or new interval enters this policy.

Regression scenarios exercise an actual native resize with an idle sampler,
a second real resize produced by the final predicate after command setup,
an actual intermediate DOM reply deferred behind an event, and a real
expired deadline at the SDK admission seam. They inspect actual command
counters and request identities, then resume real DOM work and verify computed
selector values. macOS cross-compilation cannot execute those Windows runtime
scenarios. The original eight typing modes and full Metis ordinary/selected
journeys remain separate gates; these scheduling defects do not establish the
cause of the intermittent form-input failure.

This patch adds no supported public API. The unstable
`MOIRAI_WEBVIEW_STARTUP_OBSERVATION` fixture selection admits exactly the
configured start URI and unique selectors. The dedicated
`moirai_pal::windows::webview::startup` DEBUG target must also be enabled.
Without both conditions no observer or DOM command is installed. Normal
navigation, permission policy, page messages and product filters keep their
existing owners. PAL contains no application selector constants.

## Controller input boundary (0.9.4)

The same private selection additionally observes GotFocus, LostFocus,
MoveFocusRequested and eligible AcceleratorKeyPressed callbacks on the actual
controller. Each callback copies the SDK event kind, virtual key, signed lParam
and existing Handled value, plus non-atomic owner focus/foreground and raw
Control/Shift/Alt key states. Owner-thread GetKeyState and system-wide
GetAsyncKeyState retain distinct labels. Sequence and monotonic elapsed time
belong to this observer, not the page protocol or Python actor clock.

These synchronous callbacks enqueue only Copy records into preallocated fixed
storage of MAX_WEBVIEW_EVENTS elements. They perform no formatting, subscriber
call, disk I/O, outgoing cross-process COM, Handled setter or focus operation.
The existing owner pump drains records after callback return through the same
aggregate byte/event quota. Capacity loss and reentrant contention are explicit
incomplete-evidence records; neither overwrites an earlier observation. Close
drains the final batch and attempts all four token removals before controller
close, preserving the first error after remaining cleanup.

Accelerator callbacks prove that an eligible key reaches this controller.
They do not prove renderer default action, DOM value change or Rust submission.
Unicode packets are not all eligible accelerators. Missing callbacks and a
collapsed UIA range do not establish where text is lost. Input retries, extra
sleeps, acceptance predicate or time-limit changes do not follow from these
observations. The installed-runtime fixture exercises real
foreground/controller focus and physical Ctrl+A/Escape, checking SDK values,
unchanged Handled, bounded output and callback/profile cleanup.

[SDK callback contract](https://learn.microsoft.com/en-us/microsoft-edge/webview2/reference/win32/icorewebview2controller?view=webview2-1.0.4022.49#add_acceleratorkeypressed).

### Native parent focus regression

Metis source `05a2e7c` completes its first Unicode calculation, then retains a
top-level FocusGained/FocusLost pair and controller LostFocus before its second
patient replacement. UIA reports focus and an owned renderer HWND, but no
further eligible keys reach the controller. This is distinct from script
overhead: all 502 Windows scripts pass in 34.004 seconds.

For a host containing one controller, an actual parent WM_SETFOCUS must enter
that controller before returning. PROGRAMMATIC focus preserves the previously
focused page element. The regression starts with actual controller focus,
calls native SetFocus(parent), requires an actual descendant recipient, and
sends the same six physical Ctrl+A/Escape records without a subsequent
test-side MoveFocus. Existing Handled, scan/release and cleanup assertions
apply to both entry paths. [Baseline Windows run 38748681812](https://github.com/ryancinsight/Moirai/actions/runs/38748681812),
source `7a5b3073`, fails that assertion in 1.957s because parent and recipient
are the same HWND. Its original controller-entry fixture passes in 7.193s.

The correction binds one private Weak native-window focus target to a stable,
host-owned Rc controller target. WM_SETFOCUS calls PROGRAMMATIC synchronously
after releasing every WindowState borrow; queued event replay could let later
input precede forwarding. Window and WebView dependencies remain one-way. Bind
also forwards an already-focused parent, only when GetFocus equals that parent;
it never claims focus from a different recipient. Close revokes the weak target
before controller shutdown and preserves pending native failures. MoveFocus
failure retains its operation, HRESULT and typed SDK source. The regression
also calls the actual closed controller and checks that failure retention.
No raw host address, subclass context, Handled setter or input synthesis enters
this route. Metis's unchanged full journey remains the causal acceptance gate;
[corrected provider run 38749728275](https://github.com/ryancinsight/Moirai/actions/runs/38749728275)
passes the real parent-entry regression in 7.028s with the original key,
Handled, native error and cleanup assertions. Source bd88d2c5 passes all nine
installed fixtures and 31 policy/frame cases. Full consumer Windows results
remain separate: Metis b7ad7405 run 38750658426 still times out replacing the
second dose (0.25 retained after 0.5 requested), with a Lost/Got/Lost sequence
and no later key receipt. A real native MTA UI Automation fixture now exercises
all sixteen text/numeric replacements from that journey using actual SetFocus,
physical Ctrl+A and exact UTF-16 packets. It observes unique controls and exact
ValuePattern units within one unchanged 30-second runtime deadline. It supplies
no page script, SetValue, input retry or synthetic controller events. Its result
must establish the focus mechanism before another production correction.

This fixture is an isolation, not pipeline equivalence: it uses the native
CUIAutomation MTA client in an owned child test process and separate four-key
select-all/Unicode batches. The actual Metis PowerShell actor records STA and
additionally interleaves selection/native ownership reads and WM_NULL. The
child process admits no input outside the owned foreground/native tree, keeps
all four actual field values, and verifies that each replacement preserves
every other field. Parent admission stops at 30 seconds, then owns child
termination/collection before controller/profile cleanup; nextest retains its
existing strict 60-second limit. Primary and cleanup errors remain separate.

Source cde46135's actual Windows run 38756775119 fails in 2.225s on its first
immediate HasKeyboardFocus read, before any physical input. Its five controller
records are Got/Lost/Got/Lost/Got; this is not the persistent b7 dose failure.
The refined fixture observes HasKeyboardFocus completion within the same
absolute 30-second bound and retains its apartment, sample count and elapsed
time, then performs each input exactly once. Current upstream Chromium forwards
the UIA focus action and returns S_OK independently from its separate focused
property read; that architecture supports this observation protocol but is
not a source match to the installed WebView2 runtime.

The child emits a typed computed completion bound to the actual HWND, sixteen
completed replacements, sixty-four preservation checks and all four final raw
UTF-16 values. Successful exit with no matching receipt rejects, including a
stale libtest selector. Its bounded stdout contains no browser descendants;
ordinary spans remain on inherited stderr. Native stop/collection errors remain
failed acceptance even after remaining cleanup attempts; host close can unblock
a pending remote UIA call, and final collection precedes profile cleanup.

Fadaeaaa's refined native run 38758440683 observes focus for all first-row
fields (13/7/2/1 samples), then reaches its original 30-second admission bound
in the dose path. Its tail does not identify which native operation remains
active or retain the current dose. The next isolation therefore runs both
ordinary and selected modes against the same production source and records
bounded operation entry/return, native insertion counts, recipients and actual
value changes. Neither mode drops the 16/64 domain or extends a time limit;
both results are required. Actual collection and host-close evidence admit
directory removal, including unwind paths, and unavailable proof retains the
owned directories as a failed cleanup outcome.

### Controlled observation-component comparison

Baseline `23561b59` / Windows run 38763941253 completes ordinary mode's
16 replacements and 64 preservation checks in 5.683s. Selected mode admits
weight 64 input while the last changed ValuePattern remains 60, then reaches
its unchanged 30-second bound. Its fixture log quota ends after query 99;
the final active call is unavailable. Controller Lost/Got/Lost precedes the
missing subsequent eligible key receipts. The observer completes 8,672 DOM
chains (at least 52,032 CDP calls); without per-chain timestamps this proves
additional work, not its timing or input-loss causality.

Private-parent experiment `27cd682d` / run 38766339399 preserves the fixture,
bounds, runner and available WebView2 154.0.4258.62/SDK 0.39.1. It passes 33 policy
and ten installed cases, including real inner focus, sibling/foreign focus,
resize and destruction. Ordinary mode completes 16/64 in 7.428s; selected mode
completes 6/24 then retains concentration 2 after admitted input requests 4.
The next observed failing field does not establish a correction. The private
container and its experiment-only assertions are withdrawn; source history
and the exact terminal receipts remain reproducible. The independently proved
outer-parent correction and original acceptance gates remain.

The private selection now accepts a closed optional `components` enum:
`full` (default), `controller`, `native`, `events`, `domain`, `snapshot`, or `startup`. Existing JSON therefore retains
all observers. Controller-only installs the actual four controller callbacks,
without lifecycle/resource/CDP receivers, domain enable or DOM tick. Its empty
real token owner retains common slot invalidation and terminal metadata.
Startup-only installs the existing native/CDP observation machinery without
controller callbacks. This includes resource/lifecycle, Runtime/Log enable and
DOM sampling; its label does not claim DOM-only isolation. No callback or
computation is substituted. Records identify actual retained token counts and
checked counts of attempted activation/DOM SDK calls. No state borrow crosses
those calls, and the existing quotas still bound all retained records.

The same real native client now executes ordinary, controller-only,
native-only, events-only, domain-only, snapshot-only, startup-only and full selected modes, each with all 16 edits, 64 exact value
checks and unchanged 30/60/120-second client/nextest/wrapper bounds. Both
original modes remain fatal and all eight run before aggregate failure.
Successful isolated modes additionally verify actual receiver identities,
zero CDP calls for controller/native-only, exactly two activation calls and no
DOM sampling for events-only, real selected nodes and completed chains
for startup/full, and terminal token cleanup. Partial-mode success never
substitutes for full selected or either unchanged Metis journey.

The initial four-mode source `4e7b1968` / run 38769791520 passes ordinary
16/64 in 3.812s and controller-only in 3.136s. Startup-only reaches the same
30-second bound at weight 64 (60 retained); full selected fails its first
reference replacement (demo retained). Both observe focus and admit every
requested physical record. The controller callbacks are therefore unnecessary
to reproduce. Startup-only reports 18,759 completed chains/112,555 DOM calls;
full reports 20,113/120,681. Activation is exactly two and observer loss/error
is absent in both; one request remains pending. This group-level result does
not identify the final native call or attribute a particular CDP method.
Native-only retains the six lifecycle/resource receivers without CDP or input
callbacks. Events-only adds the three actual CDP receivers and original
Runtime/Log activation, with no DOM tick. These additional controls distinguish
the recurring sampling algorithm from passive subscription and activation;
the original four modes and every bound remain unchanged.

Document-binding experiment 8c49630c / run 38774232219 is also negative:
startup/full each get the document once, then perform 14,157/13,948
native descriptions and still retain demo after admitted first-reference
input. All six actual owning environments report 154.0.4258.62; passive
controls complete 16/64. Removing repeated binding resets is insufficient.
That candidate and its extra frontend-binding lifetime are withdrawn. Actual
per-environment/per-method evidence remains; descriptions now stay zero under
the original sampler. No correction is adopted from the failed candidate.

Domain-only adds real DOM.enable to Runtime/Log activation (three activation
calls) but issues no sample. Snapshot-only completes one original
getDocument/querySelector/getOuterHTML chain, checks actual selector values,
then admits no more starts; its slot must be settled at close. These are
additional causal controls, never acceptance substitutes. Original ordinary,
controller/native/events, startup and full behavior remains unchanged. Every
physical input/value and 30/60/120-second bound remains. The next evidence
distinguishes initial domain attachment or one snapshot from continuous work,
before a scheduling correction. Activation counters include DOM.enable;
DOM-attempt counters count sampling calls separately.

### Caller-owned observation turns

Eight-mode source 197924cd / run 38776749790 completes ordinary, controller,
native, events, DOM-domain and single-snapshot controls (all 16/64) under
actual runtime 154.0.4258.62. Only continuous startup/full sampling fails.
Domain has three actual enables and no samples; snapshot has one getDocument,
six native sampling calls, one complete chain and no pending request. Initial
attachment or a complete snapshot is therefore insufficient to reproduce.

Before source9cf4d47a, the wait returned an empty application batch after an internal CDP
completion. The caller immediately waited again, letting those completions
drive fresh sampler starts. That correction coalesces positive waits for
continuous observation; the later SDK admission correction carries the same
entry deadline through every positive selected wait. Its first poll admits at most one chain. Subsequent
private wakes still dispatch every owner-thread message and advance/settle
existing stages, but admit no fresh chain. Actual application events return
immediately; otherwise one original caller-supplied deadline ends the wait.
No callback restarts that deadline. Ordinary and zero-duration paths preserve
their native pump behavior. Public direct polling admits at most one chain,
after ready public events. No fixed sampling interval or sleep is added.

For W logical wait/poll turns, A fresh admissions satisfy A <= W. A chain with
n selectors performs at most 1+2n native calls, so a measured window has
D <= (1+2n)(W+1), including one initially pending chain. Existing byte/event
quotas and callback ownership remain independent. Native fixture turns use
their existing 25ms cap; Metis supplies its existing 250ms cap. Neither value
is a new interval or a changed acceptance deadline. Fresh URI/selector/HTML
reads and original document-binding lifetime remain unchanged.

The eight-mode gate additionally requires at least two complete samples for
each continuous mode (the minimum that exercises recurrence), actual document
starts bounded by fixture wait turns, and the derived native-call bound.
Snapshot must still complete exactly one chain. Pure transition coverage
settles every document/query/HTML stage with fresh admission disabled, then
admits work on the next caller turn. Exact zero/fractional/inclusive native
deadline tests preserve the existing millisecond conversion. These Windows
tests pass in Windows run38779895376 (36 policy cases). That run still fails
startup-only on final dose0.125 after15/60 checks; actual UIA remains0.5.
Full selected16/64 passes; quota exhausts during the run, so it does not establish
active observation throughout the journey. Startup-only remains active and
within the derived work bound: coalescing alone is insufficient. All eight
owning environments report154.0.4258.62. Both complete Metis journeys remain
required. Its later coherent Gaia/Eunomia/Mnemosyne/Hermes/Consus upgrade admits
the aa532 provider in both locks; that resolution does not prove form delivery.

The [opt-in live-value probe](../focus/tests/renderer.md) now observes one
DOMSnapshot after a returned persistent UIA mismatch. Its separate execution
preserves the original eight-mode gate, input, observations and all deadlines.
Exact sparse live values can distinguish attribute/AX observations from
renderer state, subject to layout flushing and non-atomic read timing.


The original navigation deadline starts before registration/activation.
Runtime and Log listeners register first; successful `Runtime.enable` completion
precedes `Log.enable`, whose successful completion precedes Navigate. Every
wait borrows the remaining original deadline. No Ready event, retry, extra
startup wait, JavaScript evaluation or network interception is introduced.

Native passive observations retain ContentLoading, DOMContentLoaded, process
failure and resource response URI/status/Content-Type. CDP listeners retain
actual Runtime.exceptionThrown and Log.entryAdded JSON. Response bodies and
other headers are never read. Registration, command, decoding and removal
failures remain explicit observations, never absence interpreted as success. A
failed HRESULT retains its independently supplied bounded reply and decoding
error; successful malformed/protocol-error replies retain their actual text.

A single request slot owns one DOM chain: getDocument(depth=0), querySelector,
then getOuterHTML for each selected node. A zero node ID records a missing
selector. No substitute node is queried. Chromium discards frontend node
bindings on getDocument, so chains never overlap. Actual NavigationId matching prevents overlapping old completions from changing
the selected document phase. getDocument.documentURL must exactly match the
selected URI before any node query. Navigation or actual DOM.documentUpdated invalidates the
current generation; its pending completion retains the slot until it settles.
All completion closures hold Weak state, not a strong WebView/controller.

Existing owner-thread polls coalesce subsequent samples while the selected
document is available. Unchanged selector values do not produce duplicate
records; changed values, missing values and actual errors do. This is a series
of point observations, not a guarantee that every transient DOM state is seen.
A native error stops that chain; close invalidates pending requests without
waiting. Late replies cannot release another request. No callback blocks the
STA message pump or extends the existing navigation/Ready deadlines.

The existing 64 KiB message byte budget bounds the aggregate encoded ordinary
observation JSON; the existing 256 event bound caps admitted records. UTF16
scans and UTF8 copies use existing provider bounds before parsing. Complete
records are admitted atomically. Exhaustion is sticky and explicit, and stops
new sampling. Two fixed-schema terminal records report quota and close state.
The selected tracing layer omits time/span context. Its exact JSON envelope
bound is verified by the portable owner against the existing 1 MiB Metis trace
limit, including the second JSON-string escaping layer and newline. These
bounds cover Rust-retained copies, not Chromium allocations before callbacks.

The terminal `closed` field refers to request admission, not a claim that a
failed token removal released the SDK callback. All callback tokens remain
owned until removal before controller close. Every
removal is attempted; the first actual error is returned after remaining
cleanup. Pending completion and failed-removal state are reported. Disabling
CDP domains is not attempted during close: it would create new asynchronous
work after cancellation. Removing receivers and invalidating Weak state revoke
this owner's observation path; the controller then closes through its existing
owner.

Primary contracts:
- [WebView2 CallDevToolsProtocolMethod ordering](https://learn.microsoft.com/en-us/microsoft-edge/webview2/reference/win32/icorewebview2?view=webview2-1.0.4129.50#calldevtoolsprotocolmethod)
- [WebView2 threading model](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/threading-model)
- [CDP DOM methods](https://chromedevtools.github.io/devtools-protocol/tot/DOM/)
- [CDP Runtime exceptionThrown](https://chromedevtools.github.io/devtools-protocol/tot/Runtime/#event-exceptionThrown)

Verification requires strict portable selection/quota/state properties, actual
SDK compilation and a real installed-runtime fixture that observes known DOM
values and a missing selector. Actual Metis startup attribution remains pending
until its next Windows run retains resource, exception and selected DOM output.
