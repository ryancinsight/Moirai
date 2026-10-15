# Private startup observation contract

This patch adds no supported public API. The unstable
`MOIRAI_WEBVIEW_STARTUP_OBSERVATION` fixture selection admits exactly the
configured start URI and unique selectors. The dedicated
`moirai_pal::windows::webview::startup` DEBUG target must also be enabled.
Without both conditions no observer or DOM command is installed. Normal
navigation, permission policy, page messages and product filters keep their
existing owners. PAL contains no application selector constants.

## Controller input boundary (0.9.3)

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
