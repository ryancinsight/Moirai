# Private startup observation contract

This patch adds no supported public API. The unstable
`MOIRAI_WEBVIEW_STARTUP_OBSERVATION` fixture selection admits exactly the
configured start URI and unique selectors. The dedicated
`moirai_pal::windows::webview::startup` DEBUG target must also be enabled.
Without both conditions no observer or DOM command is installed. Normal
navigation, permission policy, page messages and product filters keep their
existing owners. PAL contains no application selector constants.

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
