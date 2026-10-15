# One-shot live renderer value observation

This is an opt-in fixture diagnostic, not a replacement for any acceptance
assertion. The normal eight-mode matrix explicitly removes
`MOIRAI_NATIVE_UIA_RENDERER_PROBE`. A separate startup-only invocation sets it
to `1`, preserving all sixteen physical replacements, sixty-four preservation
checks and the original 30/60/120-second client/test/command bounds.

After the first scope's existing evidence quota exhausts, a returned
CurrentValue mismatch can emit exactly one typed stdout record. The record
binds the native HWND, mode, replacement ordinal, field, requested UTF-16
units, actual native units and query ordinal. The parent validates it against
the declared input domain. A blocked getter cannot fabricate a trigger.

The parent peeks at the existing stdout pipe, reads no more than the available
bytes and frames complete UTF-8 lines within the same aggregate 64KiB output
bound. Drained bytes remain part of the original schema2 completion admission.
Malformed/duplicate progress rejects. Progress first observed after collection
cannot start a command; it is explicitly unobserved during the active journey.

One actual `DOMSnapshot.captureSnapshot` request uses empty computedStyles and
disables paint order and DOM rectangles. Setup does not renew the deadline;
the parent checks its original remaining deadline immediately before SDK entry.
Admission, callback HRESULT/body, live values and unobserved completion remain
distinct. No new waiting, input retry, SetValue, injected JS or DOM setter is
introduced. The callback owns only Weak independent state, never the normal
startup RequestSlot or a controller. Close invalidates it before host Close.

The decoder selects the exact document URI and frame, joins nodeType/nodeName,
attributes and backendNodeId by node ordinal, and requires four unique INPUT
identities and backend IDs. Sparse inputValue indexes must be unique/in-range.
An explicit string index `-1` represents an empty live value; an absent sparse
entry does not. All other string indexes must resolve. UTF-16 units are derived
without normalization from admitted Unicode strings. Unknown/malformed data
fails the probe, rather than returning attribute values or fabricated defaults.

The pinned Chromium `InspectorDOMSnapshotAgent::CaptureNode` reads
HTMLInputElement::Value(), while outerHTML serializes the value attribute.
The protocol schema and Chromium source contract are retained in the paired
investigation receipts. Installed Edge support remains an execution question.
CaptureSnapshot flushes layout and may alter timing. UIA and CDP reads are not
atomic or clock-synchronized: a requested value at snapshot execution does not
prove it was present during an earlier UIA read. Parent trigger/admission and
callback timestamps share only the parent's monotonic clock. Every conclusion
must retain that temporal and observer-effect limit.

Accessibility.getPartialAXTree is an alternative, but it activates/updates an
AX context and numeric controls can report numeric rather than exact textual
values. DOMSnapshot reads the input's live string and correlates all four
fields in one capture. OuterHTML cannot answer this question; injected JS would
change the established diagnostic boundary. Decoder/framing/owner/deadline
unit cases and strict MSVC Clippy cover the source contract; the installed
Windows invocation must establish actual method support and retained values.
