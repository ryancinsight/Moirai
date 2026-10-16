# Real document keyboard fixture

The browser payload is identical for Default, EnableS and SkipS in both native
host observer groups (selected and none), plus both new observer-none/Default
Settings3 controls. Selection/global settings vary only in native test owners;
the payload reads neither and implements no shortcut policy. The only authored
initializer imports the generated module and awaits
its normal wasm-bindgen initialization. No synthetic event, evaluated script,
preventDefault, retry, refocus or deadline exists in this payload.

## Wire and real observations

One closed shared protocol defines Command Close, Notice Ready, Notice Key and
Notice Closed. Empty variants use empty structs, so unknown fields reject.
The optional publication_error field is required as an explicit null/string.
The same source is included by the actual native Rust fixture. Document keydown
uses canonical WebDocument event ownership, KeyboardMetadata getters and the
checked WebViewBridge. Every Key retains the complete key/code and six named
Boolean values. These independent observations admit all2^6 combinations;
portable wire tests verify named serialized fields and decoded values. This is
codec evidence, not a native-event construction or execution claim.

The one-based key_events counter measures actual keydown callbacks before
closing, including the callback that exceeds the64-record publication cap.
The first metadata/publication/count error remains an owned io::Error; later
callbacks may update the checked count but publish no further Key records.
A counter overflow retains an error and cannot produce a successful completion.

## Bounded ownership and publication

Let B be WebViewBridge::message_byte_limit() and T be the serialized length of
Closed with the full u32 counter width, no publication error and three actual
successful owner counts of zero. Ready and every Key reserve their complete
encoded length against B-T. The actual Closed reserves against B. Every checked
addition and bound is admitted before the actual post. Reservation precedes
post_json, so a reentrant callback cannot spend the same bytes; a failed
publication does not refund attempted bytes. The bridge independently checks
the actual JSON/UTF16 and serialized native object bounds for each message.

At most64 Key notices are published. No input string is truncated. A complete
error string or unexpected nonzero cleanup metadata has no guaranteed finite
reserved size: if its complete actual encoding exceeds remaining B, publication
fails visibly. The fixture never invents a shorter successful Closed notice.
No RefCell borrow crosses bridge posting, listener registration/removal, or a
metadata getter. Error borrows only transfer/retain values between those calls.

A one-shot receives the sole actual native Close command. The future parses the
original closed DTO after the callback returns, so it cannot destroy its own
executing native listener. Native and key listeners stay strongly owned until
that task resumes. Both removal operations are attempted independently on every
post-install exit. The original operation/publication errors and both cleanup
errors remain typed until the final result is formed. The standard error-source
boundary carries the first cause while Display retains every actual failure.

Closed is posted only after an actual admitted Close and successful removal of
both callbacks. Its key_events/publication_error and bridge_retained/dom_active/
dom_retained values are actual observations after those removals. A failed
removal suppresses Closed and propagates all owned errors; provider Drop retains
any callback whose browser removal still fails. An invalid native command or
failed Ready also cleans both owners and cannot manufacture Closed.

## Verification boundary

Portable tests cover the actual shared codec, lifecycle/error values, strict
unknown/duplicate/missing-field rejection and reserve-length equality. Source
checks preserve the old webview_bridge_fixture and its fixed Echo protocol.
The Windows installed fixture must establish actual Ready, physical key event
sequences, complete named metadata, matching Closed counts and zero owner counts
under the original native operation deadline. Compilation or portable tests do
not establish native delivery, renderer shortcut behavior or Metis acceptance.

The focused original 43 portable laws remain separate from installed execution.
Source `14cb3fdf` / run 38835227528 executes all eight native cells: all exit 100,
seven at host close and global-false at the required trusted KeyS assertion.
The false control has actual Settings3 before=true/set-success/after=false/
post-navigation=false, false properties in all 12 SDK rows and NORMAL kind 0
cleanup with no host/profile errors. Each document retains Ready/eight trusted
Keys with D/P/no S/Closed/zero owners and 1,322 raw bytes. Later native assertions
and all three comparators/cross-eight are unrun; independent raw verification is
separate. Original 91 executions and six portable checks pass (105 total, 97 PASS/8 FAIL).
Closed establishes listener retirement, not renderer-input delivery or a Metis fix.
The controller-to-Rust-keydown boundary and both complete Metis journeys stay open.
No input/protocol/lifecycle, production callback decision/default/API/dependency,
quota or 30/60/120-second/20-minute bound changes follow from this result.
