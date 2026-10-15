# One native parent entry after UIA field selection

The installed `2fcef45d` baseline admits all failed scope-5 input records while
native focus remains at a different owned browser HWND than earlier successful
controller key callbacks. UIA reports the selected reference field focused, but
both UIA and a timely real DOMSnapshot retain its previous Unicode value. The
complete independent controller sequence ends in LostFocus with no subsequent
key callback. Ordinary mode also fails; continuous diagnostic traffic is not a
necessary cause. These observations establish an entry-contract gap, not a
complete account of the runtime's internal focus transitions.

UIA field selection and native controller entry have separate responsibilities.
Keep the existing one UIA SetFocus and its original focus observation. Next make
one real SetFocus request to the known native parent. Its existing WM_SETFOCUS
owner synchronously calls MoveFocus(PROGRAMMATIC). Microsoft specifies that this
reason restores the previously focused page element. No HWND class search,
synthetic focus message, ALT tap, focus repair callback, retry, sleep, or new
deadline participates.

The child MTA creates its Win32 message queue with a non-removing PeekMessage.
When it differs from the parent HWND creator thread, it temporarily attaches its
input queue to that actual thread. While attached, it verifies the foreground
belongs to the host, invokes SetFocus(parent) once, then requires
GetFocus to identify a non-root owned descendant. It always attempts detach;
native operation and detach failures retain their original errors. The queue is
detached before the same UIA element's CurrentHasKeyboardFocus is required true
once. UIA SetFocus is never reissued to conceal loss of field selection.

For every replacement, the entry must finish within the original deadline before
either original input batch. An admitted replacement then retains the original
exact value and four-field preservation checks. Ordinal 2 is a concrete
non-first-field oracle: weight becomes 64 while reference remains 患者-🦀-one.
The private child completion verifies 16 successful entries, including 12 to
non-first fields, alongside the original 16 replacements and 64 preservation
checks. These counters advance only after native entry and same-field focus
validation. All eight modes and the existing composition checks remain intact.

The private native ABI call preserves SetFocus's previous HWND independently
of its immediate error. The generated Result wrapper rejects a null previous
handle, including a successful transition from no prior native focus. Clear
the native error immediately before this one call, capture it immediately
afterward, reject a failed null result, and always validate actual delegated
focus. Native entry geometry is recorded before the post-entry UIA query so
that a failed field-preservation check cannot conceal the observed handoff.

Cross-compilation establishes Rust/Win32 type consistency only. The unchanged
actual native matrix must execute on Windows before this correction is accepted;
the complete Metis form is a separate consumer gate. The original baseline
receipts remain authoritative and are not overwritten by candidate results.

Primary contracts:

- [SetFocus and thread queue attachment](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setfocus)
- [AttachThreadInput lifetime and message queues](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-attachthreadinput)
- [MoveFocus(PROGRAMMATIC) preserves the prior page element](https://learn.microsoft.com/en-us/microsoft-edge/webview2/reference/win32/icorewebview2controller#movefocus)
