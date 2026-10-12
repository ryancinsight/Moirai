# 0072 — Interpret browser bodyless responses before byte admission

Status: Accepted

Date: 2026-10-12

## Contract and decision

The Fetch specification defines bodyless statuses 101, 103, 204, 205 and 304.
Their received body length is zero. RFC 9110 §8.6 permits a 304 Content-Length
to describe the selected representation; it does not declare transferred body
bytes. These responses preserve their actual status and return an empty byte
vector before payload-length admission. An absent Body stream also denotes
an empty body. A present payload stream retains declared and counted bounds,
whole-request deadlines, reader ownership and cancellation.

This corrects a real browser regression: a 304 with representation length
16,385 and no wire body rejects under the client's 16,384-byte received-body
budget. The new path admits zero received bytes; the same producer's 200 with
16,385 declared or streamed payload bytes still rejects. No budget changes.

The pure policy checks every unsigned wire status and obtains exactly the five
specified bodyless statuses. The independent live producer checks 204, 205 and
304, exact empty/Unicode/pattern bytes, excess bodies, redirect refusal,
deadlines and peer-observed cancellation. These observations remain distinct
from cross-engine and physical-device evidence.

Sources: [Fetch status contract](https://fetch.spec.whatwg.org/#null-body-status)
and [HTTP Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#section-8.6).
