# ADR 0031: Timer Wheel Lazy Cancellation

Status: Accepted

Revised 2026-11-16 in [PR 705](https://github.com/ryancinsight/Moirai/pull/705) to release cancelled wakers and bound retained heap entries.

### Context

A cancelled timer retained its waker and both membership entries until it
reached the heap head. An earlier live deadline allowed repeated cancellation
of later timers to accumulate entries and retain their wakers.

### Decision

Store live wakers by timer ID and keep deadline and ID in the binary heap.
Cancellation removes and drops the waker immediately. Discard stale entries
when they reach the head, and rebuild the heap when stale entries outnumber
live entries. Reclaim stale deadlines before invoking a waker so the bound
holds if user code unwinds. Preserve deadline order and use timer ID to break ties.

### Rationale

Every cancellation adds at most one stale entry. A rebuild scans the heap
in linear time when at least half its entries are stale, so its work can
be charged to the cancellations that produced those entries. Heap length
after cancellation or expiry polling is at most twice the live count.
An indexed heap could remove stale deadlines eagerly, at the cost of
maintaining a position index on every heap move. Reconsider eager deletion
if a measured rebuild violates a caller's latency budget.

### Verification

The `timer_wheel_cancel_releases_wakers_behind_a_live_deadline` regression
checks ownership and heap length while a live head prevents lazy draining.
The expiry tests check wake counts, duplicate or expired cancellation, and
the heap bound after a caller catches a waker panic.
