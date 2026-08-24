# The message router: topic fan-out over transports

Part IV so far moved single messages to single addresses. Real workloads
also need one message delivered to *every* endpoint interested in a
subject. `router.rs` provides that — and, like the removed
`UniversalChannel` of Chapter 3, it carries a preserved failure that
teaches more than its success does.

## One transport instance, or none

```rust,ignore
pub struct MessageRouter<T: Transport> {
    transport: Arc<T>,
    subscriptions: Mutex<HashMap<String, Vec<Address>>>,
}
```

The router is generic over `T: Transport`, so delivery monomorphizes per
backend with no dynamic dispatch. Its doc comment records why the shared
`Arc` is load-bearing: a prior implementation constructed a throwaway
`InMemoryTransport` per send, and since in-memory channels are keyed by
address *within one instance*, every message was silently discarded.
The bug class is worth naming: a router's correctness depends on
**identity**, not just type, of its dependencies — exactly the property
the Chapter 2 region system encodes for buffers.

## Subscription discipline

`subscribe` ignores duplicate `(topic, address)` pairs; `unsubscribe`
returns whether anything was removed and deletes emptied topics so the
map never accumulates tombstones. Together they uphold the invariant the
first test asserts: **each subscriber receives each published message
exactly once**, regardless of how many times callers re-subscribe.

## Publishing: lock scope and buffer ownership

`publish` makes two decisions worth copying:

- **Snapshot under lock, deliver outside it.** The subscriber list is
  cloned while the subscriptions mutex is held, then all transport sends
  happen lock-free — a slow subscriber's send can never block a
  concurrent resubscribe.
- **N subscribers cost N−1 clones.** `Transport::send` takes ownership,
  so fan-out to N targets needs N owned buffers — but the caller's
  original buffer moves to the *final* subscriber instead of being cloned
  and dropped, and `checked_sub`/`Option::take` bookkeeping enforces that
  the move happens exactly once (`debug_assert`s carry the invariant at
  the take sites).

## Failure semantics: fail-fast with visible partiality

Delivery is fail-fast: the first transport error propagates immediately.
Subscribers *before* the failing one have already received the message;
subscribers after it have not; and because the error return replaces the
delivered count, a caller that retries must treat the publish as
at-least-once for the earlier targets. That asymmetry is stated in the
doc comment rather than hidden — Part IV's standing rule that failure
semantics are part of the contract, not an implementation detail.
Callers wanting best-effort delivery wrap per-subscriber sends
themselves; the router refuses to silently swallow errors on their
behalf (there is no "skip failures" mode, by design).

## Where retry and backpressure actually live

Nothing in this module retries or bounds queues — deliberately. Those
policies sit where their state lives:

- **Backpressure** is each channel's bounded capacity (Chapter 1) plus
  the network frame cap: a full path returns `Full`/`Closed` rather than
  growing without limit.
- **Retry policy** belongs to route selection — the scheduler's
  `HybridRouter::select::<C>(priority, sequence)` (moirai-executor)
  decides *where* work goes, and the routed clients of Chapter 4 execute
  there. A pub/sub fan-out has no meaningful single retry target: which
  subscriber would you retry?

Keeping those concerns out of `MessageRouter` is what keeps it four
methods and exhaustively testable.

## Worked example

From the module tests — duplicate subscription ignored, both endpoints
receive, unknown topics are a zero-delivery success:

```rust,ignore
let transport = Arc::new(InMemoryTransport::new());
let router = MessageRouter::new(Arc::clone(&transport));
let sub_a = Address::Local("sub_a".into());

router.subscribe("topic", sub_a.clone());
router.subscribe("topic", sub_b.clone());
router.subscribe("topic", sub_a.clone()); // duplicate ignored

assert_eq!(router.publish("topic", vec![1, 2, 3])?, 2);
assert_eq!(transport.recv(&sub_a)?, vec![1, 2, 3]);
```

## Part IV remaining

One chapter remains: remote tasks bind capability negotiation and server
lifecycle onto everything built so far (`remote_task/`).
