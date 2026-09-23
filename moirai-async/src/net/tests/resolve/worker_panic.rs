//! A panic inside a resolver job fails that lookup and leaves the worker
//! serving: after more panics than there are workers, lookups still complete.

use crate::net::resolve::{RESOLVER_ADMISSIONS, RESOLVER_WORKERS, resolve, test_hooks};
use futures::executor::block_on;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::sync::{Arc, mpsc};
use std::task::{Context, Poll, Wake, Waker};

const QUERY: &str = "localhost:9";

/// Resolve on a helper thread and wait at most [`test_hooks::STAGE_LIMIT`],
/// so a pool with no live worker fails the test instead of hanging it.
fn resolve_within_limit() -> io::Result<SocketAddr> {
    let (sender, outcome) = mpsc::channel();
    std::thread::spawn(move || {
        let resolved = block_on(resolve(QUERY)).map(|addrs| addrs.first());
        sender.send(resolved).expect("the test awaits the outcome");
    });
    outcome
        .recv_timeout(test_hooks::STAGE_LIMIT)
        .expect("a lookup must finish while the pool has no live worker")
}

#[test]
fn panicking_lookups_fail_alone_and_the_pool_keeps_serving() {
    let _exclusive = test_hooks::exclusive();
    let baseline = test_hooks::progress();
    let panics = RESOLVER_WORKERS + 1;
    test_hooks::inject_panics(panics);

    for _ in 0..panics {
        let failure = resolve_within_limit().expect_err("an injected panic fails its lookup");
        assert_eq!(failure.kind(), io::ErrorKind::Other);
    }
    let resolved = resolve_within_limit().expect("the pool must still serve lookups");
    assert!(resolved.ip().is_loopback());

    let progress =
        test_hooks::wait_until(|progress| progress.disposed == baseline.disposed + panics + 1);
    assert_eq!(progress.disposed - baseline.disposed, panics + 1);
    assert_eq!(test_hooks::workers(), RESOLVER_WORKERS);
    assert_eq!(test_hooks::free_admissions(), RESOLVER_ADMISSIONS);
}

/// A waker that panics when woken, as a buggy executor's might.
struct PanickingWaker;

impl Wake for PanickingWaker {
    fn wake(self: Arc<Self>) {
        panic!("injected waker panic");
    }
}

#[test]
fn panicking_waker_does_not_take_a_worker_down() {
    let _exclusive = test_hooks::exclusive();
    let baseline = test_hooks::progress();
    let waker = Waker::from(Arc::new(PanickingWaker));
    let lookups = RESOLVER_WORKERS + 1;
    for done in 0..lookups {
        // The gate holds the worker inside the lookup until the panicking
        // waker is registered, so the worker's `send` is what wakes it.
        test_hooks::set_gate_closed(true);
        let mut lookup = Box::pin(resolve(QUERY));
        assert!(matches!(
            lookup.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        test_hooks::wait_until(|progress| progress.started == baseline.started + done + 1);
        test_hooks::set_gate_closed(false);
        test_hooks::wait_until(|progress| progress.disposed == baseline.disposed + done + 1);
        let resolved = block_on(lookup).expect("the reply is stored before the wake");
        assert!(resolved.first().ip().is_loopback());
    }
    let resolved = resolve_within_limit().expect("the pool must still serve lookups");
    assert!(resolved.ip().is_loopback());
}
