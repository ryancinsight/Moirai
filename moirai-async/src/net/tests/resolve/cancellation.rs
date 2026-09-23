//! A dropped resolve future releases its admission permit at every stage:
//! while waiting for admission, while queued, and while `getaddrinfo` runs.
//! The gate hook pins each stage, so no test depends on timing.

use crate::net::resolve::{
    RESOLVER_ADMISSIONS, RESOLVER_WORKERS, ResolvedAddrs, resolve, test_hooks,
};
use futures::executor::block_on;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A name only the system resolver can answer; the port is never dialled.
const QUERY: &str = "localhost:9";

type Lookup = Pin<Box<dyn Future<Output = io::Result<ResolvedAddrs>>>>;

fn lookup() -> Lookup {
    Box::pin(resolve(QUERY))
}

/// Poll once and require `Pending`: the lookup has taken its first step
/// (admitted and queued, or parked on admission) and not completed.
fn start(lookup: &mut Lookup) {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(
        matches!(lookup.as_mut().poll(&mut context), Poll::Pending),
        "a gated lookup cannot complete"
    );
}

fn finish(lookup: Lookup) {
    let resolved = block_on(lookup).expect("localhost must resolve");
    assert!(resolved.first().ip().is_loopback());
}

/// Fill every worker with a gated lookup and wait until all of them run.
fn occupy_workers(baseline: test_hooks::Progress) -> Vec<Lookup> {
    let mut running: Vec<Lookup> = (0..RESOLVER_WORKERS).map(|_| lookup()).collect();
    running.iter_mut().for_each(start);
    let progress =
        test_hooks::wait_until(|progress| progress.started == baseline.started + RESOLVER_WORKERS);
    assert_eq!(progress.live, RESOLVER_WORKERS);
    running
}

#[test]
fn dropped_admission_waiter_returns_no_permit_and_blocks_no_successor() {
    let _exclusive = test_hooks::exclusive();
    let baseline = test_hooks::progress();
    test_hooks::set_gate_closed(true);

    let mut admitted = occupy_workers(baseline);
    admitted.extend((RESOLVER_WORKERS..RESOLVER_ADMISSIONS).map(|_| lookup()));
    admitted[RESOLVER_WORKERS..].iter_mut().for_each(start);
    assert_eq!(test_hooks::free_admissions(), 0);

    let mut abandoned = lookup();
    start(&mut abandoned);
    let mut successor = lookup();
    start(&mut successor);
    drop(abandoned);

    test_hooks::set_gate_closed(false);
    admitted.into_iter().for_each(finish);
    finish(successor);

    let settled = RESOLVER_ADMISSIONS + 1;
    let progress =
        test_hooks::wait_until(|progress| progress.disposed == baseline.disposed + settled);
    assert_eq!(progress.disposed - baseline.disposed, settled);
    assert_eq!(
        progress.started - baseline.started,
        settled,
        "the abandoned waiter was never admitted, so it never reached getaddrinfo"
    );
    assert_eq!(
        test_hooks::free_admissions(),
        RESOLVER_ADMISSIONS,
        "a permit granted to the dropped waiter was lost"
    );
}

#[test]
fn dropped_queued_lookup_is_skipped_and_releases_its_permit() {
    let _exclusive = test_hooks::exclusive();
    let baseline = test_hooks::progress();
    test_hooks::set_gate_closed(true);

    let running = occupy_workers(baseline);
    let mut abandoned = lookup();
    start(&mut abandoned);
    let mut kept = lookup();
    start(&mut kept);
    assert_eq!(
        test_hooks::free_admissions(),
        RESOLVER_ADMISSIONS - RESOLVER_WORKERS - 2
    );
    drop(abandoned);

    test_hooks::set_gate_closed(false);
    running.into_iter().for_each(finish);
    finish(kept);

    let settled = RESOLVER_WORKERS + 2;
    let progress =
        test_hooks::wait_until(|progress| progress.disposed == baseline.disposed + settled);
    assert_eq!(progress.disposed - baseline.disposed, settled);
    assert_eq!(
        progress.started - baseline.started,
        settled - 1,
        "the dropped queued lookup must not call getaddrinfo"
    );
    assert_eq!(test_hooks::free_admissions(), RESOLVER_ADMISSIONS);
}

#[test]
fn dropped_running_lookup_releases_its_permit_when_getaddrinfo_returns() {
    let _exclusive = test_hooks::exclusive();
    let baseline = test_hooks::progress();
    test_hooks::set_gate_closed(true);

    let mut abandoned = lookup();
    start(&mut abandoned);
    test_hooks::wait_until(|progress| progress.started == baseline.started + 1);
    drop(abandoned);
    assert_eq!(
        test_hooks::free_admissions(),
        RESOLVER_ADMISSIONS - 1,
        "a running lookup holds its permit until getaddrinfo returns"
    );

    test_hooks::set_gate_closed(false);
    let progress = test_hooks::wait_until(|progress| progress.disposed == baseline.disposed + 1);
    assert_eq!(progress.disposed - baseline.disposed, 1);
    assert_eq!(progress.started - baseline.started, 1);
    assert_eq!(test_hooks::free_admissions(), RESOLVER_ADMISSIONS);
}
