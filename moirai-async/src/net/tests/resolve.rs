//! The hostname resolver runs on a fixed worker pool: more concurrent lookups
//! than workers queue and wait asynchronously, and never add threads.

use crate::executor::AsyncExecutor;
use crate::net::TcpStream;
use crate::net::resolve::{RESOLVER_QUEUE_DEPTH, RESOLVER_WORKERS, test_hooks};

mod cancellation;
use std::future::{Future, poll_fn};
use std::net::TcpListener as StdTcpListener;
use std::pin::pin;
use std::sync::{Arc, mpsc};
use std::task::Poll;

/// Three lookups per worker: enough to fill every worker and the queue and
/// leave callers waiting on admission.
const CONCURRENT_LOOKUPS: usize = 3 * RESOLVER_WORKERS;

#[test]
fn concurrent_hostname_connects_never_exceed_the_worker_bound() {
    let _exclusive = test_hooks::exclusive();
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("listener bind must succeed");
    let port = listener
        .local_addr()
        .expect("listener address must exist")
        .port();
    // `localhost` may try `::1` first, which the IPv4-only listener refuses, so
    // accept exactly the connections that land here.
    let server = std::thread::spawn(move || {
        (0..CONCURRENT_LOOKUPS)
            .map(|_| listener.accept().expect("server accept must succeed").1)
            .collect::<Vec<_>>()
    });

    let executor = Arc::new(AsyncExecutor::new().expect("async executor must start"));
    let runner_executor = Arc::clone(&executor);
    let runner = std::thread::spawn(move || runner_executor.run());

    test_hooks::set_gate_closed(true);
    let (submitted, submissions) = mpsc::channel();
    let handles: Vec<_> = (0..CONCURRENT_LOOKUPS)
        .map(|_| {
            let submitted = submitted.clone();
            executor.spawn(async move {
                let query = format!("localhost:{port}");
                let mut connect = pin!(TcpStream::connect(&query));
                // The first poll either takes an admission permit and queues
                // the lookup, or parks on admission; report it only after that.
                let first = poll_fn(|cx| Poll::Ready(connect.as_mut().poll(cx))).await;
                submitted
                    .send(())
                    .expect("test thread holds the submission receiver");
                let connected = match first {
                    Poll::Ready(connected) => connected,
                    Poll::Pending => connect.await,
                };
                connected.map(|stream| stream.local_addr())
            })
        })
        .collect();

    for _ in 0..CONCURRENT_LOOKUPS {
        submissions
            .recv_timeout(test_hooks::STAGE_LIMIT)
            .expect("every connect must reach resolver admission");
    }
    let running = test_hooks::wait_until(|progress| progress.live == RESOLVER_WORKERS).live;
    let free_admissions = test_hooks::free_admissions();
    test_hooks::set_gate_closed(false);

    let connected = handles
        .into_iter()
        .map(|handle| {
            futures::executor::block_on(handle)
                .expect("hostname connect must succeed")
                .expect("connected stream has a local address")
        })
        .collect::<Vec<_>>();

    assert_eq!(running, RESOLVER_WORKERS, "every worker must hold a lookup");
    assert_eq!(
        free_admissions, 0,
        "{CONCURRENT_LOOKUPS} lookups must exhaust {RESOLVER_WORKERS} running + {RESOLVER_QUEUE_DEPTH} queued admissions"
    );
    assert_eq!(test_hooks::workers(), RESOLVER_WORKERS);
    assert!(
        test_hooks::peak() <= RESOLVER_WORKERS,
        "{} lookups ran at once, past the {RESOLVER_WORKERS}-worker bound",
        test_hooks::peak()
    );
    let mut accepted = server.join().expect("server thread must join");
    let mut connected = connected;
    accepted.sort_unstable();
    connected.sort_unstable();
    assert_eq!(connected.len(), CONCURRENT_LOOKUPS);
    assert_eq!(accepted, connected, "every client must reach this listener");

    executor.stop().expect("executor stop must wake reactor");
    runner
        .join()
        .expect("executor thread must not panic")
        .expect("executor run must stop cleanly");
}
