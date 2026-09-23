//! `TcpStream::connect` must never block the polling thread: an outer timeout
//! or a dropped future bounds the wait, and the executor keeps running.

use crate::executor::AsyncExecutor;
use crate::io::{AsyncReadExt, AsyncWriteExt};
use crate::net::TcpStream;
use crate::timer::timeout;
use std::io::{self, Read, Write};
use std::net::TcpListener as StdTcpListener;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// RFC 1918 address with no host behind it on test networks: SYNs go
/// unanswered, so a blocking connect waits for the OS connect timeout
/// (about 21 s on Windows, 127 s on Linux with default SYN retries).
const UNANSWERED_ADDR: &str = "10.255.255.1:9";

/// Outer deadline placed on the unanswered connect.
const CONNECT_DEADLINE: Duration = Duration::from_millis(200);

/// Allowance above the deadline for timer-driver wake-up and executor
/// scheduling on a loaded CI runner (measured to run about 15x slower than a
/// developer host). One second is 20x below the smallest OS connect timeout
/// above, so a blocking connect cannot pass inside it.
const SCHEDULING_MARGIN: Duration = Duration::from_secs(1);

/// Run `executor` on its own thread until `stop` is called.
fn start_executor() -> (Arc<AsyncExecutor>, std::thread::JoinHandle<io::Result<()>>) {
    let executor = Arc::new(AsyncExecutor::new().expect("async executor must start"));
    let runner_executor = Arc::clone(&executor);
    let runner = std::thread::spawn(move || runner_executor.run());
    (executor, runner)
}

fn stop_executor(executor: &AsyncExecutor, runner: std::thread::JoinHandle<io::Result<()>>) {
    executor.stop().expect("executor stop must wake reactor");
    runner
        .join()
        .expect("executor thread must not panic")
        .expect("executor run must stop cleanly");
}

#[test]
fn unanswered_connect_returns_within_outer_timeout() {
    let (executor, runner) = start_executor();

    let handle = executor.spawn(async {
        let started = Instant::now();
        let outcome = timeout(CONNECT_DEADLINE, TcpStream::connect(UNANSWERED_ADDR)).await;
        (
            outcome.map(|connected| connected.map(drop)),
            started.elapsed(),
        )
    });
    let (outcome, elapsed) = futures::executor::block_on(handle);

    // A host without a route to 10/8 fails the connect synchronously
    // (`NetworkUnreachable`/`HostUnreachable`); either way the call must not
    // wait on the handshake past the outer deadline.
    match outcome {
        Err(_timeout) => assert!(
            elapsed >= CONNECT_DEADLINE,
            "timeout fired early after {elapsed:?}"
        ),
        Ok(Err(error)) => assert_ne!(
            error.kind(),
            io::ErrorKind::TimedOut,
            "an OS connect timeout means the connect blocked"
        ),
        Ok(Ok(())) => panic!("{UNANSWERED_ADDR} unexpectedly accepted a connection"),
    }
    assert!(
        elapsed < CONNECT_DEADLINE + SCHEDULING_MARGIN,
        "connect held the executor for {elapsed:?}, past {CONNECT_DEADLINE:?} + {SCHEDULING_MARGIN:?}"
    );

    stop_executor(&executor, runner);
}

#[test]
fn dropped_connect_leaves_executor_responsive() {
    let (executor, runner) = start_executor();

    // `timeout` drops the in-flight connect future at its deadline. The task
    // spawned behind it shares the executor thread, so it runs promptly only
    // if polling the connect returns instead of waiting on the handshake.
    let cancelled = executor.spawn(async {
        timeout(CONNECT_DEADLINE, TcpStream::connect(UNANSWERED_ADDR))
            .await
            .map(|connected| connected.map(drop))
    });
    let spawned = Instant::now();
    let concurrent = executor.spawn(async { 6_u32 * 7 });
    assert_eq!(futures::executor::block_on(concurrent), 42);
    let concurrent_latency = spawned.elapsed();

    let cancelled = futures::executor::block_on(cancelled);
    assert!(
        !matches!(cancelled, Ok(Ok(()))),
        "{UNANSWERED_ADDR} unexpectedly accepted a connection"
    );

    let respawned = Instant::now();
    let follow_up = executor.spawn(async { 6_u32 * 7 });
    assert_eq!(futures::executor::block_on(follow_up), 42);
    let follow_up_latency = respawned.elapsed();

    assert!(
        concurrent_latency < SCHEDULING_MARGIN,
        "a task queued behind the connect waited {concurrent_latency:?}"
    );
    assert!(
        follow_up_latency < SCHEDULING_MARGIN,
        "executor took {follow_up_latency:?} to run a task after the connect was dropped"
    );

    stop_executor(&executor, runner);
}

#[test]
fn connect_to_listening_port_exchanges_bytes() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("listener bind must succeed");
    let addr = listener.local_addr().expect("listener address must exist");
    let server = std::thread::spawn(move || {
        let (mut accepted, _) = listener.accept().expect("server accept must succeed");
        let mut request = [0_u8; 4];
        accepted
            .read_exact(&mut request)
            .expect("server read must succeed");
        accepted
            .write_all(b"pong")
            .expect("server write must succeed");
        request
    });

    let (executor, runner) = start_executor();
    let handle = executor.spawn(async move {
        let mut stream = TcpStream::connect(&addr.to_string())
            .await
            .expect("connect to a listening port must succeed");
        let peer = stream.peer_addr().expect("connected stream has a peer");
        stream
            .write_all(b"ping")
            .await
            .expect("client write must succeed");
        let mut reply = [0_u8; 4];
        stream
            .read_exact(&mut reply)
            .await
            .expect("client read must succeed");
        (peer, reply)
    });
    let (peer, reply) = futures::executor::block_on(handle);

    assert_eq!(peer, addr);
    assert_eq!(&reply, b"pong");
    assert_eq!(&server.join().expect("server thread must join"), b"ping");
    stop_executor(&executor, runner);
}

#[test]
fn connect_to_closed_port_reports_refusal() {
    let addr = {
        let listener = StdTcpListener::bind("127.0.0.1:0").expect("listener bind must succeed");
        listener.local_addr().expect("listener address must exist")
    };

    let (executor, runner) = start_executor();
    let handle =
        executor.spawn(async move { TcpStream::connect(&addr.to_string()).await.map(drop) });
    let error = futures::executor::block_on(handle).expect_err("closed port must refuse");

    assert_eq!(error.kind(), io::ErrorKind::ConnectionRefused);
    stop_executor(&executor, runner);
}

#[test]
fn hostname_connect_resolves_off_thread_and_connects() {
    let _exclusive = crate::net::resolve::test_hooks::exclusive();
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("listener bind must succeed");
    let port = listener
        .local_addr()
        .expect("listener address must exist")
        .port();
    let server =
        std::thread::spawn(move || listener.accept().expect("server accept must succeed").1);

    let (executor, runner) = start_executor();
    // `localhost` may resolve to `::1` first; the fallback walk must reach the
    // IPv4 listener.
    let handle = executor.spawn(async move {
        let stream = TcpStream::connect(&format!("localhost:{port}"))
            .await
            .expect("connect by hostname must succeed");
        stream
            .local_addr()
            .expect("connected stream has a local address")
    });
    let client_addr = futures::executor::block_on(handle);

    assert_eq!(server.join().expect("server thread must join"), client_addr);
    stop_executor(&executor, runner);
}
