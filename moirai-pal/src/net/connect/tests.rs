use super::*;
use std::net::TcpListener as StdTcpListener;

/// Upper wait for a probe that blocks until the connect settles. A refused
/// loopback connect settles after Windows' SYN retries, 2.06 to 2.48 s
/// measured here; the bound only catches a hang.
const SETTLE_LIMIT: Duration = Duration::from_secs(10);

/// A loopback port with nothing listening on it.
fn closed_port() -> SocketAddr {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("listener bind must succeed");
    listener.local_addr().expect("listener address must exist")
}

#[test]
fn probe_reports_refusal_without_reactor_readiness() {
    // The probe alone (`select` on Windows, `poll` on Unix) decides the
    // outcome; no reactor or `WSAPoll` is involved.
    let refused = start_connect(closed_port())
        .and_then(|stream| connect_outcome(&stream, SETTLE_LIMIT))
        .expect_err("a closed port must refuse");
    assert_eq!(refused.kind(), io::ErrorKind::ConnectionRefused);
}

#[test]
fn probe_reports_completed_connect() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("listener bind must succeed");
    let addr = listener.local_addr().expect("listener address must exist");
    let stream = start_connect(addr).expect("connect must start");
    connect_outcome(&stream, SETTLE_LIMIT).expect("connect to a listener must complete");
    let (_, client) = listener.accept().expect("listener must accept");
    assert_eq!(stream.peer_addr().expect("connected peer"), addr);
    assert_eq!(stream.local_addr().expect("local address"), client);
}

#[test]
fn zero_wait_probe_reports_pending_handshake_as_would_block() {
    // RFC 1918 address with no host behind it: the SYN goes unanswered. A host
    // with no route to 10/8 fails synchronously instead; neither may report
    // success, and an unanswered handshake must read as in progress.
    match start_connect("10.255.255.1:9".parse().expect("literal address")) {
        Ok(stream) => {
            let pending = connect_outcome(&stream, Duration::ZERO)
                .expect_err("an unanswered handshake cannot have completed");
            assert_eq!(pending.kind(), io::ErrorKind::WouldBlock);
        }
        Err(error) => assert_ne!(error.kind(), io::ErrorKind::WouldBlock),
    }
}

#[cfg(windows)]
mod windows_reprobe {
    use super::*;
    use crate::reactor::IoReactor;
    use futures::executor::block_on;
    use std::sync::{Arc, Mutex, PoisonError, mpsc};
    use std::task::{Wake, Waker};

    /// Allowance above the interval for the re-probe thread to be scheduled
    /// on a loaded CI runner.
    const SCHEDULING_MARGIN: Duration = Duration::from_secs(1);

    struct Signal(Mutex<mpsc::Sender<()>>);

    impl Wake for Signal {
        fn wake(self: Arc<Self>) {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .send(())
                .expect("the test holds the receiver until the wake arrives");
        }
    }

    #[test]
    fn scheduled_waker_is_woken_within_the_interval() {
        let (sender, woken) = mpsc::channel();
        let waker = Waker::from(Arc::new(Signal(Mutex::new(sender))));
        reprobe::schedule(&waker).expect("re-probe thread must start");
        woken
            .recv_timeout(reprobe::CONNECT_REPROBE_INTERVAL + SCHEDULING_MARGIN)
            .expect("re-probe must wake the scheduled waker");
    }

    #[test]
    fn refused_connect_completes_when_the_reactor_never_reports_it() {
        // An undriven reactor stands in for pre-2004 `WSAPoll`: the connect's
        // writable registration never fires, so only the re-probe can wake it.
        let silent = IoReactor::new().expect("reactor must build");
        let addr = closed_port();
        let refused = silent
            .with_active(|| block_on(AsyncTcpStream::connect(addr)))
            .map(drop)
            .expect_err("a closed port must refuse");
        assert_eq!(refused.kind(), io::ErrorKind::ConnectionRefused);
    }
}
