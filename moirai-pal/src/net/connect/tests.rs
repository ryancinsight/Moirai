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
    use std::future::Future;
    use std::sync::{Arc, Mutex, PoisonError, mpsc};
    use std::task::{Context, Wake, Waker};

    /// Allowance above the interval for the re-probe thread to be scheduled
    /// on a loaded CI runner.
    const SCHEDULING_MARGIN: Duration = Duration::from_secs(1);

    struct Signal(Mutex<mpsc::Sender<()>>);

    impl Wake for Signal {
        fn wake(self: Arc<Self>) {
            // A tick that cloned the waker before the registration dropped
            // may wake after the test released its receiver; that wake is
            // unobserved by design.
            let _unobserved = self
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .send(())
                .is_err();
        }
    }

    /// Serializes tests that count registrations; under `cargo test` they
    /// share the process-wide registry.
    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        static EXCLUSIVE: Mutex<()> = Mutex::new(());
        EXCLUSIVE.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[test]
    fn armed_registration_is_woken_until_dropped() {
        let _exclusive = exclusive();
        let (sender, woken) = mpsc::channel();
        let waker = Waker::from(Arc::new(Signal(Mutex::new(sender))));
        let mut registration = reprobe::Registration::new();
        registration
            .arm(&waker)
            .expect("re-probe thread must start");
        registration.arm(&waker).expect("re-arming keeps one entry");
        assert_eq!(reprobe::registered(), 1);
        for _ in 0..2 {
            woken
                .recv_timeout(reprobe::CONNECT_REPROBE_INTERVAL + SCHEDULING_MARGIN)
                .expect("re-probe must wake the registration every interval");
        }
        drop(registration);
        assert_eq!(
            reprobe::registered(),
            0,
            "a dropped registration is removed"
        );
    }

    #[test]
    fn refused_connect_completes_when_the_reactor_never_reports_it() {
        let _exclusive = exclusive();
        // An undriven reactor stands in for pre-2004 `WSAPoll`: the connect's
        // writable registration never fires, so only the re-probe can wake it.
        let silent = IoReactor::new().expect("reactor must build");
        let addr = closed_port();
        let refused = silent
            .with_active(|| block_on(AsyncTcpStream::connect(addr)))
            .map(drop)
            .expect_err("a closed port must refuse");
        assert_eq!(refused.kind(), io::ErrorKind::ConnectionRefused);
        assert_eq!(reprobe::registered(), 0, "a settled connect deregisters");
    }

    #[test]
    fn dropped_pending_connect_deregisters() {
        let _exclusive = exclusive();
        let silent = IoReactor::new().expect("reactor must build");
        let noop = futures::task::noop_waker();
        let mut context = Context::from_waker(&noop);
        let mut connect = Box::pin(AsyncTcpStream::connect(
            "10.255.255.1:9".parse().expect("literal address"),
        ));
        // A host without a route to 10/8 fails synchronously and never
        // registers; only a pending connect exercises the drop.
        if silent
            .with_active(|| connect.as_mut().poll(&mut context))
            .is_pending()
        {
            assert_eq!(reprobe::registered(), 1);
        }
        drop(connect);
        assert_eq!(reprobe::registered(), 0);
    }

    /// A waker whose drop runs another registration's drop, which takes the
    /// registry lock.
    struct HoldsRegistration(
        #[expect(dead_code, reason = "held only for its drop")] reprobe::Registration,
    );

    #[expect(
        clippy::manual_noop_waker,
        reason = "the waker exists to own a registration; its drop is under test"
    )]
    impl Wake for HoldsRegistration {
        fn wake(self: Arc<Self>) {}
    }

    /// Run `release` on a helper thread; a deadlock fails the test instead of
    /// hanging it.
    fn completes(release: impl FnOnce() + Send + 'static) {
        let (sender, done) = mpsc::channel();
        std::thread::spawn(move || {
            release();
            sender.send(()).expect("the test awaits completion");
        });
        done.recv_timeout(SCHEDULING_MARGIN)
            .expect("releasing a registration must not deadlock on the registry lock");
    }

    fn registration_holding_waker() -> Waker {
        let mut inner = reprobe::Registration::new();
        inner
            .arm(&futures::task::noop_waker())
            .expect("re-probe thread must start");
        Waker::from(Arc::new(HoldsRegistration(inner)))
    }

    #[test]
    fn removed_waker_drops_outside_the_registry_lock() {
        let _exclusive = exclusive();
        let mut outer = reprobe::Registration::new();
        outer
            .arm(&registration_holding_waker())
            .expect("re-probe thread must start");
        assert_eq!(reprobe::registered(), 2);
        completes(move || drop(outer));
        assert_eq!(reprobe::registered(), 0);
    }

    #[test]
    fn replaced_waker_drops_outside_the_registry_lock() {
        let _exclusive = exclusive();
        let mut outer = reprobe::Registration::new();
        outer
            .arm(&registration_holding_waker())
            .expect("re-probe thread must start");
        completes(move || {
            outer
                .arm(&futures::task::noop_waker())
                .expect("re-probe thread is running");
            assert_eq!(reprobe::registered(), 1);
        });
    }

    /// A waker that panics when woken, as a buggy executor's might.
    struct PanicsOnWake;

    impl Wake for PanicsOnWake {
        fn wake(self: Arc<Self>) {
            panic!("injected re-probe wake panic");
        }
    }

    /// Makes [`panics_on_clone`] wakers panic when cloned.
    static CLONE_PANICS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    static PANICS_ON_CLONE: std::task::RawWakerVTable =
        std::task::RawWakerVTable::new(clone_or_panic, noop_raw, noop_raw, noop_raw);

    fn clone_or_panic(data: *const ()) -> std::task::RawWaker {
        assert!(
            !CLONE_PANICS.load(std::sync::atomic::Ordering::SeqCst),
            "injected re-probe clone panic"
        );
        std::task::RawWaker::new(data, &PANICS_ON_CLONE)
    }

    fn noop_raw(_: *const ()) {}

    /// A waker whose clone panics once [`CLONE_PANICS`] is set.
    fn panics_on_clone() -> Waker {
        // SAFETY: every vtable function ignores the data pointer, which is
        // never dereferenced, so a null pointer satisfies the `RawWaker`
        // contract; clone returns a waker with the same vtable.
        unsafe { Waker::from_raw(std::task::RawWaker::new(std::ptr::null(), &PANICS_ON_CLONE)) }
    }

    /// Arm `faulty` beside a signalling registration and require the
    /// signalling one to be woken on each of two full ticks after both are
    /// registered, which the tick count shows the thread survived.
    fn faulty_registration_leaves_reprobe_running(faulty: &Waker, arm_fault: impl FnOnce()) {
        let (sender, woken) = mpsc::channel();
        let signal = Waker::from(Arc::new(Signal(Mutex::new(sender))));
        let mut faulty_registration = reprobe::Registration::new();
        faulty_registration
            .arm(faulty)
            .expect("re-probe thread must start");
        let mut signal_registration = reprobe::Registration::new();
        signal_registration
            .arm(&signal)
            .expect("re-probe thread must start");
        arm_fault();

        // A tick already in flight may have cloned the registry before both
        // entries were present, so require three completions past this one.
        let start = reprobe::ticks::count();
        let limit = 3 * reprobe::CONNECT_REPROBE_INTERVAL + SCHEDULING_MARGIN;
        let reached = reprobe::ticks::wait_for(start + 3, limit);
        drop(faulty_registration);
        drop(signal_registration);

        assert!(
            reached >= start + 3,
            "the re-probe thread stopped ticking after {} of 3 ticks",
            reached - start
        );
        assert!(
            woken.try_iter().count() >= 2,
            "the healthy registration must be woken on every tick"
        );
    }

    #[test]
    fn panicking_wake_leaves_other_registrations_reprobed() {
        let _exclusive = exclusive();
        faulty_registration_leaves_reprobe_running(&Waker::from(Arc::new(PanicsOnWake)), || {});
    }

    #[test]
    fn panicking_clone_leaves_other_registrations_reprobed() {
        let _exclusive = exclusive();
        faulty_registration_leaves_reprobe_running(&panics_on_clone(), || {
            CLONE_PANICS.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        CLONE_PANICS.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}
