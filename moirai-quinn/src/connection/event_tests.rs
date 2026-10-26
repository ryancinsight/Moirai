//! Event fairness through the actual driver, protocol engine, socket and timer.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Wake;

struct CountWake(AtomicUsize);

impl Wake for CountWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[tokio::test]
async fn connection_backlog_yields_to_transmission_and_timers() {
    let _trace =
        tracing::subscriber::set_default(tracing_subscriber::fmt().with_test_writer().finish());
    let _arrange = tracing::info_span!("arrange_oracle").entered();
    // Preserve the reviewer's full 65,536-event domain and use the actual
    // upstream receiver as the compatibility oracle for one fresh task poll.
    const BACKLOG: usize = 65_536;
    let (upstream_sender, mut upstream_receiver) = tokio::sync::mpsc::unbounded_channel();
    for event in 0..BACKLOG {
        upstream_sender.send(event).expect("upstream admission");
    }
    drop(_arrange);
    let upstream_budget = tokio::spawn(async move {
        std::future::poll_fn(|cx| {
            let mut count = 0;
            while let Poll::Ready(Some(_)) = upstream_receiver.poll_recv(cx) {
                count += 1;
            }
            Poll::Ready(count)
        })
        .await
    })
    .await
    .expect("upstream oracle task");
    assert_eq!(upstream_budget, 128);

    for backlog in [
        BACKLOG,
        0,
        1,
        upstream_budget - 1,
        upstream_budget,
        upstream_budget + 1,
    ] {
        let _case = tracing::info_span!("case", backlog, upstream_budget).entered();
        let _arrange = tracing::info_span!("arrange").entered();
        let endpoint = crate::tests::endpoint();
        let connecting = endpoint
            .connect(
                endpoint.local_addr().expect("real loopback address"),
                "localhost",
            )
            .expect("real client protocol state");
        let connection = connecting.conn.as_ref().expect("connecting state").clone();
        let (sender, receiver) = mpsc::unbounded_channel();
        let (previous_receiver, socket, transmitted_before) = {
            let mut state = connection.state.lock("arrange fairness");
            let previous_receiver = std::mem::replace(&mut state.conn_events, receiver);
            (
                previous_receiver,
                state.socket.clone(),
                state.inner.stats().udp_tx.datagrams,
            )
        };
        for _ in 0..backlog {
            sender
                .enqueue(ConnectionEvent::Rebind(socket.clone()))
                .expect("native admission");
        }
        let count = Arc::new(CountWake(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        let mut cx = Context::from_waker(&waker);
        let mut driver = ConnectionDriver::new(connection.clone());
        drop(_arrange);
        let _act = tracing::info_span!("act").entered();
        assert!(Pin::new(&mut driver).poll(&mut cx).is_pending());
        drop(_act);
        let _assert = tracing::info_span!("assert").entered();
        let (remaining, transmitted_after, timer_present, timer_deadline_matches) = {
            let mut state = connection.state.lock("inspect fairness");
            let mut remaining = 0;
            while state.conn_events.try_recv().is_some() {
                remaining += 1;
            }
            let transmitted_after = state.inner.stats().udp_tx.datagrams;
            let timer_present = state.timer.is_some();
            let protocol_deadline = state.inner.poll_timeout();
            (
                remaining,
                transmitted_after,
                timer_present,
                state.timer_deadline == protocol_deadline,
            )
        };
        let first_wakes = count.0.load(Ordering::Relaxed);

        // After continuation or an empty-queue registration, a subsequent
        // event remains observable and re-establishes normal parked wakeups.
        count.0.store(0, Ordering::Relaxed);
        sender
            .enqueue(ConnectionEvent::Rebind(socket.clone()))
            .expect("continuation admission");
        assert!(Pin::new(&mut driver).poll(&mut cx).is_pending());
        count.0.store(0, Ordering::Relaxed);
        sender
            .enqueue(ConnectionEvent::Rebind(socket))
            .expect("parked admission");
        assert_eq!(count.0.load(Ordering::Relaxed), 1);
        let parked_wakes = count.0.load(Ordering::Relaxed);
        let detached_receiver = {
            let mut state = connection.state.lock("restore receiver");
            std::mem::replace(&mut state.conn_events, previous_receiver)
        };
        drop(detached_receiver);
        assert_eq!(remaining, backlog.saturating_sub(upstream_budget));
        assert!(transmitted_after > transmitted_before);
        assert!(timer_present);
        assert!(timer_deadline_matches);
        assert_eq!(first_wakes, usize::from(backlog >= upstream_budget));
        assert_eq!(parked_wakes, 1);
    }
}
