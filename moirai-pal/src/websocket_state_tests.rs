use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn state() -> WebSocketState {
    WebSocketState::new(WebSocketLimits::new(4, 2).expect("test limits must be valid"))
}

#[test]
fn limits_reject_zero_bounds() {
    assert_eq!(
        WebSocketLimits::new(0, 1)
            .expect_err("zero message bound must fail")
            .kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        WebSocketLimits::new(1, 0)
            .expect_err("zero queue bound must fail")
            .kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn messages_preserve_order_and_close_drains_queue() {
    let mut state = state();
    assert!(state.open());
    assert!(matches!(
        state.enqueue_message(vec![1]),
        MessageEnqueue::Accepted(None)
    ));
    assert!(matches!(
        state.enqueue_message(vec![2]),
        MessageEnqueue::Accepted(None)
    ));
    assert_eq!(state.take_message().expect("first message"), vec![1]);
    assert_eq!(state.take_message().expect("second message"), vec![2]);
    assert!(state.close(1000).is_none());
    assert_eq!(
        state
            .take_message()
            .expect_err("closed connection must report EOF")
            .kind(),
        io::ErrorKind::UnexpectedEof
    );
}

#[test]
fn oversized_message_fails_without_retaining_data() {
    let mut state = state();
    assert!(state.open());
    let MessageEnqueue::Rejected { error, waker } = state.enqueue_message(vec![0; 5]) else {
        panic!("oversized message must be rejected");
    };
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(waker.is_none());
    assert_eq!(
        state
            .take_message()
            .expect_err("oversized message must fail")
            .kind(),
        io::ErrorKind::InvalidData
    );
    assert!(matches!(
        state.enqueue_message(vec![1]),
        MessageEnqueue::Rejected { .. }
    ));
}

#[test]
fn queue_overflow_is_terminal_and_clears_messages() {
    let mut state = state();
    assert!(state.open());
    assert!(matches!(
        state.enqueue_message(vec![1]),
        MessageEnqueue::Accepted(None)
    ));
    assert!(matches!(
        state.enqueue_message(vec![2]),
        MessageEnqueue::Accepted(None)
    ));
    let MessageEnqueue::Rejected { error, waker } = state.enqueue_message(vec![3]) else {
        panic!("queue overflow must be rejected");
    };
    assert_eq!(error.kind(), io::ErrorKind::OutOfMemory);
    assert!(waker.is_none());
    assert_eq!(
        state
            .take_message()
            .expect_err("queue overflow must fail")
            .kind(),
        io::ErrorKind::OutOfMemory
    );
}

#[test]
fn dropped_receive_unregisters_waiter() {
    let state = Arc::new(Mutex::new(state()));
    let mut receive = WebSocketReceive::new(Arc::clone(&state));
    let (waker, _) = counting_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        Pin::new(&mut receive).poll(&mut context),
        Poll::Pending
    ));
    drop(receive);

    let mut replacement = WebSocketReceive::new(Arc::clone(&state));
    assert!(matches!(
        Pin::new(&mut replacement).poll(&mut context),
        Poll::Pending
    ));
}

#[test]
fn second_receive_is_rejected_without_replacing_first_waiter() {
    let state = Arc::new(Mutex::new(state()));
    let mut first = WebSocketReceive::new(Arc::clone(&state));
    let mut second = WebSocketReceive::new(Arc::clone(&state));
    let (first_waker, _) = counting_waker();
    let (second_waker, _) = counting_waker();
    let mut first_context = Context::from_waker(&first_waker);
    let mut second_context = Context::from_waker(&second_waker);
    assert!(matches!(
        Pin::new(&mut first).poll(&mut first_context),
        Poll::Pending
    ));
    let Poll::Ready(result) = Pin::new(&mut second).poll(&mut second_context) else {
        panic!("second receive must resolve with an error");
    };
    assert_eq!(
        result.expect_err("second receive must be rejected").kind(),
        io::ErrorKind::AlreadyExists
    );
}

#[test]
fn enqueue_wakes_the_pending_receive() {
    let state = Arc::new(Mutex::new(state()));
    let mut receive = WebSocketReceive::new(Arc::clone(&state));
    let (waker, wake_count) = counting_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        Pin::new(&mut receive).poll(&mut context),
        Poll::Pending
    ));

    let enqueue = state
        .lock()
        .expect("test state lock must remain healthy")
        .enqueue_message(vec![7]);
    let MessageEnqueue::Accepted(Some(waiter)) = enqueue else {
        panic!("pending receive must provide its waiter to the producer");
    };
    waiter.wake();
    assert_eq!(wake_count.load(Ordering::Relaxed), 1);

    let Poll::Ready(Ok(message)) = Pin::new(&mut receive).poll(&mut context) else {
        panic!("woken receive must return the queued message");
    };
    assert_eq!(message, vec![7]);
}

#[test]
fn pending_receive_accepts_a_replacement_executor_waker() {
    let state = Arc::new(Mutex::new(state()));
    let mut receive = WebSocketReceive::new(Arc::clone(&state));
    let (first_waker, _) = counting_waker();
    let (second_waker, second_wake_count) = counting_waker();
    let mut first_context = Context::from_waker(&first_waker);
    let mut second_context = Context::from_waker(&second_waker);

    assert!(matches!(
        Pin::new(&mut receive).poll(&mut first_context),
        Poll::Pending
    ));
    assert!(matches!(
        Pin::new(&mut receive).poll(&mut second_context),
        Poll::Pending
    ));

    let enqueue = state
        .lock()
        .expect("test state lock must remain healthy")
        .enqueue_message(vec![9]);
    let MessageEnqueue::Accepted(Some(waiter)) = enqueue else {
        panic!("replacement waker must remain registered");
    };
    waiter.wake();
    assert_eq!(second_wake_count.load(Ordering::Relaxed), 1);
}

#[test]
fn dropped_open_unregisters_waiter() {
    let state = Arc::new(Mutex::new(state()));
    let mut open = WebSocketOpen::new(Arc::clone(&state));
    let (waker, _) = counting_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        Pin::new(&mut open).poll(&mut context),
        Poll::Pending
    ));
    drop(open);

    let mut replacement = WebSocketOpen::new(Arc::clone(&state));
    assert!(matches!(
        Pin::new(&mut replacement).poll(&mut context),
        Poll::Pending
    ));
}

#[test]
fn open_resolves_and_rejects_second_waiter() {
    let state = Arc::new(Mutex::new(state()));
    let mut first = WebSocketOpen::new(Arc::clone(&state));
    let mut second = WebSocketOpen::new(Arc::clone(&state));
    let (first_waker, _) = counting_waker();
    let (second_waker, _) = counting_waker();
    let mut first_context = Context::from_waker(&first_waker);
    let mut second_context = Context::from_waker(&second_waker);
    assert!(matches!(
        Pin::new(&mut first).poll(&mut first_context),
        Poll::Pending
    ));
    let Poll::Ready(result) = Pin::new(&mut second).poll(&mut second_context) else {
        panic!("second OPEN waiter must be rejected");
    };
    assert_eq!(
        result.expect_err("second OPEN waiter must fail").kind(),
        io::ErrorKind::AlreadyExists
    );

    let waiter = state
        .lock()
        .expect("test state lock must remain healthy")
        .take_open_waiter()
        .expect("first OPEN waiter must be registered");
    waiter.wake();
    assert!(state.lock().expect("test state lock").open());
    assert!(matches!(
        Pin::new(&mut first).poll(&mut first_context),
        Poll::Ready(Ok(()))
    ));
}

fn counting_waker() -> (Waker, Arc<AtomicUsize>) {
    let wake_count = Arc::new(AtomicUsize::new(0));
    (
        Waker::from(Arc::new(CountingWake(Arc::clone(&wake_count)))),
        wake_count,
    )
}

struct CountingWake(Arc<AtomicUsize>);

impl std::task::Wake for CountingWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}
