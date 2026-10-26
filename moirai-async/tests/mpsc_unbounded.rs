//! Model-based validation of synchronous admission and receive cancellation.

use moirai_async::sync::mpsc::unbounded_channel;
use proptest::prelude::*;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

#[derive(Clone, Debug)]
enum Operation {
    Send(Vec<u8>),
    Receive,
    CancelReceive,
    Close,
}

fn operation() -> impl Strategy<Value = Operation> {
    prop_oneof![
        prop::collection::vec(any::<u8>(), 0..1024).prop_map(Operation::Send),
        Just(Operation::Receive),
        Just(Operation::CancelReceive),
        Just(Operation::Close),
    ]
}

proptest! {
    #[test]
    fn arbitrary_operations_match_owned_fifo_model(operations in prop::collection::vec(operation(), 0..256)) {
        let _trace = tracing::subscriber::set_default(tracing_subscriber::fmt().with_test_writer().finish());
        let _arrange = tracing::info_span!("arrange").entered();
        let (sender, mut receiver) = unbounded_channel();
        let mut model = VecDeque::new();
        let mut closed = false;
        let mut cx = Context::from_waker(Waker::noop());
        drop(_arrange);

        for operation in operations {
            let _act = tracing::info_span!("act", ?operation).entered();
            match operation {
                Operation::Send(value) => {
                    let expected = if closed { Err(value.clone()) } else {
                        model.push_back(value.clone());
                        Ok(())
                    };
                    let actual = sender.enqueue(value);
                    let _assert = tracing::info_span!("assert").entered();
                    prop_assert_eq!(actual, expected);
                }
                Operation::Receive => {
                    let expected = model.pop_front();
                    let actual = receiver.try_recv();
                    let _assert = tracing::info_span!("assert").entered();
                    prop_assert_eq!(actual, expected);
                }
                Operation::CancelReceive => {
                    let mut receive = receiver.recv();
                    let expected = model.pop_front().map_or_else(
                        || if closed { Poll::Ready(Err(())) } else { Poll::Pending },
                        |value| Poll::Ready(Ok(value)),
                    );
                    let actual = Pin::new(&mut receive).poll(&mut cx);
                    drop(receive);
                    let _assert = tracing::info_span!("assert").entered();
                    prop_assert_eq!(actual, expected);
                }
                Operation::Close => {
                    receiver.close();
                    closed = true;
                }
            }
        }
        drop(sender);
        let _assert = tracing::info_span!("assert").entered();
        while let Some(expected) = model.pop_front() {
            prop_assert_eq!(receiver.poll_recv(&mut cx), Poll::Ready(Some(expected)));
        }
        prop_assert_eq!(receiver.poll_recv(&mut cx), Poll::Ready(None));
    }
}
