use super::*;

struct AlternatingReadyStream {
    next: usize,
    end: usize,
    yield_pending: bool,
}

impl Stream for AlternatingReadyStream {
    type Item = core::future::Ready<usize>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.next == self.end {
            return Poll::Ready(None);
        }
        if self.yield_pending {
            self.yield_pending = false;
            context.waker().wake_by_ref();
            return Poll::Pending;
        }
        let value = self.next;
        self.next += 1;
        self.yield_pending = true;
        Poll::Ready(Some(core::future::ready(value)))
    }
}

struct DropOutput {
    value: usize,
    drops: Arc<AtomicUsize>,
    panic_on_drop: bool,
}

impl Drop for DropOutput {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
        assert!(!self.panic_on_drop, "completed output drop panic sentinel");
    }
}

struct ReadyDropPanic {
    value: usize,
    drops: Arc<AtomicUsize>,
    panic_on_drop: bool,
}

impl Future for ReadyDropPanic {
    type Output = usize;

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(self.as_ref().get_ref().value)
    }
}

impl Drop for ReadyDropPanic {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
        assert!(!self.panic_on_drop, "ready future drop panic sentinel");
    }
}

#[test]
fn unknown_sequential_stream_retains_one_output_cell() {
    let stream = AlternatingReadyStream {
        next: 0,
        end: 97,
        yield_pending: false,
    };
    let mut buffered = retained_buffered(stream, usize::MAX);
    let values = futures::executor::block_on(async {
        let mut values = Vec::new();
        while let Some(value) = buffered.next().await {
            values.push(value);
        }
        values
    });

    assert_eq!(values, (0..97).collect::<Vec<_>>());
    assert_eq!(buffered.storage_capacities(), (1, 1));
}

#[test]
fn ordered_outputs_cross_geometric_blocks_and_ragged_tail() {
    let mut next = 0_usize;
    let stream = futures::stream::poll_fn(move |_| {
        if next == 130 {
            Poll::Ready(None)
        } else {
            let value = next;
            next += 1;
            Poll::Ready(Some(pending_once(value)))
        }
    });
    let mut buffered = retained_buffered(stream, 130);
    let values = futures::executor::block_on(async {
        let mut values = Vec::new();
        while let Some(value) = buffered.next().await {
            values.push(value);
        }
        values
    });

    assert_eq!(values, (0..130).collect::<Vec<_>>());
    assert_eq!(buffered.storage_capacities(), (130, 130));
}

#[test]
fn dropping_ordered_stream_drops_completed_outputs_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let stream = futures::stream::iter((0..5).map({
        let drops = Arc::clone(&drops);
        move |value| {
            core::future::ready(DropOutput {
                value,
                drops: Arc::clone(&drops),
                panic_on_drop: false,
            })
        }
    }));
    let mut buffered = retained_buffered(stream, 5);
    let first = futures::executor::block_on(buffered.next())
        .expect("ordered stream must yield its first completed output");
    assert_eq!(first.value, 0);
    drop(first);
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    drop(buffered);
    assert_eq!(drops.load(Ordering::SeqCst), 5);
}

#[test]
fn panicking_completed_output_drop_releases_the_remaining_outputs() {
    let drops = Arc::new(AtomicUsize::new(0));
    let stream = futures::stream::iter((0..3).map({
        let drops = Arc::clone(&drops);
        move |value| {
            core::future::ready(DropOutput {
                value,
                drops: Arc::clone(&drops),
                panic_on_drop: value == 1,
            })
        }
    }));
    let mut buffered = retained_buffered(stream, 3);
    let first = futures::executor::block_on(buffered.next())
        .expect("ordered stream must yield its first completed output");
    drop(first);

    let result = catch_unwind(AssertUnwindSafe(|| drop(buffered)));
    assert!(result.is_err(), "completed output destructor must panic");
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn ready_future_drop_panic_does_not_abort_stream_cleanup() {
    let drops = Arc::new(AtomicUsize::new(0));
    let result = catch_unwind(AssertUnwindSafe({
        let drops = Arc::clone(&drops);
        move || {
            let stream = futures::stream::iter((0..2).map(|value| ReadyDropPanic {
                value,
                drops: Arc::clone(&drops),
                panic_on_drop: value == 0,
            }));
            futures::executor::block_on(retained_buffered(stream, 2).collect::<Vec<_>>())
        }
    }));

    assert!(result.is_err(), "ready future destructor must panic");
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn ordered_slots_preserve_full_width_output_values() {
    let stream = futures::stream::iter([usize::MAX, 3, 11].map(pending_once));
    let values = futures::executor::block_on(retained_buffered(stream, 3).collect::<Vec<_>>());
    assert_eq!(values, [usize::MAX, 3, 11]);
}
