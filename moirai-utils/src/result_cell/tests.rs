use super::{RESULT_PENDING, RESULT_READY, RESULT_WAITING, ResultCell, Waiter};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicUsize, Ordering},
};
use std::task::{Wake, Waker};
use std::thread;
use std::time::Duration;

struct CountingWake(AtomicUsize);

impl Wake for CountingWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct DropCountingWaiter {
    drops: Arc<AtomicUsize>,
    wakes: Arc<AtomicUsize>,
}

impl Drop for DropCountingWaiter {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl Waiter for DropCountingWaiter {
    const REPLACE_ON_REPEAT: bool = true;

    fn wake(self) {
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

struct TrackedResult {
    payload: &'static str,
    drops: Arc<AtomicUsize>,
}

impl Drop for TrackedResult {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn take_observes_the_published_result_exactly_once() {
    let cell = ResultCell::<u32, Waker>::new();
    assert_eq!(cell.try_take_ready(), None);
    assert!(!cell.is_completed());

    cell.complete(7);
    assert!(cell.is_completed());
    assert_eq!(cell.try_take_ready(), Some(7));
    assert!(cell.is_completed(), "consumption is still completion");
    assert_eq!(cell.try_take_ready(), None, "result taken twice");
}

#[test]
fn complete_before_register_does_not_wake_and_the_recheck_finds_it() {
    // The PENDING -> WRITING path has no waiter to wake; the consumer's
    // re-check after `register` is what makes this liveness-safe.
    let wake = Arc::new(CountingWake(AtomicUsize::new(0)));
    let cell = ResultCell::<u32, Waker>::new();

    cell.complete(1);
    assert_eq!(cell.try_take_ready(), Some(1));

    // SAFETY: this test thread is the cell's only consumer.
    unsafe { cell.register(&Waker::from(Arc::clone(&wake))) };
    assert_eq!(
        wake.0.load(Ordering::SeqCst),
        0,
        "nothing was parked to wake"
    );
}

#[test]
fn complete_wakes_a_registered_waker_once() {
    let wake = Arc::new(CountingWake(AtomicUsize::new(0)));
    let cell = ResultCell::<u32, Waker>::new();

    // SAFETY: this test thread is the cell's only consumer.
    unsafe { cell.register(&Waker::from(Arc::clone(&wake))) };
    cell.complete(2);

    assert_eq!(wake.0.load(Ordering::SeqCst), 1);
    assert_eq!(cell.try_take_ready(), Some(2));
}

#[test]
fn a_waker_may_be_replaced_but_a_thread_registration_is_once() {
    const _: () = assert!(<Waker as Waiter>::REPLACE_ON_REPEAT);
    const _: () = assert!(!<thread::Thread as Waiter>::REPLACE_ON_REPEAT);

    let first = Arc::new(CountingWake(AtomicUsize::new(0)));
    let second = Arc::new(CountingWake(AtomicUsize::new(0)));
    let cell = ResultCell::<u32, Waker>::new();

    // SAFETY: this test thread is the cell's only consumer.
    unsafe { cell.register(&Waker::from(Arc::clone(&first))) };
    // SAFETY: this test thread is the cell's only consumer.
    unsafe { cell.register(&Waker::from(Arc::clone(&second))) };
    cell.complete(3);

    assert_eq!(first.0.load(Ordering::SeqCst), 0, "stale waker was woken");
    assert_eq!(second.0.load(Ordering::SeqCst), 1, "newest waker must win");
}

#[test]
fn a_thread_waiter_is_unparked_across_the_hand_off() {
    let cell = ResultCell::<u32, thread::Thread>::new();
    // SAFETY: this test thread is the cell's only consumer.
    unsafe { cell.register(&thread::current()) };
    cell.complete(4);

    // `unpark` leaves a token, so a park here returns immediately. The
    // timeout only exists so a lost wake fails as a bounded delay rather
    // than a hung suite.
    let start = std::time::Instant::now();
    thread::park_timeout(Duration::from_secs(5));
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "the parked thread was never unparked"
    );
    assert_eq!(cell.try_take_ready(), Some(4));
}

#[test]
fn drop_releases_the_untaken_result_and_the_parked_waiter() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let cell = ResultCell::<TrackedResult, Waker>::new();
    // SAFETY: this test thread is the cell's only consumer.
    unsafe { cell.register(&Waker::noop().clone()) };
    cell.complete(TrackedResult {
        payload: "untaken",
        drops: Arc::clone(&dropped),
    });
    drop(cell);

    assert_eq!(dropped.load(Ordering::SeqCst), 1, "untaken result leaked");
}

#[test]
fn final_owner_cross_thread_disposes_untaken_and_preserves_consumed_result() {
    let untaken_drops = Arc::new(AtomicUsize::new(0));
    let untaken = Arc::new(ResultCell::<TrackedResult, Waker>::new());
    let producer_cell = Arc::clone(&untaken);
    let producer = thread::spawn({
        let untaken_drops = Arc::clone(&untaken_drops);
        move || {
            producer_cell.complete(TrackedResult {
                payload: "untaken",
                drops: untaken_drops,
            });
        }
    });
    producer
        .join()
        .expect("invariant: producer thread returns normally");
    thread::spawn(move || drop(untaken))
        .join()
        .expect("invariant: final owner thread returns normally");
    assert_eq!(untaken_drops.load(Ordering::SeqCst), 1);

    let consumed_drops = Arc::new(AtomicUsize::new(0));
    let consumed = Arc::new(ResultCell::<TrackedResult, Waker>::new());
    let producer_cell = Arc::clone(&consumed);
    let producer = thread::spawn({
        let consumed_drops = Arc::clone(&consumed_drops);
        move || {
            producer_cell.complete(TrackedResult {
                payload: "consumed",
                drops: consumed_drops,
            });
        }
    });
    producer
        .join()
        .expect("invariant: producer thread returns normally");
    let result = consumed
        .try_take_ready()
        .expect("invariant: joined producer published the result");
    assert_eq!(result.payload, "consumed");
    assert_eq!(consumed_drops.load(Ordering::SeqCst), 0);
    assert_eq!(
        consumed.try_take_ready().map(|result| result.payload),
        None,
        "result was returned twice"
    );
    thread::spawn(move || drop(consumed))
        .join()
        .expect("invariant: final owner thread returns normally");
    assert_eq!(consumed_drops.load(Ordering::SeqCst), 0);
    drop(result);
    assert_eq!(consumed_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn drop_releases_exactly_one_retained_waiter() {
    let drops = Arc::new(AtomicUsize::new(0));
    let wakes = Arc::new(AtomicUsize::new(0));
    let cell = ResultCell::<usize, DropCountingWaiter>::new();
    {
        let waiter = DropCountingWaiter {
            drops: Arc::clone(&drops),
            wakes: Arc::clone(&wakes),
        };
        // SAFETY: this test thread is the cell's only consumer.
        unsafe { cell.register(&waiter) };
    }
    assert_eq!(drops.swap(0, Ordering::SeqCst), 1, "caller waiter retained");

    drop(cell);

    assert_eq!(
        drops.load(Ordering::SeqCst),
        1,
        "retained waiter leaked or duplicated"
    );
    assert_eq!(wakes.load(Ordering::SeqCst), 0, "pending waiter was woken");
}

#[test]
fn an_aligned_state_word_keeps_the_publish_off_the_result_line() {
    use crate::cache::{CacheAligned, DESTRUCTIVE_INTERFERENCE_SIZE};
    use core::mem::{align_of, size_of};
    use core::sync::atomic::AtomicU8;

    type Aligned = ResultCell<u32, thread::Thread, CacheAligned<AtomicU8>>;

    // This is the layout `moirai-core`'s blocking slot is written against: the
    // state owns one interference sector and the result lies outside it, so the
    // producer's publish does not invalidate the line the consumer reads the
    // result from. The packed default is what the async handle wants instead,
    // one allocation per spawned task. The sector claim is the whole assertion —
    // two fields cannot overlap, so a state word that fills its sector already
    // excludes the result — and it is a property of the types, so it is checked
    // at compile time.
    const _: () = assert!(size_of::<CacheAligned<AtomicU8>>() >= DESTRUCTIVE_INTERFERENCE_SIZE);
    const _: () = assert!(align_of::<Aligned>() >= DESTRUCTIVE_INTERFERENCE_SIZE);

    let cell = Aligned::new();
    cell.complete(9);
    assert_eq!(cell.try_take_ready(), Some(9));
}

#[test]
fn the_packed_default_does_not_pad_each_cell() {
    // One allocation per spawned async task: the packed cell must stay the size
    // of its parts, not a cache line. The check is the compile-time assertion
    // below, so a regression fails the build rather than this run.
    const PACKED: usize = core::mem::size_of::<ResultCell<u32, Waker>>();
    const _: () = assert!(
        PACKED < crate::cache::DESTRUCTIVE_INTERFERENCE_SIZE,
        "the packed cell grew past a cache line"
    );
}

struct CloneFailureWaiter<const REPLACE_ON_REPEAT: bool> {
    clones: Arc<AtomicUsize>,
    wakes: Arc<AtomicUsize>,
}

impl<const REPLACE_ON_REPEAT: bool> Clone for CloneFailureWaiter<REPLACE_ON_REPEAT> {
    fn clone(&self) -> Self {
        match self.clones.fetch_add(1, Ordering::SeqCst) {
            1 => panic!("replacement clone"),
            _ => Self {
                clones: Arc::clone(&self.clones),
                wakes: Arc::clone(&self.wakes),
            },
        }
    }
}

impl<const REPLACE_ON_REPEAT: bool> Waiter for CloneFailureWaiter<REPLACE_ON_REPEAT> {
    const REPLACE_ON_REPEAT: bool = REPLACE_ON_REPEAT;

    fn wake(self) {
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn first_clone_panic_leaves_the_pending_cell_unmodified() {
    let waiter = CloneFailureWaiter::<true> {
        clones: Arc::new(AtomicUsize::new(1)),
        wakes: Arc::new(AtomicUsize::new(0)),
    };
    let cell = ResultCell::<usize, CloneFailureWaiter<true>>::new();

    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: this thread is the cell's only consumer.
        unsafe { cell.register(&waiter) };
    })) {
        Err(panic) => panic,
        Ok(()) => panic!("first clone unexpectedly succeeded"),
    };

    assert_eq!(panic.downcast_ref::<&str>(), Some(&"replacement clone"));
    assert_eq!(cell.state.load(Ordering::Acquire), RESULT_PENDING);
    assert!(!cell.has_registered_waiter());
    // SAFETY: no registration succeeded and this thread is the sole consumer.
    assert!(unsafe { (*cell.waiter.get()).is_none() });
    cell.complete(81);
    assert_eq!(cell.try_take_ready(), Some(81));
    assert_eq!(cell.try_take_ready(), None, "result taken twice");
    assert_eq!(waiter.clones.load(Ordering::SeqCst), 2);
    assert_eq!(waiter.wakes.load(Ordering::SeqCst), 0);
}

#[test]
fn replacement_clone_panic_preserves_the_published_waiter() {
    let waiter = CloneFailureWaiter::<true> {
        clones: Arc::new(AtomicUsize::new(0)),
        wakes: Arc::new(AtomicUsize::new(0)),
    };
    let cell = ResultCell::<usize, CloneFailureWaiter<true>>::new();

    // SAFETY: this thread is the cell's only consumer.
    unsafe { cell.register(&waiter) };
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the first registration returned; this is the only consumer.
        unsafe { cell.register(&waiter) };
    })) {
        Err(panic) => panic,
        Ok(()) => panic!("replacement clone unexpectedly succeeded"),
    };

    assert_eq!(panic.downcast_ref::<&str>(), Some(&"replacement clone"));
    assert_eq!(cell.state.load(Ordering::Acquire), RESULT_WAITING);
    cell.complete(71);
    assert_eq!(cell.try_take_ready(), Some(71));
    assert_eq!(waiter.clones.load(Ordering::SeqCst), 2);
    assert_eq!(waiter.wakes.load(Ordering::SeqCst), 1);
}

#[test]
fn repeated_nonreplacing_registration_keeps_the_first_waiter() {
    let waiter = CloneFailureWaiter::<false> {
        clones: Arc::new(AtomicUsize::new(0)),
        wakes: Arc::new(AtomicUsize::new(0)),
    };
    let cell = ResultCell::<usize, CloneFailureWaiter<false>>::new();

    // SAFETY: this thread is the cell's only consumer.
    unsafe { cell.register(&waiter) };
    // SAFETY: the first registration returned; this is the only consumer.
    unsafe { cell.register(&waiter) };

    assert_eq!(waiter.clones.load(Ordering::SeqCst), 1);
    assert!(cell.has_registered_waiter());
    assert_eq!(cell.state.load(Ordering::Acquire), RESULT_WAITING);
    // SAFETY: this thread is the sole consumer and completion has not started.
    assert!(unsafe { (*cell.waiter.get()).is_some() });
    cell.complete(83);
    assert_eq!(cell.try_take_ready(), Some(83));
    assert_eq!(cell.try_take_ready(), None, "result taken twice");
    assert_eq!(waiter.wakes.load(Ordering::SeqCst), 1);
}

#[derive(Clone)]
struct ReentrantDropWaiter {
    completion: Option<ReentrantCompletion>,
    wakes: Arc<AtomicUsize>,
}

#[derive(Clone)]
struct ReentrantCompletion {
    cell: Weak<ResultCell<usize, ReentrantDropWaiter>>,
    calls: Arc<AtomicUsize>,
    value: usize,
}

impl Waiter for ReentrantDropWaiter {
    const REPLACE_ON_REPEAT: bool = true;

    fn wake(self) {
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for ReentrantDropWaiter {
    fn drop(&mut self) {
        let Some(completion) = self.completion.take() else {
            return;
        };
        completion.calls.fetch_add(1, Ordering::SeqCst);
        let cell = completion
            .cell
            .upgrade()
            .expect("invariant: the reentrant completion test retains the result cell");
        cell.complete(completion.value);
    }
}

#[test]
fn replaced_waiter_drop_may_complete_the_same_cell() {
    let cell = Arc::new(ResultCell::<usize, ReentrantDropWaiter>::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let old_wakes = Arc::new(AtomicUsize::new(0));
    let new_wakes = Arc::new(AtomicUsize::new(0));
    let mut old = ReentrantDropWaiter {
        completion: Some(ReentrantCompletion {
            cell: Arc::downgrade(&cell),
            calls: Arc::clone(&calls),
            value: 73,
        }),
        wakes: Arc::clone(&old_wakes),
    };

    // SAFETY: this thread is the cell's only consumer.
    unsafe { cell.register(&old) };
    old.completion = None;
    let replacement = ReentrantDropWaiter {
        completion: None,
        wakes: Arc::clone(&new_wakes),
    };
    // SAFETY: the first registration returned; this is the only consumer.
    unsafe { cell.register(&replacement) };

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(cell.state.load(Ordering::Acquire), RESULT_READY);
    assert_eq!(cell.try_take_ready(), Some(73));
    assert_eq!(old_wakes.load(Ordering::SeqCst), 0);
    assert_eq!(new_wakes.load(Ordering::SeqCst), 1);
}

#[derive(Clone)]
struct PanicDropWaiter {
    panic_on_drop: bool,
    panic_drops: Arc<AtomicUsize>,
    wakes: Arc<AtomicUsize>,
}

impl Waiter for PanicDropWaiter {
    const REPLACE_ON_REPEAT: bool = true;

    fn wake(self) {
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for PanicDropWaiter {
    fn drop(&mut self) {
        if self.panic_on_drop {
            self.panic_drops.fetch_add(1, Ordering::SeqCst);
            panic!("replaced waiter drop");
        }
    }
}

#[test]
fn replacement_survives_replaced_waiter_drop_panic() {
    let panic_drops = Arc::new(AtomicUsize::new(0));
    let old_wakes = Arc::new(AtomicUsize::new(0));
    let new_wakes = Arc::new(AtomicUsize::new(0));
    let cell = ResultCell::<usize, PanicDropWaiter>::new();
    let mut old = PanicDropWaiter {
        panic_on_drop: true,
        panic_drops: Arc::clone(&panic_drops),
        wakes: Arc::clone(&old_wakes),
    };

    // SAFETY: this thread is the cell's only consumer.
    unsafe { cell.register(&old) };
    old.panic_on_drop = false;
    let replacement = PanicDropWaiter {
        panic_on_drop: false,
        panic_drops: Arc::clone(&panic_drops),
        wakes: Arc::clone(&new_wakes),
    };
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the first registration returned; this is the only consumer.
        unsafe { cell.register(&replacement) };
    })) {
        Err(panic) => panic,
        Ok(()) => panic!("replaced waiter drop unexpectedly succeeded"),
    };

    assert_eq!(panic.downcast_ref::<&str>(), Some(&"replaced waiter drop"));
    assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
    assert_eq!(cell.state.load(Ordering::Acquire), RESULT_WAITING);
    cell.complete(79);
    assert_eq!(cell.try_take_ready(), Some(79));
    assert_eq!(old_wakes.load(Ordering::SeqCst), 0);
    assert_eq!(new_wakes.load(Ordering::SeqCst), 1);
}
