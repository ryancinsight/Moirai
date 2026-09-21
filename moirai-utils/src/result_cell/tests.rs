use super::{ResultCell, Waiter};
use std::sync::{
    Arc,
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

#[test]
fn take_observes_the_published_result_exactly_once() {
    let cell = ResultCell::<u32, Waker>::new();
    assert_eq!(cell.try_take_ready(), None);
    assert!(!cell.is_completed());

    cell.complete(7);
    assert!(cell.is_completed());
    assert_eq!(cell.try_take_ready(), Some(7));
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

    cell.register(&Waker::from(Arc::clone(&wake)));
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

    cell.register(&Waker::from(Arc::clone(&wake)));
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

    cell.register(&Waker::from(Arc::clone(&first)));
    cell.register(&Waker::from(Arc::clone(&second)));
    cell.complete(3);

    assert_eq!(first.0.load(Ordering::SeqCst), 0, "stale waker was woken");
    assert_eq!(second.0.load(Ordering::SeqCst), 1, "newest waker must win");
}

#[test]
fn a_thread_waiter_is_unparked_across_the_hand_off() {
    let cell = ResultCell::<u32, thread::Thread>::new();
    cell.register(&thread::current());
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

    struct Tracked(Arc<AtomicUsize>);

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let cell = ResultCell::<Tracked, Waker>::new();
    cell.register(&Waker::noop().clone());
    cell.complete(Tracked(Arc::clone(&dropped)));
    drop(cell);

    assert_eq!(dropped.load(Ordering::SeqCst), 1, "untaken result leaked");
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
