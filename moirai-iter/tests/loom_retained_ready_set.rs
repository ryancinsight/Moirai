//! Loom model for retained-slot wake publication and claiming.
//!
//! Production wake tokens publish one bit with `fetch_or(Release)`. The stream
//! claims one bit with an Acquire load followed by `compare_exchange_weak`
//! using AcqRel/Acquire. This two-bit model races both producers against the
//! consumer and proves each distinct ready slot is either claimed during the
//! race or remains published for the next poll, never lost or duplicated.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run -p moirai-iter --test loom_retained_ready_set --release`

#![cfg(loom)]

use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

fn take_one(word: &AtomicUsize) -> Option<usize> {
    let mut observed = word.load(Ordering::Acquire);
    while observed != 0 {
        let bit_index = observed.trailing_zeros() as usize;
        let bit = 1usize << bit_index;
        match word.compare_exchange_weak(
            observed,
            observed & !bit,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return Some(bit_index),
            Err(current) => observed = current,
        }
    }
    None
}

fn record_claim(seen: &AtomicUsize, bit_index: usize) {
    let bit = 1usize << bit_index;
    let prior = seen.fetch_or(bit, Ordering::SeqCst);
    assert_eq!(prior & bit, 0, "a ready slot was claimed twice");
}

#[test]
fn concurrent_ready_bits_are_claimed_or_remain_published() {
    loom::model(|| {
        let ready = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(AtomicUsize::new(0));

        let first_ready = Arc::clone(&ready);
        let first = thread::spawn(move || {
            first_ready.fetch_or(0b01, Ordering::Release);
        });

        let second_ready = Arc::clone(&ready);
        let second = thread::spawn(move || {
            second_ready.fetch_or(0b10, Ordering::Release);
        });

        let consumer_ready = Arc::clone(&ready);
        let consumer_seen = Arc::clone(&seen);
        let consumer = thread::spawn(move || {
            for _ in 0..2 {
                if let Some(bit_index) = take_one(&consumer_ready) {
                    record_claim(&consumer_seen, bit_index);
                }
                thread::yield_now();
            }
        });

        first.join().unwrap();
        second.join().unwrap();
        consumer.join().unwrap();

        while let Some(bit_index) = take_one(&ready) {
            record_claim(&seen, bit_index);
        }
        assert_eq!(seen.load(Ordering::SeqCst), 0b11);
        assert_eq!(ready.load(Ordering::SeqCst), 0);
    });
}
