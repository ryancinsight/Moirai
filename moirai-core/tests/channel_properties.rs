//! Property coverage for the public channel primitives.
//!
//! Model-based and round-trip properties over generated operation sequences,
//! complementing the curated unit tests and the loom channel models. Every
//! concurrent case terminates deterministically: threads join on exact
//! expected counts, never on wall-clock timing.

use moirai_core::channel::{mpmc, spsc};

use proptest::prelude::*;

/// A generated operation against a FIFO model of capacity `cap`.
#[derive(Clone, Debug, PartialEq)]
enum SpscOp {
    Push(u64),
    Pop,
}

fn spsc_ops() -> impl Strategy<Value = Vec<SpscOp>> {
    (proptest::collection::vec(any::<u64>(), 0..64), 2usize..=160).prop_map(|(values, flips)| {
        let mut ops = Vec::with_capacity(values.len() + flips);
        let mut next = values.iter().copied();
        // Interleave pushes (while values remain) with pops by parity.
        for push_turn in (0..flips).map(|i| i % 2 == 0) {
            if push_turn {
                if let Some(v) = next.next() {
                    ops.push(SpscOp::Push(v));
                    continue;
                }
            }
            ops.push(SpscOp::Pop);
        }
        ops.extend(next.map(SpscOp::Push));
        ops
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn spsc_shared_matches_fifo_model(
        cap_exp in 1u32..7,
        ops in spsc_ops(),
    ) {
        // Power-of-two capacities exercise index masking rather than modulo.
        let cap = 1usize << cap_exp;
        let mut ring = spsc::SpscRing::<u64>::new(cap);
        let (producer, consumer) = ring.split();
        let mut model: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
        for op in ops {
            match op {
                SpscOp::Push(v) => match producer.try_send(v) {
                    Ok(()) => model.push_back(v),
                    Err(_) => prop_assert_eq!(model.len(), cap),
                },
                SpscOp::Pop => {
                    let got = consumer.try_recv().ok();
                    prop_assert_eq!(got.as_ref(), model.front());
                    model.pop_front();
                }
            }
        }
    }

    #[test]
    fn spsc_wraparound_preserves_order(
        cap_exp in 1u32..6,
        total in 1usize..512,
    ) {
        // Push sequential values far beyond one lap of the ring, draining as
        // needed, then verify the full sequence survives mask wraparound.
        let cap = 1usize << cap_exp;
        let mut ring = spsc::SpscRing::<usize>::new(cap);
        let (producer, consumer) = ring.split();
        let mut received = Vec::with_capacity(total);
        for sent in 0..total {
            loop {
                match producer.try_send(sent) {
                    Ok(()) => break,
                    Err(_) => {
                        if let Ok(v) = consumer.try_recv() {
                            received.push(v);
                        }
                    }
                }
            }
        }
        while let Ok(v) = consumer.try_recv() {
            received.push(v);
        }
        let expected: Vec<usize> = (0..total).collect();
        prop_assert_eq!(received, expected);
    }

    #[test]
    fn mpmc_roundtrip_preserves_multiset(count in 0usize..512) {
        let (tx, rx) = mpmc::MpmcChannel::channel(Some(16));
        let values: Vec<usize> = (0..count).collect();

        let producer_tx = tx.clone();
        let producer = std::thread::spawn(move || {
            for v in &values {
                producer_tx.send(*v)?;
            }
            Ok::<(), moirai_core::channel::ChannelError>(())
        });
        let consumer = std::thread::spawn(move || {
            let mut got = Vec::with_capacity(count);
            while got.len() < count {
                match rx.recv() {
                    Ok(v) => got.push(v),
                    Err(moirai_core::channel::ChannelError::Closed) => break,
                    Err(e) => panic!("unexpected channel error: {e:?}"),
                }
            }
            got
        });

        producer.join().expect("producer").expect("send errors");
        drop(tx);
        let mut got = consumer.join().expect("consumer");
        got.sort_unstable();
        let expected: Vec<usize> = (0..count).collect();
        prop_assert_eq!(got, expected);
    }
}
