use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_message_zero_copy() {
    struct NonClone {
        val: i32,
    }
    let msg = Message::new(NonClone { val: 42 });
    let msg_clone = msg.clone();
    assert_eq!(msg.data().val, 42);
    assert_eq!(msg_clone.data().val, 42);

    let Err(msg) = msg.try_unwrap() else {
        panic!("should fail to unwrap")
    };

    drop(msg_clone);
    let unwrapped_ok = msg.try_unwrap().unwrap();
    assert_eq!(unwrapped_ok.val, 42);
}

#[test]
fn test_broadcast_channel() {
    let channel = BroadcastChannel::new();
    let mut rx1 = channel.subscribe();
    let mut rx2 = channel.subscribe();

    channel.broadcast(42);

    assert_eq!(rx1.try_recv(), Some(42));
    assert_eq!(rx2.try_recv(), Some(42));

    // No new broadcasts
    assert_eq!(rx1.try_recv(), None);
}

#[test]
fn test_collective_ops() {
    let values = vec![1, 2, 3, 4];
    let result = CollectiveOps::all_reduce(values, |a, b| a + b);
    assert_eq!(result, vec![10, 10, 10, 10]);

    let data = vec![1, 2, 3, 4, 5, 6];
    let scattered = CollectiveOps::scatter(data, 3);
    assert_eq!(scattered.num_chunks(), 3);
    assert_eq!(scattered.chunks().next(), Some([1, 2].as_slice()));

    let gathered = CollectiveOps::gather(scattered);
    assert_eq!(gathered, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_collective_scatter_gather_round_trip() {
    let data: Vec<u64> = (0..100).collect();
    let scattered = CollectiveOps::scatter(data.clone(), 7);
    let chunks: Vec<Vec<u64>> = scattered.chunks().map(<[u64]>::to_vec).collect();
    assert_eq!(chunks.len(), 7);
    // Chunks tile the original data contiguously and in order.
    assert_eq!(chunks.concat(), data);
    assert_eq!(CollectiveOps::gather(scattered), data);
}

#[test]
fn test_collective_scatter_empty_and_zero_participants() {
    // The CSR form turns the historical `chunks(0)` panic into an empty buffer.
    let empty = CollectiveOps::scatter(Vec::<u64>::new(), 4);
    assert!(empty.is_empty());
    assert_eq!(empty.num_chunks(), 0);

    let single = CollectiveOps::scatter(vec![1, 2, 3], 0);
    assert_eq!(single.chunks().count(), 1);
    assert_eq!(single.chunks().next(), Some([1, 2, 3].as_slice()));
}

#[test]
fn test_collective_all_to_all_parity() {
    // scatter(10, 4) -> chunks [3, 3, 3, 1]: [1,2,3] [4,5,6] [7,8,9] [10].
    let data: Vec<u64> = (1..=10).collect();
    let chunked = CollectiveOps::scatter(data, 4);
    let expected: Vec<Vec<u64>> = vec![
        vec![1, 4, 7, 10], // column 0 of every row
        vec![2, 5, 8],     // column 1 (row [10] has no column 1)
        vec![3, 6, 9],     // column 2
        vec![],            // column 3 (no row reaches it)
    ];
    let transposed = CollectiveOps::all_to_all(chunked);
    let got: Vec<Vec<u64>> = transposed.chunks().map(<[u64]>::to_vec).collect();
    assert_eq!(got, expected);
}

#[test]
fn test_ring_buffer() {
    let rb = RingBuffer::new(4);

    assert!(rb.try_produce(1).is_ok());
    assert!(rb.try_produce(2).is_ok());

    assert_eq!(rb.try_consume(), Some(1));
    assert_eq!(rb.try_consume(), Some(2));
    assert_eq!(rb.try_consume(), None);
}

#[test]
fn test_ring_buffer_drop_safety() {
    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
    struct TrackDrop;
    impl Drop for TrackDrop {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    {
        let rb = RingBuffer::new(4);
        assert!(rb.try_produce(TrackDrop).is_ok());
        assert!(rb.try_produce(TrackDrop).is_ok());
        // Do not consume, just drop the RingBuffer
    }

    assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 2);
}

#[test]
fn test_pubsub() {
    let pubsub = PubSub::new();
    let rx = pubsub.subscribe("topic1");

    assert_eq!(pubsub.publish(&"topic1", 42).unwrap(), 1);
    assert_eq!(rx.try_recv().unwrap(), 42);

    assert_eq!(pubsub.publish(&"topic2", 99).unwrap(), 0);
}

#[test]
fn test_pubsub_prunes_dropped_subscribers() {
    let pubsub = PubSub::new();
    let survivors: Vec<_> = (0..4).map(|_| pubsub.subscribe("topic")).collect();
    assert_eq!(pubsub.subscriber_count(&"topic"), 4);

    // Drop 3 of 4 receivers; their senders are now closed but still listed.
    let keeper = survivors.into_iter().next().unwrap();

    // First publish delivers to the survivor and detects the closed channels.
    assert_eq!(pubsub.publish(&"topic", 7).unwrap(), 1);
    assert_eq!(
        pubsub.subscriber_count(&"topic"),
        1,
        "closed subscriber senders must be pruned during publish"
    );

    // Second publish sees only the pruned list; survivor receives both values.
    assert_eq!(pubsub.publish(&"topic", 8).unwrap(), 1);
    assert_eq!(keeper.try_recv().unwrap(), 7);
    assert_eq!(keeper.try_recv().unwrap(), 8);
}

#[test]
fn test_pubsub_removes_empty_topic_after_all_subscribers_drop() {
    let pubsub: PubSub<&str, i32> = PubSub::new();
    drop(pubsub.subscribe("gone"));

    assert_eq!(pubsub.publish(&"gone", 1).unwrap(), 0);
    assert_eq!(pubsub.subscriber_count(&"gone"), 0);
}
