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
    assert_eq!(scattered.len(), 3);
    assert_eq!(scattered[0], vec![1, 2]);

    let gathered = CollectiveOps::gather(scattered);
    assert_eq!(gathered, vec![1, 2, 3, 4, 5, 6]);
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
