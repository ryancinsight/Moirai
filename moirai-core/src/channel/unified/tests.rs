use super::*;
use crate::channel::config::ChannelConfig;
use crate::channel::error::ChannelError;

#[test]
fn test_unified_channel_basic() {
    let (sender, receiver) = unified_channel::<i32>(16).unwrap();

    // Test basic send/receive
    sender.send(42).unwrap();
    assert_eq!(receiver.recv().unwrap(), 42);

    // Test try operations
    assert!(sender.try_send(100).is_ok());
    assert_eq!(receiver.try_recv().unwrap(), 100);
}

#[test]
fn test_unified_channel_batch() {
    let config = ChannelConfig {
        capacity: 64,
        enable_batching: true,
        batch_size: 10,
        ..Default::default()
    };

    let (sender, receiver) = unified_channel_with_config::<i32>(config).unwrap();

    // Test batch send
    let messages = vec![1, 2, 3, 4, 5];
    let sent = sender.send_batch(messages).unwrap();
    assert_eq!(sent, 5);

    // Test batch receive
    let received = receiver.recv_batch(10);
    assert_eq!(received, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_unified_channel_stats() {
    let (sender, receiver) = unified_channel::<i32>(16).unwrap();

    // Send some messages
    for i in 0..5 {
        sender.send(i).unwrap();
    }

    // Receive some messages
    for _ in 0..3 {
        receiver.recv().unwrap();
    }

    let stats = receiver.stats();
    assert_eq!(stats.messages_sent, 5);
    assert_eq!(stats.messages_received, 3);
    assert_eq!(stats.current_length, 2);
}

#[test]
fn test_unified_channel_close() {
    let (sender, receiver) = unified_channel::<i32>(16).unwrap();

    // Send a message
    sender.send(42).unwrap();

    // Close the channel
    receiver.channel.close();

    // Channel should still allow receiving existing messages
    assert_eq!(receiver.recv().unwrap(), 42);
    assert!(receiver.recv().is_err());
}

#[test]
fn test_unified_channel_adaptive_overflow_fifo() {
    // Capacity 2 rounded to power of two is 2. The max stored in ring buffer is 1.
    // Overflow queue capacity is 4.
    let config = ChannelConfig {
        capacity: 2,
        enable_pooling: true,
        max_pool_size: 4,
        ..Default::default()
    };

    let (sender, receiver) = unified_channel_with_config::<i32>(config).unwrap();

    // Push 5 items (1 goes to ring buffer, 4 go to overflow queue)
    for i in 1..=5 {
        sender.send(i).unwrap();
    }

    // Attempting to push 6th item should fail (both full)
    assert!(sender.send(6).is_err());

    // Verify FIFO ordering of all 5 items
    for expected in 1..=5 {
        assert_eq!(receiver.recv().unwrap(), expected);
    }

    // Now empty
    assert!(receiver.recv().is_err());
}

#[test]
fn try_send_returns_value_on_full_send_consumes() {
    // Ring capacity 1, no overflow pool: the channel holds exactly one item.
    let config = ChannelConfig {
        capacity: 1,
        enable_pooling: false,
        max_pool_size: 0,
        ..Default::default()
    };
    let (sender, receiver) = unified_channel_with_config::<i32>(config).unwrap();

    sender.send(1).unwrap();

    // `try_send` hands the rejected value back so the caller can retry it.
    let err = sender.try_send(2).expect_err("channel is full");
    assert_eq!(err, (2, ChannelError::Full));

    // `send` surfaces the same failure but consumes the value (documented).
    assert_eq!(sender.send(3), Err(ChannelError::Full));

    // The one buffered item is still intact and retrievable.
    assert_eq!(receiver.recv().unwrap(), 1);
    assert!(receiver.recv().is_err());
}
