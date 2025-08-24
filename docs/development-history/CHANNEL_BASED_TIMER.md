# Channel-Based Async Timer Implementation

## Overview

The Moirai async timer has been refactored to use Moirai's high-performance MPMC channels instead of mutex-based shared state. This provides better performance characteristics and cleaner separation of concerns.

## Architecture Changes

### Previous Implementation (Mutex-Based)
- Used `Arc<Mutex<BinaryHeap<TimerEntry>>>` for shared timer queue
- Background thread held mutex lock while processing timers
- Potential contention between timer registration and processing

### New Implementation (Channel-Based)
- Uses Moirai MPMC channel for timer commands
- Background thread receives commands via non-blocking `try_recv`
- No mutex contention - clean producer/consumer pattern

## Key Benefits

1. **Better Performance**
   - Non-blocking channel operations
   - No mutex contention under high load
   - Batch processing of timer commands
   - Wakers called outside critical sections

2. **Cleaner Architecture**
   - Clear separation between async runtime and timer thread
   - Command pattern for timer operations
   - Simpler shutdown mechanism

3. **Non-Blocking Guarantees**
   - All channel operations use `try_recv`/`try_send`
   - Async runtime never blocks on timer operations
   - Graceful handling of channel overflow

## Implementation Details

### Timer Commands
```rust
enum TimerCommand {
    Register { deadline: Instant, waker: Waker },
    Shutdown,
}
```

### Timer Loop
```rust
loop {
    // Process all pending commands (non-blocking)
    while let Ok(cmd) = receiver.try_recv() {
        match cmd {
            TimerCommand::Register { deadline, waker } => {
                timers.push(TimerEntry { deadline, waker });
            }
            TimerCommand::Shutdown => {
                return;
            }
        }
    }
    
    // Process expired timers
    let now = Instant::now();
    let mut wakers_to_wake = Vec::new();
    
    while let Some(entry) = timers.peek() {
        if entry.deadline <= now {
            let entry = timers.pop().unwrap();
            wakers_to_wake.push(entry.waker);
        } else {
            break;
        }
    }
    
    // Wake all expired timers outside of the critical section
    for waker in wakers_to_wake {
        waker.wake();
    }
    
    // Sleep until next timer or 10ms
    let sleep_duration = calculate_sleep_duration(&timers);
    thread::sleep(sleep_duration);
}
```

## Performance Characteristics

- **Timer Registration**: O(1) channel send + O(log n) heap insertion
- **Timer Processing**: O(k log n) where k is number of expired timers
- **Memory Usage**: O(n) for n active timers + fixed channel buffer
- **Latency**: 10ms maximum delay for timer expiration
- **Throughput**: Limited by channel capacity (1024 pending timers)

## Trade-offs

1. **10ms Polling Interval**
   - Timers may fire up to 10ms late
   - Acceptable for most async use cases
   - Could be made configurable if needed

2. **Fixed Channel Capacity**
   - 1024 pending timer registrations
   - Excess registrations are dropped
   - Suitable for typical workloads

3. **Memory Allocation**
   - Wakers are collected in a Vec before waking
   - Prevents holding locks while calling wakers
   - Small overhead for better concurrency

## Future Improvements

1. **Adaptive Polling**
   - Adjust sleep duration based on timer density
   - Wake immediately when many timers are pending

2. **Timer Coalescing**
   - Group timers with similar deadlines
   - Reduce wake-ups for better efficiency

3. **Hierarchical Timer Wheels**
   - Multiple resolution levels
   - Better performance for long-duration timers

## Conclusion

The channel-based timer implementation provides:
- ✅ Better performance under concurrent load
- ✅ Non-blocking guarantees for async runtime
- ✅ Clean separation of concerns
- ✅ Simpler and more maintainable code

This aligns with Moirai's design principles of high performance, zero-cost abstractions, and clean architecture.