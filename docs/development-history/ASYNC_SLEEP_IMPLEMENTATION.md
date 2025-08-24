# Async Sleep Implementation in Moirai

## Overview

Moirai provides a zero-dependency async sleep implementation that allows non-blocking delays in async contexts without requiring external crates like `tokio` or `async-std`.

## Problem Statement

Using `std::thread::sleep()` in an async context is problematic because:
1. It blocks the entire thread, preventing other async tasks from running
2. It defeats the purpose of async programming (cooperative multitasking)
3. It can cause deadlocks or severe performance degradation

## Solution: Custom Async Timer

### Architecture

```rust
pub mod timer {
    // Core components:
    // 1. Delay - A Future that completes after a duration
    // 2. Timer - Global timer instance managing all delays
    // 3. TimerEntry - Individual timer entries in a min-heap
}
```

### Key Design Decisions

1. **Zero Dependencies**: Uses only `std` library features
   - `std::sync::OnceLock` for global state (instead of `lazy_static`)
   - `std::collections::BinaryHeap` for timer queue
   - Single background thread for timer management

2. **Efficient Timer Wheel**: 
   - Min-heap (priority queue) for O(log n) insertion/removal
   - Single thread processes all timers
   - Wakes only when timers expire

3. **Minimal Overhead**:
   - Background thread sleeps between timer expirations
   - Wakers are called only when timers expire
   - No busy-waiting or polling

### Implementation Details

#### Delay Future
```rust
pub struct Delay {
    deadline: Instant,
    registered: bool,
    waker: Option<Waker>,
}

impl Future for Delay {
    type Output = ();
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if Instant::now() >= self.deadline {
            Poll::Ready(())
        } else {
            // Register with global timer
            get_timer().register(self.deadline, cx.waker().clone());
            Poll::Pending
        }
    }
}
```

#### Global Timer Management
```rust
static TIMER: OnceLock<Timer> = OnceLock::new();

pub struct Timer {
    timers: Arc<Mutex<BinaryHeap<TimerEntry>>>,
    thread: Option<JoinHandle<()>>,
    shutdown: Arc<Mutex<bool>>,
}
```

The timer thread:
1. Checks for expired timers
2. Wakes their associated tasks
3. Sleeps until the next timer expires
4. Checks for shutdown every 100ms

### Usage Examples

#### Basic Sleep
```rust
use moirai::sleep;
use std::time::Duration;

async fn example() {
    println!("Starting...");
    sleep(Duration::from_secs(1)).await;
    println!("One second later!");
}
```

#### Concurrent Sleeps
```rust
async fn concurrent_example() {
    let task1 = async {
        sleep(Duration::from_millis(100)).await;
        println!("Task 1 done");
    };
    
    let task2 = async {
        sleep(Duration::from_millis(200)).await;
        println!("Task 2 done");
    };
    
    // Both run concurrently
    join!(task1, task2);
}
```

#### With Iterators
```rust
use moirai_iter::moirai_iter_async;

let results = moirai_iter_async(items)
    .map(|item| async move {
        // Non-blocking delay
        sleep(Duration::from_millis(100)).await;
        process(item)
    })
    .collect::<Vec<_>>()
    .await;
```

## Performance Characteristics

- **Timer Registration**: O(log n) where n is number of active timers
- **Timer Expiration**: O(log n) for each expired timer
- **Memory Usage**: O(n) for n active timers
- **Thread Overhead**: 1 background thread
- **Accuracy**: ~1-10ms depending on system timer resolution

## Comparison with Alternatives

### vs `std::thread::sleep()`
- ✅ Non-blocking
- ✅ Allows concurrent task execution
- ✅ Proper async/await integration
- ❌ Slightly less accurate timing

### vs `tokio::time::sleep()`
- ✅ Zero external dependencies
- ✅ Simpler implementation
- ❌ Less sophisticated timer wheel
- ❌ No timer coalescing optimizations

### vs Busy-waiting
- ✅ Minimal CPU usage
- ✅ Scales to many timers
- ✅ Thread yields properly
- ❌ Requires one background thread

## Future Improvements

1. **Timer Coalescing**: Group nearby timers to reduce wakeups
2. **Hierarchical Timer Wheels**: Better performance for many timers
3. **Platform-specific Optimizations**: Use epoll/kqueue timers
4. **Sub-millisecond Precision**: High-resolution timer support

## Conclusion

This implementation provides a production-ready async sleep function that:
- Maintains Moirai's zero-dependency philosophy
- Enables proper async programming patterns
- Integrates seamlessly with the existing runtime
- Provides good performance for typical use cases

The design follows Rust's async principles while keeping the implementation simple and maintainable.