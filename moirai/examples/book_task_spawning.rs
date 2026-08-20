//! Spawn work-stealing tasks and collect results.
//!
//! [`Moirai`] is a hybrid async+parallel runtime built on a work-stealing
//! scheduler.  This example creates a runtime, spawns independent compute
//! tasks, and demonstrates that results arrive in any order because the
//! work-stealing scheduler assigns tasks to idle workers dynamically.

use moirai::{Moirai, Priority};

fn main() {
    let runtime = Moirai::new().expect("failed to create Moirai runtime");

    // Spawn three independent compute tasks.  The scheduler may run them on
    // any available worker thread.
    let h1 = runtime.spawn_fn(|| (0_u64..1_000).sum::<u64>());
    let h2 = runtime.spawn_fn(|| (0_u64..1_000).map(|n| n * n).sum::<u64>());
    let h3 = runtime.spawn_fn(|| (1_u64..=10).product::<u64>());

    let sum = h1.join().expect("join").expect("task");
    let sumsq = h2.join().expect("join").expect("task");
    let fact = h3.join().expect("join").expect("task");

    println!("sum(0..1000)    = {sum}"); // 499500
    println!("sumsq(0..1000)  = {sumsq}"); // 332833500
    println!("10!             = {fact}"); // 3628800

    assert_eq!(sum, 499_500);
    assert_eq!(sumsq, 332_833_500);
    assert_eq!(fact, 3_628_800);

    // Priority hint: high-priority tasks are scheduled before normal tasks
    // when the ready queue is non-empty.
    let high = runtime.spawn_fn_with_priority(|| 1_u32, Priority::High);
    let low = runtime.spawn_fn_with_priority(|| 0_u32, Priority::Low);

    let h = high.join().expect("join").expect("task");
    let l = low.join().expect("join").expect("task");
    println!("high={h}, low={l}");
    assert_eq!(h, 1);
    assert_eq!(l, 0);

    println!("all task-spawning assertions passed");
}
